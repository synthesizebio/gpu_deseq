"""Empirical roofline (no perf counters needed).

1. Measure this GPU's sustainable HBM bandwidth (streaming triad).
2. Measure achieved bandwidth of the actual NR-loop kernels at (G,S).
   - a plain elementwise op (the bulk of the loop) -> expect near-peak BW = memory-bound
   - a transcendental (lgamma/digamma) -> if BW << peak, it's compute-bound
3. Measure the full _lp_and_dlp and report its effective bandwidth vs peak.

If the loop runs near peak BW, it's memory-bandwidth-bound and kernel fusion
(Triton) would help by removing HBM round-trips. If it runs far below peak BW
but is still slow, it's compute-bound (transcendentals) and fusion helps less.

Run with --json to commit the measurements; the paper cites this file and
scripts/audit_paper_numbers.py checks the cited values against it.
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
import torch
import gpu_deseq._deseq2_core as core

dev = "cuda"
GB = 1024**3
PEAK_SPEC = 1555.0  # A100-SXM4-40GB spec HBM2e GB/s

ap = argparse.ArgumentParser()
ap.add_argument("--json", type=str, default=None)
_args = ap.parse_args()


def cuda_time_ms(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).decode().strip())
    except Exception:                                             # noqa: BLE001
        commit, dirty = "unknown", False
    return dict(
        utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        cuda=torch.version.cuda, torch=torch.__version__,
        python=sys.version.split()[0], platform=platform.platform(),
        commit=commit, working_tree_dirty=dirty,
        peak_spec_gbs=PEAK_SPEC,
    )


payload = {"provenance": provenance(), "cases": []}

print(f"A100 spec peak HBM: {PEAK_SPEC:.0f} GB/s\n")

# --- 1. sustainable bandwidth: triad c = a + s*b on large fp64 tensors ---
N = 64 * 1024 * 1024  # 64M elements
a = torch.randn(N, device=dev, dtype=torch.float64)
b = torch.randn(N, device=dev, dtype=torch.float64)
t = cuda_time_ms(lambda: torch.add(a, b, alpha=2.0))  # reads a,b writes out = 3 arrays
bytes_triad = 3 * N * 8
bw_triad = bytes_triad / (t / 1e3) / GB
# PEAK_SPEC is vendor-quoted decimal GB/s, so the comparison and everything
# derived from it must be decimal too. Printing the GiB/s figure next to a
# decimal percentage (as this line used to) reads as a 15% discrepancy that
# isn't there.
sustain = bw_triad * GB / 1e9  # GiB/s -> decimal GB/s
print(f"[peak probe] fp64 add (3 arrays x {N/1e6:.0f}M): {t:.3f} ms -> "
      f"{sustain:.0f} GB/s ({bw_triad:.0f} GiB/s), "
      f"{100*sustain/PEAK_SPEC:.0f}% of spec\n")
payload["sustained_gbs"] = round(sustain, 1)
payload["sustained_pct_of_spec"] = round(100 * sustain / PEAK_SPEC, 1)

# --- 2. actual NR-loop tensors at (G, S) ---
for (G, S) in [(20000, 60), (2000, 60)]:
    mu = torch.rand(G, S, device=dev, dtype=torch.float64) * 100 + 0.5
    counts = torch.randint(0, 500, (G, S), device=dev).double()
    a_col = torch.full((G, 1), 0.3, device=dev, dtype=torch.float64)
    ainv = 1.0 / a_col
    gs_bytes = G * S * 8

    # (a) plain elementwise: w = 1/(1/mu + a)  -> read mu, write w = 2 arrays
    t_w = cuda_time_ms(lambda: 1.0 / (1.0 / mu + a_col))
    bw_w = (2 * gs_bytes) / (t_w / 1e3) / 1e9
    # (b) transcendental: lgamma(counts + ainv) -> read counts, write = 2 arrays
    t_lg = cuda_time_ms(lambda: torch.lgamma(counts + ainv))
    bw_lg = (2 * gs_bytes) / (t_lg / 1e3) / 1e9
    # (c) digamma
    t_dg = cuda_time_ms(lambda: torch.digamma(counts + ainv))
    bw_dg = (2 * gs_bytes) / (t_dg / 1e3) / 1e9

    print(f"(G={G}, S={S}):")
    print(f"  elementwise  1/(1/mu+a) : {t_w*1e3:6.1f} us  {bw_w:6.0f} GB/s  "
          f"({100*bw_w/sustain:3.0f}% of sustained)  -> {'MEMORY-bound' if bw_w/sustain>0.6 else 'not BW-saturated'}")
    print(f"  lgamma                  : {t_lg*1e3:6.1f} us  {bw_lg:6.0f} GB/s  "
          f"({100*bw_lg/sustain:3.0f}% of sustained)  -> {'COMPUTE-bound' if bw_lg/sustain<0.3 else 'mixed'}")
    print(f"  digamma                 : {t_dg*1e3:6.1f} us  {bw_dg:6.0f} GB/s  "
          f"({100*bw_dg/sustain:3.0f}% of sustained)  -> {'COMPUTE-bound' if bw_dg/sustain<0.3 else 'mixed'}")

    # (d) full _lp_and_dlp: time it, report effective BW vs an ESSENTIAL-traffic
    #     lower bound (must read counts+mu at least once = 2 arrays).
    design = torch.tensor([[1.0, 0.0]] * (S // 2) + [[1.0, 1.0]] * (S - S // 2),
                          device=dev, dtype=torch.float64)
    la = torch.full((G,), -1.0, device=dev, dtype=torch.float64)
    t_lp = cuda_time_ms(lambda: core._lp_and_dlp(counts, mu, design, la), iters=100, warmup=20)
    essential_bw = (2 * gs_bytes) / (t_lp / 1e3) / 1e9  # if it only moved counts+mu once
    print(f"  FULL _lp_and_dlp        : {t_lp*1e3:6.1f} us   "
          f"(essential-traffic BW = {essential_bw:5.0f} GB/s = {100*essential_bw/sustain:.0f}% of sustained)")
    print(f"      -> effective BW {'>' if essential_bw>sustain else '<='} sustained means it moves "
          f"{'MORE' if essential_bw>sustain else 'about the essential'} data (unfused intermediates)\n")

    payload["cases"].append(dict(
        G=G, S=S,
        elementwise_us=round(t_w * 1e3, 1),
        elementwise_gbs=round(bw_w, 1),
        elementwise_pct_sustained=round(100 * bw_w / sustain, 1),
        lgamma_us=round(t_lg * 1e3, 1),
        lgamma_gbs=round(bw_lg, 1),
        lgamma_pct_sustained=round(100 * bw_lg / sustain, 1),
        digamma_us=round(t_dg * 1e3, 1),
        digamma_gbs=round(bw_dg, 1),
        digamma_pct_sustained=round(100 * bw_dg / sustain, 1),
        lp_and_dlp_us=round(t_lp * 1e3, 1),
        lp_and_dlp_essential_gbs=round(essential_bw, 1),
        lp_and_dlp_pct_sustained=round(100 * essential_bw / sustain, 1),
    ))

if _args.json:
    with open(_args.json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {_args.json}")
