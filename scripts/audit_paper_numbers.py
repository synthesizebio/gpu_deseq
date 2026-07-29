"""Audit every numeric claim in paper/main.tex against a committed data file.

Read-only: parses main.tex and the committed JSON results. Runs no experiments.

Two jobs:
  1. Every checked claim must equal its source value.
  2. Every \fact{} claim must be REGISTERED as either checked here or traced to
     a named artifact. An unbacked number is a failure, not a silent pass --
     the whole point is that a claim with no logged provenance cannot hide.

Registered-but-unchecked entries carry the artifact that would substantiate
them; UNBACKED entries have no committed source and must not ship.
"""
import json, math, re, pathlib, statistics

tex = pathlib.Path("paper/main.tex").read_text()
P = json.load(open("bench/results/parity.json"))
T = json.load(open("bench/results/timings.json"))
S = json.load(open("benchmarks/results_per_step.json"))
V = {n: json.load(open(f"validation/results/{n}.json")) for n in
     ["pasilla", "pasilla_2fac", "airway_dex", "airway_cell", "airway",
      "gtex_blood_muscle"]}
# Second CPU baseline. summary.json is the derived view; r_scaling.jsonl is the
# authoritative harness output, so sweep-level claims are checked against that.
E = json.load(open("bench/results/r_epyc9b14/summary.json"))
EH = "EPYC 9B14"
_erecs = [json.loads(l) for l in
          open("bench/results/r_epyc9b14/r_scaling.jsonl")]
ESUB = {r["case"]: r for r in _erecs if r["kind"] == "serial_fine"}
ESWEEP = [r for r in _erecs if r["kind"] == "sweep"]
# The host-B column is not a controlled A/B against the reference (different
# DESeq2), and the paper must keep saying so. Assert the flag the harness set.
assert E["hosts"][EH]["comparable_to_reference"] is False, \
    "host B became comparable to the reference: update the paper's caveat"

fails, checks = [], 0


def chk(label, ok, got=None, want=None):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{label}: paper={want!r} data={got!r}")


def close(a, b, tol=0.03):
    return abs(a - b) <= tol * abs(b)


# ---- median- and worst-of-six per substep (Table 2 + prose) ----------------
vals, worst, worst_case, median = {}, {}, {}, {}
for c, rows in P.items():
    for r in rows:
        k, v = r["substep"], r["value"]
        vals.setdefault(k, []).append(v)
        better = (v < worst[k]) if (k in worst and r["higher_better"]) else \
                 (v > worst[k]) if k in worst else True
        if better:
            worst[k], worst_case[k] = v, c
for k, vv in vals.items():
    median[k] = statistics.median(vv)

# Table 2 no longer carries a pass/fail column: it was implied by "worst beats
# tolerance" and constant down every row. The verdict the prose states instead
# ("All 30 substep checks pass") is asserted here.
allrows = [r for rows in P.values() for r in rows]
chk("prose: 30 substep checks", len(allrows) == 30, len(allrows), 30)
chk("prose: all substep checks pass", all(r["pass"] for r in allrows),
    sorted(f"{r['case']}/{r['substep']}" for r in allrows if not r["pass"]),
    "all pass")

t2 = re.search(r"label\{tab:parity\}.*?end\{tabular\}", tex, re.S).group()
for sub in ("normalization", "dispersion", "glm_fit", "significance",
            "lfc_shrink"):
    lbl = sub.replace("_", r"\_")
    row = [l for l in t2.splitlines() if l.startswith(lbl + " &")][0]
    cells = [c.strip() for c in row.split("&")]
    med, wst = float(cells[-2]), float(cells[-1].split(r"\\")[0])
    chk(f"T2 {sub} median", close(med, median[sub], 0.05), median[sub], med)
    chk(f"T2 {sub} worst", close(wst, worst[sub], 0.05), worst[sub], wst)

chk("T2 dispersion worst case is gtex",
    worst_case["dispersion"] == "gtex_blood_muscle", worst_case["dispersion"], "gtex")
chk("prose: all six dispersions < 0.4%", worst["dispersion"] < 0.004,
    worst["dispersion"], "<0.004")
chk("prose: gtex dispersion 3.7e-3", close(3.7e-3, worst["dispersion"]),
    worst["dispersion"], 3.7e-3)

aw = {r["substep"]: r for r in P["airway"]}
chk("prose: airway dispersion 4.5e-4", close(4.5e-4, aw["dispersion"]["value"]),
    aw["dispersion"]["value"], 4.5e-4)
chk("prose: airway raw LFC 3.0e-5", close(3.0e-5, aw["glm_fit"]["value"]),
    aw["glm_fit"]["value"], 3.0e-5)
chk("prose: airway Jaccard 1.000", aw["significance"]["value"] == 1.0,
    aw["significance"]["value"], 1.0)
chk("prose: apeGLM r >= 0.997 everywhere", worst["lfc_shrink"] >= 0.997,
    worst["lfc_shrink"], ">=0.997")

# shrunk-LFC p95 |D| bound
mx = max(v["shrunk_lfc_p95_abs"] for v in V.values())
m = re.search(r"with p95 \$\|\\Delta\|\\le\$([\d.e-]+)\.", tex)
chk("prose: shrunk p95 bound", float(m.group(1)) >= mx, mx, float(m.group(1)))

# Triton-vs-eager dispersion over the five small-n sets
small = [c for c in P if c != "gtex_blood_muscle"]
mx_small = max(next(r for r in P[c] if r["substep"] == "dispersion")["gpu_modes_maxdiff"]
               for c in small)
m = re.search(r"it agrees with eager to \$\\le\$([\d.e-]+) in", tex)
chk("prose: Triton small-n dispersion bound", float(m.group(1)) >= mx_small,
    mx_small, float(m.group(1)))
g = {r["substep"]: r["gpu_modes_maxdiff"] for r in P["gtex_blood_muscle"]}
chk("prose: gtex Triton dispersion 2.6e-1", close(0.26, g["dispersion"]),
    g["dispersion"], 0.26)
chk("prose: gtex Triton shrunk LFC 5.7e-3", close(5.7e-3, g["lfc_shrink"]),
    g["lfc_shrink"], 5.7e-3)

# ---- Table 3 + speedups ----------------------------------------------------
t3 = re.search(r"label\{tab:realtiming\}.*?Triton is the best mode", tex, re.S).group()
LBL = [("pasilla", "pasilla"), ("pasilla_2fac", r"pasilla\_2fac"),
       ("airway_dex", r"airway\_dex"), ("airway_cell", r"airway\_cell"),
       ("airway", "airway"), ("gtex_blood_muscle", "gtex")]
best = {}
# Columns are indexed from the FRONT (P, n, hostA, hostB serial, hostB best MC,
# eager, graph, triton), not the back: the table gained R columns once a second
# CPU host was measured, and back-indexing silently reads the wrong one.
for c, lbl in LBL:
    row = [l for l in t3.splitlines() if l.strip().startswith(lbl + " ")][0]
    nums = re.findall(r"(?<![\w.])(\d+\.?\d*)(?![\w.])", row)
    chk(f"T3 {c} column count", len(nums) == 8, len(nums), 8)
    rsA, rsB, rsBmc = (float(nums[2]), float(nums[3]), float(nums[4]))
    e, gph, tri = (float(x) for x in nums[5:8])
    # The R columns print one decimal, so compare the correctly-rounded value
    # rather than a relative tolerance: at 3.6 s a 1% band is tighter than the
    # displayed precision and rejects a correctly-rounded cell.
    for nm, shown, want_ms in (
            ("host A", rsA, T["r"][c]["total"]),
            ("host B serial", rsB, E["datasets"][c]["r_serial_ms"][EH]),
            ("host B best multicore", rsBmc,
             E["datasets"][c]["r_multicore_ms"][EH]["best_ms"])):
        chk(f"T3 {c} R(s) {nm}", round(shown, 1) == round(want_ms / 1000, 1),
            round(want_ms / 1000, 1), shown)
    for nm, shown, key in (("eager", e, "eager"), ("graph", gph, "graph"),
                           ("triton", tri, "triton")):
        chk(f"T3 {c} {nm}", round(shown) == round(T["cu"][c][key]["total"]),
            T["cu"][c][key]["total"], shown)
    best[c] = T["r"][c]["total"] / min(T["cu"][c][m2]["total"] for m2 in
                                       ("eager", "graph", "triton"))

lo, hi = min(best.values()), max(best.values())
m = re.search(r"\\fact\{(\d+)--(\d+)\$\\times\$\} faster end-to-end", tex)
chk("abstract speedup range", int(m.group(1)) <= lo and round(hi) == int(m.group(2)),
    f"{lo:.1f}-{hi:.1f}", m.group(0))
m = re.search(r"driving the highest end-to-end speedup, (\d+\.\d)\$\\times\$", tex)
chk("prose: highest speedup", close(float(m.group(1)), hi, 0.01), round(hi, 1),
    float(m.group(1)))

# ---- second CPU baseline (host B) ------------------------------------------
# Every speedup range in the paper must name its denominator, so each is checked
# against the host it is claimed for. summary.json is derived from r_scaling.jsonl,
# so its recomputed speedups are re-derived here rather than trusted: a stale
# summary would otherwise agree with a stale paper.
def rng(vals):
    # Round half away from zero, not Python's half-to-even: summary.json stores
    # speedups at 2 dp, and formatting its exact-binary 4.25 with %.1f yields
    # "4.2" while the underlying ratio is 4.2535, i.e. "4.3".
    def r1(v):
        return f"{math.floor(v * 10 + 0.5) / 10:.1f}"
    return f"{r1(min(vals))}--{r1(max(vals))}"


# Re-derived from the raw millisecond fields rather than read from
# speedup_recomputed, so the range cannot inherit that field's rounding.
spd = {
    f"vs_{EH}_serial": [d["r_serial_ms"][EH] / d["gpu_best_a100"]["total_ms"]
                        for d in E["datasets"].values()],
    f"vs_{EH}_best_multicore": [d["r_multicore_ms"][EH]["best_ms"] /
                                d["gpu_best_a100"]["total_ms"]
                                for d in E["datasets"].values()],
}
for c, d in E["datasets"].items():
    gpu = d["gpu_best_a100"]["total_ms"]
    chk(f"hostB {c} GPU best matches timings.json",
        close(gpu, min(T["cu"][c][m2]["total"] for m2 in ("eager", "graph", "triton")), 1e-6),
        gpu, min(T["cu"][c][m2]["total"] for m2 in ("eager", "graph", "triton")))
    chk(f"hostB {c} serial speedup re-derived",
        close(d["speedup_recomputed"][f"vs_{EH}_serial"], d["r_serial_ms"][EH] / gpu, 0.01),
        d["r_serial_ms"][EH] / gpu, d["speedup_recomputed"][f"vs_{EH}_serial"])
    chk(f"hostB {c} multicore speedup re-derived",
        close(d["speedup_recomputed"][f"vs_{EH}_best_multicore"],
              d["r_multicore_ms"][EH]["best_ms"] / gpu, 0.01),
        d["r_multicore_ms"][EH]["best_ms"] / gpu,
        d["speedup_recomputed"][f"vs_{EH}_best_multicore"])

chk("T3 footnote range vs host A", rng(best.values()) == "8.0--78.2",
    rng(best.values()), "8.0--78.2")
chk("T3 footnote range vs host B serial",
    rng(spd[f"vs_{EH}_serial"]) == "4.3--32.8", rng(spd[f"vs_{EH}_serial"]), "4.3--32.8")
chk("T3 footnote range vs host B multicore",
    rng(spd[f"vs_{EH}_best_multicore"]) == "1.3--8.8",
    rng(spd[f"vs_{EH}_best_multicore"]), "1.3--8.8")

# per-core ratio between the two hosts, and the gtex dispersion it comes from
ratios = [E["datasets"][c]["r_serial_ms"]["A100 host"] /
          E["datasets"][c]["r_serial_ms"][EH] for c in E["datasets"]]
chk("prose: host A/B per-core ratio 1.9--2.4", rng(ratios) == "1.9--2.4",
    rng(ratios), "1.9--2.4")
gdisp_b = ESUB["gtex_blood_muscle"]["dispersion"] / 1000
chk("prose: hostB gtex dispersion 81 s", round(gdisp_b) == 81, gdisp_b, 81)

# Amdahl remainder: results() is flat across the sweep while the fit scales.
gsw = {r["workers"]: r for r in ESWEEP if r["case"] == "gtex_blood_muscle"}
res = [gsw[w]["results"] / 1000 for w in sorted(gsw)]
chk("prose: gtex results() flat ~11 s", all(10.5 <= v <= 12.5 for v in res),
    [round(v, 1) for v in res], "all in 10.5-12.5")
fit_scale = gsw[min(gsw)]["deseq_fit"] / gsw[max(gsw)]["deseq_fit"]
chk("prose: gtex deseq_fit scales 7.9x", close(7.9, fit_scale, 0.01),
    round(fit_scale, 1), 7.9)
chk("prose: gtex multicore speedup 8.8x",
    close(8.8, E["datasets"]["gtex_blood_muscle"]["speedup_recomputed"][
        f"vs_{EH}_best_multicore"], 0.01),
    E["datasets"]["gtex_blood_muscle"]["speedup_recomputed"][f"vs_{EH}_best_multicore"], 8.8)

# airway_cell is shrink-bound, not dispersion-bound -- the reason its row is thin.
ac = T["cu"]["airway_cell"]["triton"]
chk("prose: airway_cell 94% lfc_shrink",
    round(100 * ac["lfc_shrink"] / ac["total"]) == 94,
    100 * ac["lfc_shrink"] / ac["total"], 94)

# ---- Table 4 (every cell) --------------------------------------------------
t4 = re.search(r"label\{tab:realsubstep\}.*?end\{tabular\}\}", tex, re.S).group()
for c, lbl in LBL:
    blk = t4.split(lbl + " & ")[1].split(r"\midrule")[0] if lbl + " & " in t4 else None
    for sub in ("dispersion", "glm_fit", "significance", "lfc_shrink"):
        nm = sub.replace("_", r"\_")
        m = re.search(re.escape(nm) + r" & (\d+) & (\d+) & (\d+) & (\d+)", t4)
        rows = [l for l in t4.splitlines() if nm + " &" in l]
        # match the row belonging to this dataset block
        idx = [i for i, (cc, ll) in enumerate(LBL) if cc == c][0]
        vals = [int(x) for x in re.findall(r"& (\d+)", rows[idx])]
        want = [round(T["r"][c][sub])] + [round(T["cu"][c][m2][sub])
                                          for m2 in ("eager", "graph", "triton")]
        chk(f"T4 {c} {sub}", vals == want, want, vals)

# ---- Table 5 ---------------------------------------------------------------
t5 = re.search(r"label\{tab:perstep\}.*?end\{tabular\}", tex, re.S).group()
tab = S["tables"]["60x20000_cond"]
for r in tab["rows"]:
    nm = r[0].replace("_", r"\_")
    row = [l for l in t5.splitlines() if l.startswith(nm + " &")][0]
    vals = re.findall(r"& ([\d(]+[\w)]*)", row)
    got = [round(r[1])] + [round(v) for v in r[3:]]
    shown = [int(vals[0])] + [int(v) for v in vals[2:5]]
    chk(f"T5 {r[0]}", shown == got, got, shown)
tot = tab["total"]
shown = [int(x) for x in re.findall(r"\\textbf\{(\d+)\}", t5)]
want = [round(tot[k]) for k in ("R_1thread", "R_multicore", "eager", "graph", "triton")]
chk("T5 TOTAL", shown == want, want, shown)
sp = []
for c in ["6x2000_cond", "60x2000_cond", "60x20000_cond", "60x1500_multi"]:
    t = S["tables"][c]["total"]
    sp.append(round(min(v for v in (t["R_1thread"], t["R_multicore"]) if v) / t["triton"], 1))
m = re.search(r"speedups of\n([\d./]+)\$\\times\$", tex)
chk("T5 footnote speedups", m.group(1) == "/".join(f"{x:.1f}" for x in sp),
    "/".join(f"{x:.1f}" for x in sp), m.group(1))

# ---- gtex dispersion 231 s -> 0.2 s ---------------------------------------
gd = T["r"]["gtex_blood_muscle"]["dispersion"] / 1000
gt = T["cu"]["gtex_blood_muscle"]["triton"]["dispersion"] / 1000
chk("abstract/prose 231s->0.2s", close(231, gd, 0.01) and close(0.2, gt, 0.05),
    f"{gd:.0f}s->{gt:.1f}s", "231s->0.2s")

# ---- graph-vs-eager: 23 configs, all exactly 0 ----------------------------
ncfg = nzero = 0
for f in ("results_a100", "results_a100_samplesweep", "results_a100_sweep"):
    d = json.load(open(f"benchmarks/{f}.json"))
    key = [k for k in d if k != "provenance"][0]
    for rec in d[key]:
        v = rec.get("max_abs_diff")
        if isinstance(v, dict):
            ncfg += 1; nzero += all(x == 0 for x in v.values())
        elif v is not None:
            ncfg += 1; nzero += (v == 0)
        for sub in rec.get("sweep", []):
            ncfg += 1; nzero += (sub["max_abs_diff"] == 0)
m = re.search(r"Across the \\fact\{(\d+)\} synthetic dispersion", tex)
chk("prose: 23 graph configs", int(m.group(1)) == ncfg, ncfg, int(m.group(1)))
chk("prose: graph max|D| = 0 on all of them", nzero == ncfg,
    f"{nzero}/{ncfg} zero", "all zero")

# ---- Sec 4.2.1: the standard-DE-plot figures (Figs 3-4) --------------------
# These are scored against the DESeq() wrapper reference in validation/, NOT the
# substep chain of Table 2 -- the two differ on gtex by replaceOutliers, which is
# the whole point of that subsection, so the distinction is asserted here too.
G = V["gtex_blood_muscle"]
chk("4.2.1 gtex Jaccard@.05 (DESeq wrapper ref)", close(0.9805, G["sig_jaccard_0.05"], 1e-3),
    G["sig_jaccard_0.05"], 0.9805)
chk("4.2.1 airway_dex Jaccard@.05", close(0.9996, V["airway_dex"]["sig_jaccard_0.05"], 1e-3),
    V["airway_dex"]["sig_jaccard_0.05"], 0.9996)
chk("4.2.1 gtex Jaccard (substep-chain ref)", close(0.99994, worst_sig_gtex := [
        r["value"] for r in P["gtex_blood_muscle"] if r["substep"] == "significance"][0], 1e-4),
    worst_sig_gtex, 0.99994)
chk("4.2.1 gtex n_sig R", G["n_sig_r_0.05"] == 33942, G["n_sig_r_0.05"], 33942)
chk("4.2.1 gtex n_sig ours", G["n_sig_ours_0.05"] == 33284, G["n_sig_ours_0.05"], 33284)
chk("4.2.1 airway_cell n_sig R", V["airway_cell"]["n_sig_r_0.05"] == 206,
    V["airway_cell"]["n_sig_r_0.05"], 206)
chk("4.2.1 airway n_sig R", V["airway"]["n_sig_r_0.05"] == 3993,
    V["airway"]["n_sig_r_0.05"], 3993)
# "four of the six agree exactly" at alpha=0.05
n_exact = sum(1 for v in V.values() if v["sig_jaccard_0.05"] == 1.0)
chk("4.2.1 four designs reproduce R's called set exactly", n_exact == 4, n_exact, 4)
# airway_cell and airway are the ones the minmu fix took to exact agreement
for c in ("airway_cell", "airway"):
    chk(f"4.2.1 {c} called set exact", V[c]["sig_jaccard_0.05"] == 1.0,
        V[c]["sig_jaccard_0.05"], 1.0)

# ---- provenance registry: every \fact{} must appear here ------------------
# value-pattern -> provenance. "CHECKED" = verified numerically above.
REGISTRY = {
    "76/76 step-parity tests; 30/30 real-data substep checks":
        "CHECKED (pytest --collect-only; parity.json row count)",
    "8--78$\\times$": "CHECKED (timings.json)",
    "231\\,s to 0.2\\,s": "CHECKED (timings.json)",

    # --- second CPU baseline, host B (EPYC 9B14). Every range names its
    # --- denominator: "vs R" is ambiguous once two hosts are reported.
    "8.0--78.2$\\times$": "CHECKED (timings.json; vs host A, 1 thread)",
    "4.3--32.8$\\times$":
        "CHECKED (r_epyc9b14/summary.json; vs host B, 1 thread, re-derived "
        "from r_serial_ms / gpu_best_a100)",
    "1.3--8.8$\\times$":
        "CHECKED (r_epyc9b14/summary.json; vs host B best multi-core, "
        "re-derived from r_multicore_ms.best_ms / gpu_best_a100)",
    "1.9--2.4$\\times$":
        "CHECKED (per-dataset host A / host B serial total, summary.json). "
        "NOT a clean hardware ratio -- DESeq2 1.30.1 vs 1.46.0 is confounded "
        "with the CPU; stated as such in the paper",
    "81\\,s": "CHECKED (r_epyc9b14/r_scaling.jsonl serial_fine gtex dispersion)",
    "$\\approx$11\\,s":
        "CHECKED (r_scaling.jsonl sweep: gtex results() spans 11.1-12.1 s "
        "over w=1..15, i.e. does not parallelise)",
    "7.9$\\times$": "CHECKED (r_scaling.jsonl sweep: gtex deseq_fit w=1 / w=15)",
    "8.8$\\times$": "CHECKED (summary.json gtex vs host B best multi-core)",
    "94\\%": "CHECKED (timings.json airway_cell triton lfc_shrink / total)",
    "95{,": "EXTERNAL (Google Scholar citation count for DESeq2)",
    "31": "loess kd-tree cuts; R loess(kd$xi) -- tests/test_loess.py",
    "5e-15": "loess kd cuts / KL curve vs R -- tests/test_loess.py, test_r_rng.py",
    "1e-14": "loess predict vs R -- tests/test_loess.py",
    "$0.5285285285285285$": "prior var bit-exact vs R -- tests/test_r_rng.py",
    "3e-15": "Triton-vs-eager log-posterior at fixed alpha, worst over covered "
             "P x S in {8,12,24} -- tests/test_triton_dispersion.py "
             "(committed bound 1e-13)",
    "3e-14": "Triton-vs-eager gradient at fixed alpha, same sweep -- "
             "tests/test_triton_dispersion.py (committed bound 1e-12)",
    "$0.016$": "0.5445445 (PCG64) - 0.5285285 (R replay)",
    "$0.080$": "0.6086086 (Savitzky-Golay) - 0.5285285",
    "0.045\\%": "CHECKED (parity.json airway dispersion)",
    "1.4\\%": "dispersion p95 at prior var 0.5445445",
    "6.7\\%": "dispersion p95 at prior var 0.6086086",
    "$\\approx$0.3\\,s": "compiled R-stream replay cost (warm _prior_var_kl_grid)",
    "0.4\\%": "CHECKED (parity.json dispersion worst)",
    "3.7e-3": "CHECKED (parity.json gtex dispersion)",
    "4.5e-4": "CHECKED (parity.json airway dispersion)",
    "3.0e-5": "CHECKED (parity.json airway glm_fit)",
    "1.000": "CHECKED (parity.json airway significance)",
    "23": "CHECKED (count of eager-vs-graph configs in benchmarks/results_a100*.json)",
    "0": "CHECKED (max eager-vs-graph max_abs_diff over those configs)",

    # --- Sec 4.2.1, the standard-DE-plot figures ---------------------------
    "0.9805": "CHECKED (validation/results/gtex_blood_muscle.json sig_jaccard_0.05)",
    "0.9996": "CHECKED (validation/results/airway_dex.json sig_jaccard_0.05)",
    "0.99994": "CHECKED (parity.json gtex significance, substep-chain reference)",
    "33{,": "CHECKED (gtex_blood_muscle.json n_sig_r_0.05 / n_sig_ours_0.05)",
    "3{,": "CHECKED (airway.json n_sig_r_0.05)",
    "206": "CHECKED (airway_cell.json n_sig_r_0.05)",
    "$\\ge$0.980": "min Jaccard over the alpha sweep of Fig 4(d); computed by "
                  "validation/make_paper_figures.py (committed J@0.05 floor 0.9805)",
    "$\\ge$99.6\\%": "min top-N concordance over Fig 4(b); computed by "
                     "validation/make_paper_figures.py",
    "141": "dispOutlier count, identical in R and cuDESeq2 -- annotated in "
           "Fig 3 left column; source r_disp_details.csv (regenerable via "
           "validation/export_r_dispersion_details.R)",
    # The Wald-SE minmu deviation and its fix. The before/after ratios are
    # regression-pinned rather than stored: the test fails if the floor is removed.
    "22\\%": "SE ratio ours/R on airway_cell empty-group genes BEFORE the minmu "
             "fix -- regime pinned by tests/test_wald_se_minmu.py",
    "0.03\\%": "same genes AFTER the fix (ratio 1.00032) -- "
               "tests/test_wald_se_minmu.py",
    "0.07": "max |delta p| on dispersion-ceiling genes before the fix (airway)",
    "0.971": "airway_cell Jaccard@0.05 BEFORE the minmu fix (now 1.000; see "
             "validation/README.md residuals section)",
    "878": "genes replaceOutliers rescues from the Cook's filter on gtex = "
           "NaN-padj difference between bench/cache (substep chain) and "
           "validation/data (DESeq wrapper) R references",
    "654": "of those 878, the ones R calls at alpha=0.05",
}
# Claims with no committed source. Must stay empty: an unsubstantiated number
# is removed from the paper, not annotated.
UNBACKED = {}
facts = set(re.findall(r"\\fact\{([^}]*)\}", tex))
facts = {f for f in facts if not f.startswith("Reproduce:") and f != "..."}
unregistered = sorted(f for f in facts if f not in REGISTRY)
for f in unregistered:
    fails.append(f"UNREGISTERED \\fact{{{f}}}: no provenance entry")
    checks += 1

print(f"{checks} checks run over {len(facts)} registered \\fact{{}} claims")
if UNBACKED:
    print(f"\n{len(UNBACKED)} CLAIM(S) WITH NO COMMITTED SOURCE (must not ship):")
    for k, v in UNBACKED.items():
        print(f"  ! {k}\n      {v}")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
else:
    print("ALL NUMERIC CLAIMS MATCH THEIR SOURCE DATA")
