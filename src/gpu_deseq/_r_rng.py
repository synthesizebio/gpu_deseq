"""Bit-exact replay of R's random-number stream.

R DESeq2's small-residual-dof prior-variance estimator
(`estimateDispersionsPriorVar`, the `(m-p) <= 3` branch) is a Monte-Carlo grid
search seeded with `set.seed(2)`. That seed makes the result reproducible
*within* R only: any other generator draws different numbers, shifts the KL
curve's simulation noise, and moves the reported argmin. Matching DESeq2 there
therefore means reproducing R's stream itself, not merely its distributions.

This module does that, porting the pieces R actually uses:

  * `RNG_Init` for MERSENNE_TWISTER -- R does *not* use MT's `init_genrand`; it
    fills all 625 state words from a plain LCG after 50 rounds of scrambling
    (`src/main/RNG.c`).
  * `MT_genrand` + `fixup` -- standard MT19937 tempering, but R converts with a
    single 32-bit word (`y / 2^32`) where NumPy uses 53 bits from two words.
  * `norm_rand` INVERSION -- two uniforms into `qnorm5`, which is Wichura's
    AS 241 (`src/nmath/qnorm.c`), not the Cephes `ndtri` SciPy exposes.
  * `exp_rand` -- Ahrens & Dieter (1972) (`src/nmath/sexp.c`).
  * `rgamma` -- Ahrens & Dieter GD for shape >= 1, GS for shape < 1
    (`src/nmath/rgamma.c`). Both are rejection samplers, so each draw consumes a
    variable number of uniforms and the stream cannot be vectorised; the loop is
    compiled with numba instead.

Verified against R 4.x: `runif`, `rnorm` and `rchisq` streams reproduce to the
last bit.
"""
from __future__ import annotations

import numpy as np

try:                                                  # pragma: no cover
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*args, **kwargs):
        """No-op fallback: the same code runs, interpreted and ~300x slower,
        producing identical numbers."""
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


_M32 = 0xFFFFFFFF
_N, _M = 624, 397
_MATRIX_A = 0x9908B0DF
_UPPER, _LOWER = 0x80000000, 0x7FFFFFFF
_I2_32M1 = 2.328306437080797e-10                      # 1 / (2^32 - 1)
_BIG = 134217728.0                                    # 2^27, R's norm_rand

# R sexp.c: q[k-1] = sum(log(2)^k / k!), k = 1..16
_EXP_Q = np.array([
    0.6931471805599453, 0.9333736875190459, 0.9888777961838675,
    0.9984959252914960040, 0.9998292811061389, 0.9999833164100727,
    0.99999855082196313338, 0.99999988631034085000, 0.99999999183430124630,
    0.99999999945867460330, 0.99999999996696138075, 0.99999999999811452368,
    0.99999999999989906800, 0.99999999999999493200, 0.99999999999999976000,
    0.99999999999999999000,
])


@njit(cache=True)
def _mt_init(seed):
    """R RNG.c `RNG_Init` for MERSENNE_TWISTER: 50 scrambling rounds, then 625
    LCG words. i_seed[0] is the index (fixed to 624); i_seed[1:] is the state."""
    s = seed & _M32
    for _ in range(50):
        s = (69069 * s + 1) & _M32
    mt = np.empty(_N, dtype=np.int64)
    s = (69069 * s + 1) & _M32                        # i_seed[0], then fixed up
    for j in range(_N):
        s = (69069 * s + 1) & _M32
        mt[j] = s
    return mt


@njit(cache=True)
def _mt_word(mt, st):
    """One tempered 32-bit MT output; `st[0]` carries the index."""
    idx = st[0]
    if idx >= _N:
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
            mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
        mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        idx = 0
    y = mt[idx]
    idx += 1
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y = y & _M32
    y ^= (y >> 18)
    st[0] = idx
    return y & _M32


@njit(cache=True)
def _unif(mt, st):
    """R `unif_rand()` for MT: word / 2^32, nudged strictly inside (0, 1)."""
    u = _mt_word(mt, st) * 2.3283064365386963e-10
    if u <= 0.0:
        return 0.5 * _I2_32M1
    if 1.0 - u <= 0.0:
        return 1.0 - 0.5 * _I2_32M1
    return u


@njit(cache=True)
def _qnorm(p):
    """R `qnorm5(p, 0, 1, TRUE, FALSE)` -- Wichura's AS 241."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                    + 67265.770927008700853) * r + 45921.953931549871457) * r
                  + 13731.693765509461125) * r + 1971.5909503065514427) * r
                + 133.14166789178437745) * r + 3.387132872796366608)
        den = (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                    + 39307.89580009271061) * r + 21213.794301586595867) * r
                  + 5394.1960214247511077) * r + 687.1870074920579083) * r
                + 42.313330701600911252) * r + 1.0)
        return q * num / den
    r = p if q < 0.0 else 1.0 - p
    r = np.sqrt(-np.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                    + 0.24178072517745061177) * r + 1.27045825245236838258) * r
                  + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                + 4.6303378461565452959) * r + 1.42343711074968357734)
        den = (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966) * r + 0.14810397642748007459) * r
                  + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5.0
        num = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                    + 0.0012426609473880784386) * r + 0.026532189526576123093) * r
                  + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                + 5.4637849111641143699) * r + 6.6579046435011037772)
        den = (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
                  + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                + 0.59983220655588793769) * r + 1.0)
    val = num / den
    return -val if q < 0.0 else val


@njit(cache=True)
def _norm(mt, st):
    """R `norm_rand()`, case INVERSION: two uniforms give 27+32 bits of p."""
    u1 = _unif(mt, st)
    u1 = float(int(_BIG * u1)) + _unif(mt, st)
    return _qnorm(u1 / _BIG)


@njit(cache=True)
def _exp(mt, st, q):
    """R `exp_rand()` -- Ahrens & Dieter (1972)."""
    a = 0.0
    u = _unif(mt, st)
    while u <= 0.0 or u >= 1.0:
        u = _unif(mt, st)
    while True:
        u += u
        if u > 1.0:
            break
        a += q[0]
    u -= 1.0
    if u <= q[0]:
        return a + u
    i = 0
    ustar = _unif(mt, st)
    umin = ustar
    while True:
        ustar = _unif(mt, st)
        umin = min(umin, ustar)
        i += 1
        if u <= q[i]:
            break
    return a + umin * q[0]


@njit(cache=True)
def _rgamma(mt, st, a, scale, q):
    """R `rgamma(a, scale)` -- GS for a < 1, Ahrens & Dieter GD for a >= 1."""
    exp_m1 = 0.36787944117144232
    if a < 1.0:                                       # GS
        if a == 0.0:
            return 0.0
        e = 1.0 + exp_m1 * a
        while True:
            p = e * _unif(mt, st)
            if p >= 1.0:
                x = -np.log((e - p) / a)
                if _exp(mt, st, q) >= (1.0 - a) * np.log(x):
                    break
            else:
                x = np.exp(np.log(p) / a)
                if _exp(mt, st, q) >= x:
                    break
        return scale * x

    sqrt32 = 5.656854                                 # GD
    s2 = a - 0.5
    s = np.sqrt(s2)
    d = sqrt32 - s * 12.0

    t = _norm(mt, st)
    x = s + 0.5 * t
    ret = x * x
    if t >= 0.0:
        return scale * ret

    u = _unif(mt, st)
    if d * u <= t * t * t:
        return scale * ret

    r = 1.0 / a
    q0 = ((((((2.424e-4 * r + 2.4511e-4) * r - 7.388e-5) * r
             + 0.00144121) * r + 0.00801191) * r + 0.02083148) * r
          + 0.04166669) * r
    if a <= 3.686:
        b = 0.463 + s + 0.178 * s2
        si = 1.235
        c = 0.195 / s - 0.079 + 0.16 * s
    elif a <= 13.022:
        b = 1.654 + 0.0076 * s2
        si = 1.68 / s + 0.275
        c = 0.062 / s + 0.024
    else:
        b = 1.77
        si = 0.75
        c = 0.1515 / s

    if x > 0.0:
        v = t / (s + s)
        if abs(v) <= 0.25:
            qq = q0 + 0.5 * t * t * ((((((0.1233795 * v - 0.1367177) * v
                                         + 0.1423657) * v - 0.1662921) * v
                                       + 0.2000062) * v - 0.250003) * v
                                     + 0.3333333) * v
        else:
            qq = q0 - s * t + 0.25 * t * t + (s2 + s2) * np.log(1.0 + v)
        if np.log(1.0 - u) <= qq:
            return scale * ret

    while True:
        e = _exp(mt, st, q)
        u = _unif(mt, st)
        u = u + u - 1.0
        t = b - si * e if u < 0.0 else b + si * e
        if t >= -0.71874483771719:
            v = t / (s + s)
            if abs(v) <= 0.25:
                qq = q0 + 0.5 * t * t * ((((((0.1233795 * v - 0.1367177) * v
                                             + 0.1423657) * v - 0.1662921) * v
                                           + 0.2000062) * v - 0.250003) * v
                                         + 0.3333333) * v
            else:
                qq = q0 - s * t + 0.25 * t * t + (s2 + s2) * np.log(1.0 + v)
            if qq > 0.0:
                w = np.expm1(qq)
                if c * abs(u) <= w * np.exp(e - 0.5 * t * t):
                    break
    x = s + 0.5 * t
    return scale * x * x


@njit(cache=True)
def _kl_draws(seed, dof, var_grid, n_samp, q):
    """R's `sapply(obsVarGrid, ...)` body, in stream order: per grid point,
    `rchisq(n, dof)` then `rnorm(n, 0, sqrt(x))`, returning
    log(chisq) + normal - log(dof)."""
    mt = _mt_init(seed)
    st = np.empty(1, dtype=np.int64)
    st[0] = _N                                        # FixupSeeds: I624 = 624
    out = np.empty((var_grid.size, n_samp), dtype=np.float64)
    shape = dof / 2.0
    logdof = np.log(float(dof))
    chi = np.empty(n_samp, dtype=np.float64)
    for i in range(var_grid.size):
        for j in range(n_samp):
            chi[j] = _rgamma(mt, st, shape, 2.0, q)   # rchisq(dof) = rgamma(dof/2, 2)
        sd = np.sqrt(var_grid[i])
        if sd == 0.0:
            # R's rnorm() returns mu directly when sigma == 0 and does NOT call
            # norm_rand(). Drawing here would consume 10^4 variates R never
            # takes and desynchronise every later grid point.
            for j in range(n_samp):
                out[i, j] = np.log(chi[j]) - logdof
        else:
            for j in range(n_samp):
                out[i, j] = np.log(chi[j]) + sd * _norm(mt, st) - logdof
    return out


def kl_grid_draws(dof: int, var_grid: np.ndarray, n_samp: int = 10000,
                  seed: int = 2) -> np.ndarray:
    """R's simulated log-dispersion residuals for the whole KL grid."""
    return _kl_draws(int(seed), int(dof),
                     np.asarray(var_grid, dtype=np.float64), int(n_samp), _EXP_Q)


def unif_rand(n: int, seed: int = 2) -> np.ndarray:
    """R `set.seed(seed); runif(n)` -- exposed for testing."""
    mt = _mt_init(int(seed))
    st = np.empty(1, dtype=np.int64)
    st[0] = _N
    return np.array([_unif(mt, st) for _ in range(n)])


def norm_rand(n: int, seed: int = 2) -> np.ndarray:
    """R `set.seed(seed); rnorm(n)` -- exposed for testing."""
    mt = _mt_init(int(seed))
    st = np.empty(1, dtype=np.int64)
    st[0] = _N
    return np.array([_norm(mt, st) for _ in range(n)])


def rchisq(n: int, df: int, seed: int = 2) -> np.ndarray:
    """R `set.seed(seed); rchisq(n, df)` -- exposed for testing."""
    mt = _mt_init(int(seed))
    st = np.empty(1, dtype=np.int64)
    st[0] = _N
    return np.array([_rgamma(mt, st, df / 2.0, 2.0, _EXP_Q) for _ in range(n)])
