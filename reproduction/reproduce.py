#!/usr/bin/env python
"""Reproduction script for
'Spurious Information: Random Matrix Theory, Entropy and the Value of
 Information in High-Dimensional Markets'.

Every number, table and figure in the paper is produced here from fixed seeds
and from the committed industry return file.  Run from the repository root:

    python reproduction/reproduce.py                 # full run (a few minutes)
    python reproduction/reproduce.py --quick         # reduced Monte Carlo sizes

Outputs
    reproduction/results/*.csv, reproduction/results/numbers.json
    paper/generated/numbers.tex   (LaTeX macros used in the text)
    paper/generated/*.tex         (tables)
    paper/figures/*.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ff49_vw_daily_1990_2026.csv"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})
C0, C1, C2, C3 = "#1f4e79", "#c0504d", "#4f8a3c", "#7f6000"

# ---------------------------------------------------------------------------
# Analytic results (Sections 3-6 of the paper)
# ---------------------------------------------------------------------------

def mp_edges(q, s2=1.0):
    return s2 * (1 - math.sqrt(q)) ** 2, s2 * (1 + math.sqrt(q)) ** 2


def mp_density(x, q, s2=1.0):
    lo, hi = mp_edges(q, s2)
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2 * math.pi * q * s2 * x[m])
    return out


def phi(c):
    """Limit of N^{-1}[E log det S - log det Sigma] at N/n -> c in (0,1)."""
    return -1.0 - (1.0 - c) / c * math.log1p(-c)


def logdet_bias(N, n):
    """Exact E log det(W/n) - log det Sigma for W ~ Wishart_N(n, Sigma)."""
    i = np.arange(1, N + 1)
    return float(np.sum(digamma((n - i + 1) / 2.0) - math.log(n / 2.0)))


def psi_sum(N, n):
    i = np.arange(1, N + 1)
    return float(np.sum(digamma((n - i + 1) / 2.0)))


def mi_bias(p, r, n):
    """Exact E[I_hat] - I for the Gaussian plug-in mutual information."""
    return 0.5 * (psi_sum(p, n) + psi_sum(r, n) - psi_sum(p + r, n))


def mi_null_var(p, r, n):
    """Exact Var(I_hat) under independence (Wilks Lambda as product of Betas)."""
    i = np.arange(1, p + 1)
    return 0.25 * float(np.sum(polygamma(1, (n - r - i + 1) / 2.0)
                               - polygamma(1, (n - i + 1) / 2.0)))


def kappa(c1, c2):
    """Spurious information per degree of freedom, proportional limit."""
    def h(c):
        return (1 - c) * math.log1p(-c) if c < 1 else 0.0
    return 0.5 * (-h(c1) - h(c2) + h(c1 + c2))


def nu(c1, c2):
    """Limit null variance of I_hat in the proportional regime."""
    return 0.5 * math.log((1 - c1) * (1 - c2) / (1 - c1 - c2))


def kl_forward_exact(N, n):
    """E KL(N(0,Sigma) || N(0,S)) = E[Stein loss]/2, S = W/n."""
    return 0.5 * (N * n / (n - N - 1) - N + logdet_bias(N, n))


def kl_reverse_exact(N, n):
    """E KL(N(0,S) || N(0,Sigma))."""
    return -0.5 * logdet_bias(N, n)


def kl_forward_limit(q):
    return 0.5 * (q / (1 - q) + phi(q))


def kl_reverse_limit(q):
    return -0.5 * phi(q)


def gmv_in_exact(N, n):
    return (n - N + 1) / n


def gmv_out_exact(N, n):
    return (n - 1) / (n - N)


def bbp_outlier(ell, q):
    return ell * (1 + q / (ell - 1)) if ell > 1 + math.sqrt(q) else (1 + math.sqrt(q)) ** 2


def bbp_overlap(ell, q):
    if ell <= 1 + math.sqrt(q):
        return 0.0
    return (1 - q / (ell - 1) ** 2) / (1 + q / (ell - 1))


def bbp_invert(lam, q):
    """Population spike ell implied by a sample outlier lam > (1+sqrt q)^2."""
    if lam <= (1 + math.sqrt(q)) ** 2:
        return float("nan")
    b = lam + 1 - q
    return 0.5 * (b + math.sqrt(b * b - 4 * lam))


def mou_kl_rate(N, a, b):
    """KL rate of symmetric matrix OU with drift -aH versus -bH (stationary)."""
    return N * (N + 1) * (b - a) ** 2 / (8 * a)


def wachter_edge(c1, c2):
    return (math.sqrt(c1 * (1 - c2)) + math.sqrt(c2 * (1 - c1))) ** 2


def cca_threshold(c1, c2):
    return (c1 * c2 / ((1 - c1) * (1 - c2))) ** 0.25


def cca_outlier(t, c1, c2):
    """Limit of the leading squared sample canonical correlation (rank one)."""
    if t <= cca_threshold(c1, c2):
        return wachter_edge(c1, c2)
    t2 = t * t
    return t2 * (1 - c1 + c1 / t2) * (1 - c2 + c2 / t2)


def info_frontier(c1, c2):
    """Smallest rank-one mutual information detectable by the leading CC."""
    return -0.5 * math.log1p(-cca_threshold(c1, c2) ** 2)


def narayana_moment(k, q):
    if k == 0:
        return 1.0
    return sum(math.comb(k, j) * math.comb(k, j + 1) / k * q ** j for j in range(k))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def gaussian_mi(S, p):
    """Plug-in Gaussian mutual information between first p and remaining coords."""
    s1 = np.linalg.slogdet(S[:p, :p])[1]
    s2 = np.linalg.slogdet(S[p:, p:])[1]
    s = np.linalg.slogdet(S)[1]
    return 0.5 * (s1 + s2 - s)


def ledoit_wolf(X):
    """Ledoit-Wolf (2004) linear shrinkage towards a scaled identity. X demeaned."""
    T, N = X.shape
    S = X.T @ X / T
    m = np.trace(S) / N
    d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    xs = X ** 2
    b2bar = (np.sum(xs.T @ xs) / T - np.sum(S ** 2)) / (T * N)
    b2 = min(b2bar, d2)
    return (b2 / d2) * m * np.eye(N) + (1 - b2 / d2) * S


def corr_from_cov(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d), d


def clipped(S, T):
    C, d = corr_from_cov(S)
    N = C.shape[0]
    q = N / T
    lam, V = np.linalg.eigh(C)
    s2 = max(1 - lam[-1] / N, 1e-3)
    edge = s2 * (1 + math.sqrt(q)) ** 2
    noise = lam <= edge
    lam2 = lam.copy()
    if noise.any():
        lam2[noise] = lam[noise].mean()
    Cc = (V * lam2) @ V.T
    return Cc * np.outer(d, d)


def rie(S, T):
    """Rotationally invariant estimator of Bun, Bouchaud and Potters (2016)."""
    C, d = corr_from_cov(S)
    N = C.shape[0]
    q = N / T
    lam, V = np.linalg.eigh(C)
    eta = N ** -0.5
    z = lam - 1j * eta
    g = np.mean(1.0 / (z[:, None] - lam[None, :]), axis=1)
    xi = lam / np.abs(1 - q + q * z * g) ** 2
    xi *= N / xi.sum()
    Cc = (V * xi) @ V.T
    return Cc * np.outer(d, d)


def gmv_weights(S):
    x = np.linalg.solve(S, np.ones(S.shape[0]))
    return x / x.sum()


def wishart_cov(rng, Sig_half, n):
    """Return W/n where W ~ Wishart(n, Sigma), Sigma = Sig_half Sig_half^T."""
    N = Sig_half.shape[0]
    Z = rng.standard_normal((n, N)) @ Sig_half.T
    return Z.T @ Z / n, Z


def random_cov(rng, N, cond=10.0):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    ev = np.exp(np.linspace(0, math.log(cond), N))
    return (Q * ev) @ Q.T


# ---------------------------------------------------------------------------
# Monte Carlo verification of the analytic results
# ---------------------------------------------------------------------------

def mc_checks(reps, rng):
    rows = []

    def add(name, analytic, sims, label, note=""):
        sims = np.asarray(sims, float)
        se = sims.std(ddof=1) / math.sqrt(len(sims))
        rows.append(dict(check=name, label=label, analytic=analytic,
                         simulated=float(sims.mean()), mc_se=float(se),
                         z=float((sims.mean() - analytic) / se) if se > 0 else 0.0,
                         reps=len(sims), note=note))

    # MP second and third moments at finite N (exact second moment)
    N, n = 100, 400
    q = N / n
    m2, m3 = [], []
    for _ in range(reps):
        S, _ = wishart_cov(rng, np.eye(N), n)
        m2.append(np.trace(S @ S) / N)
        m3.append(np.trace(S @ S @ S) / N)
    add("mp_m2", 1 + q + 1 / n, m2, r"$N^{-1}\E\tr S^2$ (exact)")
    rows[-1]["limit"] = narayana_moment(2, q)
    add("mp_m3", 1 + 3 * q + q * q + 3 * (1 + q) / n + 4 / n ** 2, m3,
        r"$N^{-1}\E\tr S^3$ (exact)")
    rows[-1]["limit"] = narayana_moment(3, q)

    # log det: exact and limit
    N, n = 100, 400
    Sig = random_cov(rng, N)
    Sh = np.linalg.cholesky(Sig)
    ld0 = np.linalg.slogdet(Sig)[1]
    ld = []
    for _ in range(reps):
        S, _ = wishart_cov(rng, Sh, n)
        ld.append(np.linalg.slogdet(S)[1] - ld0)
    add("logdet_exact", logdet_bias(N, n), ld, r"$\E\log\det S-\log\det\Sigma$ (exact)")
    rows[-1]["limit"] = N * phi(N / n)

    # MI bias with non-trivial true dependence
    p, r, n = 10, 20, 100
    N = p + r
    Sig = random_cov(rng, N, cond=20.0)
    I_true = gaussian_mi(Sig, p)
    Sh = np.linalg.cholesky(Sig)
    mis = []
    for _ in range(reps):
        S, _ = wishart_cov(rng, Sh, n)
        mis.append(gaussian_mi(S, p) - I_true)
    add("mi_bias", mi_bias(p, r, n), mis, r"$\E\hat I-I$, $(p,r,n)=(10,20,100)$",
        f"true I={I_true:.3f}")
    rows[-1]["limit"] = n * kappa(p / n, r / n)

    # Null variance of I_hat
    p, r, n = 10, 20, 100
    v = []
    for _ in range(reps):
        S, _ = wishart_cov(rng, np.eye(p + r), n)
        v.append(gaussian_mi(S, p))
    v = np.asarray(v)
    add("mi_null_mean", mi_bias(p, r, n), v, r"$\E_0\hat I$ (independence)")
    vv = (v - v.mean()) ** 2 * len(v) / (len(v) - 1)
    add("mi_null_var", mi_null_var(p, r, n), vv, r"$\Var_0\hat I$ (Wilks)")
    rows[-1]["limit"] = nu(p / n, r / n)

    # Stein / KL costs
    N, n = 20, 100
    Sig = random_cov(rng, N)
    Sh = np.linalg.cholesky(Sig)
    Si = np.linalg.inv(Sig)
    ld0 = np.linalg.slogdet(Sig)[1]
    kf, kr = [], []
    for _ in range(reps):
        S, _ = wishart_cov(rng, Sh, n)
        ldS = np.linalg.slogdet(S)[1]
        kf.append(0.5 * (np.trace(np.linalg.solve(S, Sig)) - N + ldS - ld0))
        kr.append(0.5 * (np.trace(Si @ S) - N - ldS + ld0))
    add("kl_forward", kl_forward_exact(N, n), kf, r"$\E\KL(\N_\Sigma\|\N_S)$")
    rows[-1]["limit"] = N * kl_forward_limit(N / n)
    add("kl_reverse", kl_reverse_exact(N, n), kr, r"$\E\KL(\N_S\|\N_\Sigma)$")
    rows[-1]["limit"] = N * kl_reverse_limit(N / n)

    # GMV in/out of sample
    N, n = 50, 200
    Sig = random_cov(rng, N)
    Sh = np.linalg.cholesky(Sig)
    one = np.ones(N)
    Rt = 1 / (one @ np.linalg.solve(Sig, one))
    rin, rout = [], []
    for _ in range(reps):
        S, _ = wishart_cov(rng, Sh, n)
        w = gmv_weights(S)
        rin.append((w @ S @ w) / Rt)
        rout.append((w @ Sig @ w) / Rt)
    add("gmv_in", gmv_in_exact(N, n), rin, r"$\E R_{\rm in}/R$")
    rows[-1]["limit"] = 1 - N / n
    add("gmv_out", gmv_out_exact(N, n), rout, r"$\E R_{\rm out}/R$")
    rows[-1]["limit"] = 1 / (1 - N / n)

    # BBP outlier and overlap
    N, n = 400, 1600
    q = N / n
    for ell in (2.0, 4.0):
        lam1, ov = [], []
        for _ in range(max(reps // 10, 50)):
            Sh = np.eye(N)
            Sh[0, 0] = math.sqrt(ell)
            S, _ = wishart_cov(rng, Sh, n)
            w, V = np.linalg.eigh(S)
            lam1.append(w[-1])
            ov.append(V[0, -1] ** 2)
        add(f"bbp_outlier_{ell:g}", bbp_outlier(ell, q), lam1,
            rf"BBP outlier, $\ell={ell:g}$", "limit only")
        add(f"bbp_overlap_{ell:g}", bbp_overlap(ell, q), ov,
            rf"BBP overlap$^2$, $\ell={ell:g}$", "limit only")

    # Matrix OU information rate (Euler log-likelihood ratio)
    N, a, b, dt, steps = 10, 0.5, 1.0, 0.01, 2000
    Hs = []
    paths = max(reps // 20, 40)
    llr = np.zeros(paths)
    iu = np.triu_indices(N)
    var = np.where(iu[0] == iu[1], 2.0 / N, 1.0 / N) * dt
    for k in range(paths):
        G = rng.standard_normal((N, N))
        H = (G + G.T) / math.sqrt(2) * math.sqrt(1 / (2 * a * N))  # stationary draw
        tot = 0.0
        for _ in range(steps):
            G = rng.standard_normal((N, N))
            dB = (G + G.T) / math.sqrt(2) * math.sqrt(dt)
            Hn = H - a * H * dt + dB / math.sqrt(N)
            inc = (Hn - H)[iu]
            mq = (-a * H * dt)[iu]
            mp = (-b * H * dt)[iu]
            tot += np.sum(((inc - mp) ** 2 - (inc - mq) ** 2) / (2 * var))
            H = Hn
        llr[k] = tot / (steps * dt)
    add("mou_kl_rate", mou_kl_rate(N, a, b), llr, r"matrix OU KL rate",
        "Euler, dt=0.01")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Spurious-information experiments (simulation)
# ---------------------------------------------------------------------------

def spurious_curves(reps, rng):
    n = 250
    rows = []
    for p in (2, 5, 10, 20, 30, 40, 60, 80, 100):
        sims = []
        for _ in range(reps // 4):
            S, _ = wishart_cov(rng, np.eye(2 * p), n)
            sims.append(gaussian_mi(S, p))
        sims = np.asarray(sims)
        rows.append(dict(p=p, r=p, n=n, exact=mi_bias(p, p, n),
                         limit=n * kappa(p / n, p / n), classical=p * p / (2 * n),
                         sim_mean=sims.mean(), sim_se=sims.std(ddof=1) / math.sqrt(len(sims)),
                         exact_sd=math.sqrt(mi_null_var(p, p, n)), sim_sd=sims.std(ddof=1)))
    return pd.DataFrame(rows)


def noise_dims(reps, rng):
    """Signal block of k informative + m uninformative signals about v (dim N)."""
    n, Nv, k = 250, 5, 5
    rho = 0.4  # canonical correlations of informative signals
    rows = []
    for m in (0, 5, 10, 20, 40, 80):
        r = k + m
        p = Nv
        Sig = np.eye(p + r)
        for j in range(k):
            Sig[j, p + j] = Sig[p + j, j] = rho
        I_true = gaussian_mi(Sig, p)
        Sh = np.linalg.cholesky(Sig)
        plug, corr, lw, ri = [], [], [], []
        for _ in range(reps // 4):
            S, Z = wishart_cov(rng, Sh, n)
            Ihat = gaussian_mi(S, p)
            plug.append(Ihat)
            corr.append(Ihat - mi_bias(p, r, n))
            lw.append(gaussian_mi(ledoit_wolf(Z), p))
            ri.append(gaussian_mi(rie(S, n), p))
        rows.append(dict(m=m, r=r, I_true=I_true, plug_in=np.mean(plug),
                         plug_se=np.std(plug, ddof=1) / math.sqrt(len(plug)),
                         corrected=np.mean(corr),
                         corrected_se=np.std(corr, ddof=1) / math.sqrt(len(corr)),
                         ledoit_wolf=np.mean(lw), rie=np.mean(ri),
                         predicted_plug_in=I_true + mi_bias(p, r, n)))
    return pd.DataFrame(rows)


def cca_phase(reps, rng):
    """Leading sample canonical correlation in a rank-one spiked CCA model."""
    n, p, r = 1000, 100, 200
    c1, c2 = p / n, r / n
    rows = []
    for t in np.linspace(0.0, 0.8, 17):
        Sig = np.eye(p + r)
        Sig[0, p] = Sig[p, 0] = t
        Sh = np.linalg.cholesky(Sig)
        tops, ovs = [], []
        for _ in range(max(reps // 25, 30)):
            S, _ = wishart_cov(rng, Sh, n)
            A = np.linalg.cholesky(S[:p, :p])
            B = np.linalg.cholesky(S[p:, p:])
            K = np.linalg.solve(A, S[:p, p:])
            K = np.linalg.solve(B, K.T).T
            sv = np.linalg.svd(K, compute_uv=False)
            tops.append(sv[0] ** 2)
        rows.append(dict(t=t, t2=t * t, limit=cca_outlier(t, c1, c2), top_cc2=np.mean(tops),
                         top_cc2_sd=np.std(tops, ddof=1)))
    return pd.DataFrame(rows), wachter_edge(c1, c2), cca_threshold(c1, c2), (n, p, r)


# ---------------------------------------------------------------------------
# Empirical analysis
# ---------------------------------------------------------------------------

CARBON = ["Oil", "Coal", "Util", "Mines", "Steel", "Chems", "Trans", "Autos"]
FIN = ["Banks", "Insur", "RlEst", "Fin"]


def load_returns():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    return df / 100.0


def rolling_spectrum(R, T=250, step=21):
    X = R.values
    N = X.shape[1]
    q = N / T
    rows = []
    u = np.ones(N) / math.sqrt(N)
    for end in range(T, len(X) + 1, step):
        W = X[end - T:end]
        W = W - W.mean(0)
        C = np.corrcoef(W, rowvar=False)
        lam, V = np.linalg.eigh(C)
        lam, V = lam[::-1], V[:, ::-1]
        s2 = 1 - lam[0] / N
        edge = s2 * (1 + math.sqrt(q)) ** 2
        n_out = int(np.sum(lam > edge))
        rows.append(dict(date=R.index[end - 1], lambda1=lam[0], absorption=lam[0] / N,
                         n_outliers=n_out, edge=edge, sigma2=s2,
                         market_overlap=float((V[:, 0] @ u) ** 2),
                         lambda2=lam[1], lambda3=lam[2]))
    return pd.DataFrame(rows)


def snapshot(R, start, end):
    W = R.loc[start:end].values
    T, N = W.shape
    W = W - W.mean(0)
    C = np.corrcoef(W, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    q = N / T
    s2 = 1 - lam[0] / N
    rows = []
    for k in range(8):
        lk = lam[k] / s2 if k > 0 else lam[k]
        ell = bbp_invert(lk, q)
        rows.append(dict(k=k + 1, eigenvalue=lam[k], scaled=lk, implied_spike=ell,
                         predicted_overlap=bbp_overlap(ell, q) if ell == ell else 0.0,
                         top_loadings=", ".join(R.columns[np.argsort(-np.abs(V[:, k]))[:4]])))
    return lam, q, s2, T, pd.DataFrame(rows)


def backtest(R, windows=(60, 120, 250, 500), hold=21, start_index=500):
    X = R.values
    N = X.shape[1]
    est = {
        "Sample": lambda W, T: W.T @ W / T,
        "Clipped": lambda W, T: clipped(W.T @ W / T, T),
        "Ledoit-Wolf": lambda W, T: ledoit_wolf(W),
        "RIE": lambda W, T: rie(W.T @ W / T, T),
    }
    rows = []
    rets = {}
    for T in windows:
        for name, f in est.items():
            realized, pred = [], []
            for t0 in range(start_index, len(X) - hold + 1, hold):
                W = X[t0 - T:t0]
                W = W - W.mean(0)
                S = f(W, T)
                w = gmv_weights(S)
                fut = X[t0:t0 + hold] @ w
                realized.append(fut)
                pred.append(w @ S @ w)
            rr = np.concatenate(realized)
            vol = rr.std(ddof=1) * math.sqrt(252)
            pv = math.sqrt(np.mean(pred) * 252)
            rows.append(dict(window=T, q=N / T, estimator=name, realized_vol=vol,
                             predicted_vol=pv, var_ratio=(vol / pv) ** 2,
                             theory_ratio=(1 / (1 - N / T) ** 2 if name == "Sample" else float("nan"))))
            rets[(T, name)] = rr
    # equal weight
    rr = np.concatenate([X[t0:t0 + hold].mean(1)
                         for t0 in range(start_index, len(X) - hold + 1, hold)])
    rows.append(dict(window=0, q=float("nan"), estimator="Equal weight",
                     realized_vol=rr.std(ddof=1) * math.sqrt(252), predicted_vol=float("nan"),
                     var_ratio=float("nan"), theory_ratio=float("nan")))
    return pd.DataFrame(rows)


def rolling_mi(R, T=250, step=21, rng=None, noise_m=(0, 8, 24)):
    A = R[CARBON].values
    B = R[FIN].values
    M = R.values.mean(1)
    p, r = A.shape[1], B.shape[1]
    n = T - 1
    rows = []
    for end in range(T, len(A) + 1, step):
        Xa, Xb = A[end - T:end], B[end - T:end]
        Z = np.hstack([Xa, Xb])
        Z = Z - Z.mean(0)
        S = Z.T @ Z / T
        Ihat = gaussian_mi(S, p)
        # null: circularly shift the carbon block by a random offset within window
        sh = rng.integers(21, T - 21)
        Zn = np.hstack([np.roll(Xa, sh, axis=0), Xb])
        Zn = Zn - Zn.mean(0)
        In = gaussian_mi(Zn.T @ Zn / T, p)
        # partial information given the equal-weighted market return
        mk = M[end - T:end] - M[end - T:end].mean()
        beta = (mk @ Z) / (mk @ mk)
        Zp = Z - np.outer(mk, beta)
        Ip = gaussian_mi(Zp.T @ Zp / T, p)
        row = dict(date=R.index[end - 1], plug_in=Ihat, corrected=Ihat - mi_bias(p, r, n),
                   partial_plug_in=Ip, partial_corrected=Ip - mi_bias(p, r, n - 1),
                   partial_z=(Ip - mi_bias(p, r, n - 1)) / math.sqrt(mi_null_var(p, r, n - 1)),
                   z=(Ihat - mi_bias(p, r, n)) / math.sqrt(mi_null_var(p, r, n)),
                   null_plug_in=In, null_corrected=In - mi_bias(p, r, n))
        for m in noise_m:
            if m == 0:
                continue
            E = rng.standard_normal((T, m))
            Zm = np.hstack([Xa, Xb, E])
            Zm = Zm - Zm.mean(0)
            Sm = Zm.T @ Zm / T
            Im = gaussian_mi(Sm, p)
            row[f"plug_in_m{m}"] = Im
            row[f"corrected_m{m}"] = Im - mi_bias(p, r + m, n)
        rows.append(row)
    return pd.DataFrame(rows), (p, r, n)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_mp_snapshot(lam, q, s2, T, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bulk = lam[lam < 4]
    ax.hist(bulk, bins=22, density=True, color=C0, alpha=0.5, edgecolor="white", linewidth=0.5, label="Sample eigenvalues")
    xs = np.linspace(0.001, 3.5, 800)
    ax.plot(xs, mp_density(xs, q, s2) * (1 - 1 / len(lam)), color=C1, lw=1.6,
            label=fr"MP, $q={q:.3f}$, $\sigma^2={s2:.2f}$")
    ax.plot(xs, mp_density(xs, q, 1.0), color=C3, lw=1.0, ls="--", label=r"MP, $\sigma^2=1$")
    ax.set_xlabel(r"eigenvalue of correlation matrix")
    ax.set_ylabel("density")
    big = ", ".join(f"{x:.1f}" for x in lam[lam >= 4])
    ax.set_title(f"49 industries, window of T={T} days; eigenvalues off scale: {big}", fontsize=8)
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_rolling_spectrum(sp, path):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    axes[0].plot(sp.date, sp.absorption, color=C0, lw=0.9)
    axes[0].set_ylabel(r"$\lambda_1/N$")
    ax2 = axes[0].twinx()
    ax2.plot(sp.date, sp.market_overlap, color=C1, lw=0.6, alpha=0.7)
    ax2.set_ylabel(r"$\langle u_1,\mathbf{1}\rangle^2/N$", color=C1)
    ax2.grid(False)
    axes[1].step(sp.date, sp.n_outliers, color=C2, lw=0.9, where="post")
    axes[1].set_ylabel("eigenvalues above edge")
    for ax in axes:
        for d in ("2001-09-11", "2008-09-15", "2020-03-16"):
            ax.axvline(pd.Timestamp(d), color="0.5", lw=0.6, ls=":")
    fig.savefig(path)
    plt.close(fig)


def fig_spurious(curves, nd, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    ax.errorbar(curves.p, curves.sim_mean, yerr=2 * curves.sim_sd, fmt="o", ms=3, color=C0,
                label=r"simulated $\hat I$ ($\pm 2$ sd)")
    ax.plot(curves.p, curves.exact, color=C1, lw=1.4, label="exact mean")
    ax.plot(curves.p, curves.limit, color=C3, lw=1.0, ls="--", label=r"$n\kappa(c_1,c_2)$")
    ax.plot(curves.p, curves.classical, color="0.5", lw=1.0, ls=":", label=r"$pr/(2n)$")
    ax.set_xlabel(r"$p=r$ (independent blocks, $n=250$)")
    ax.set_ylabel("nats")
    ax.legend(fontsize=6.5)
    ax = axes[1]
    ax.plot(nd.m, nd.plug_in, "o-", ms=3, color=C0, label="plug-in")
    ax.plot(nd.m, nd.predicted_plug_in, color=C1, lw=1.0, ls="--", label="predicted plug-in")
    ax.plot(nd.m, nd.corrected, "s-", ms=3, color=C2, label="digamma corrected")
    ax.plot(nd.m, nd.ledoit_wolf, "^-", ms=3, color=C3, label="Ledoit-Wolf plug-in")
    ax.plot(nd.m, nd.rie, "v-", ms=3, color="0.4", label="RIE plug-in")
    ax.axhline(nd.I_true.iloc[0], color="k", lw=0.7, label="true $I$")
    ax.set_xlabel("uninformative signals added, $m$")
    ax.set_ylabel("nats")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cca(cc, edge, thr, dims, path):
    n, p, r = dims
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.errorbar(cc.t, cc.top_cc2, yerr=2 * cc.top_cc2_sd, fmt="o", ms=3, color=C0,
                label="leading sample canonical corr.$^2$")
    ax.plot(cc.t, cc.t2, color="0.5", lw=0.9, ls=":", label="population $t^2$")
    ax.axhline(edge, color=C1, lw=1.0, ls="--", label="Wachter edge")
    tt = np.linspace(0, 0.8, 200)
    c1, c2 = p / n, r / n
    ax.plot(tt, [cca_outlier(x, c1, c2) for x in tt], color=C3, lw=1.2, label="outlier limit")
    ax.axvline(thr, color=C2, lw=1.0, ls="--", label="detection threshold $t_c$")
    ax.set_xlabel("population canonical correlation $t$")
    ax.set_title(f"$n={n}$, $p={p}$, $r={r}$", fontsize=8)
    ax.legend(fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)


def fig_rolling_mi(mi, path):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(mi.date, mi.plug_in, color=C0, lw=0.8, label="plug-in $\\hat I$")
    ax.plot(mi.date, mi.corrected, color=C2, lw=0.8, label="digamma corrected")
    ax.plot(mi.date, mi.partial_corrected, color=C3, lw=0.8, label="corrected, given market")
    ax.plot(mi.date, mi.null_plug_in, color="0.6", lw=0.6, label="plug-in, shifted null")
    if "plug_in_m24" in mi:
        ax.plot(mi.date, mi.plug_in_m24, color=C1, lw=0.6, ls="--",
                label="plug-in, 24 noise signals added")
    ax.set_ylabel("nats")
    ax.legend(fontsize=6.5, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_dyson(rng, path):
    N, a, dt, steps = 40, 0.5, 0.002, 1500
    H = np.zeros((N, N))
    ev = np.zeros((steps, N))
    for s in range(steps):
        G = rng.standard_normal((N, N))
        dB = (G + G.T) / math.sqrt(2) * math.sqrt(dt)
        H = H - a * H * dt + dB / math.sqrt(N)
        ev[s] = np.linalg.eigvalsh(H)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    t = np.arange(1, steps + 1) * dt
    axes[0].plot(t, ev, color=C0, lw=0.4)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("eigenvalues of $H_t$")
    # stationary spectrum of a larger matrix
    M = 400
    G = rng.standard_normal((M, M))
    Hs = (G + G.T) / math.sqrt(2) / math.sqrt(2 * a * M)
    e = np.linalg.eigvalsh(Hs)
    R2 = 2 / (2 * a)
    xs = np.linspace(-math.sqrt(R2) * 2, math.sqrt(R2) * 2, 400)
    rad = 2 * math.sqrt(1 / (2 * a))
    xs = np.linspace(-rad, rad, 400)
    axes[1].hist(e, bins=40, density=True, color=C0, alpha=0.55)
    axes[1].plot(xs, 2 / (math.pi * rad ** 2) * np.sqrt(rad ** 2 - xs ** 2), color=C1)
    axes[1].set_xlabel("stationary eigenvalue, $N=400$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def wachter_density(x, c1, c2):
    lo = (math.sqrt(c1 * (1 - c2)) - math.sqrt(c2 * (1 - c1))) ** 2
    hi = wachter_edge(c1, c2)
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((x[m] - lo) * (hi - x[m])) / (2 * math.pi * c1 * x[m] * (1 - x[m]))
    return out


def fig_theory(rng, path_mp, path_wachter, path_maps):
    # Marchenko-Pastur family
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    xs = np.linspace(1e-3, 3.2, 1200)
    for q, col in zip((0.05, 0.2, 0.5, 0.8), (C0, C2, C3, C1)):
        ax.plot(xs, mp_density(xs, q), color=col, lw=1.3, label=f"$q={q}$")
    ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho_q(x)$")
    ax.legend(fontsize=7)
    fig.savefig(path_mp)
    plt.close(fig)
    # Wachter law versus simulated null canonical correlations
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (n, p, r) in zip(axes, ((1000, 100, 200), (600, 150, 150))):
        cc = []
        for _ in range(30):
            Qx, _ = np.linalg.qr(rng.standard_normal((n, p)))
            Qy, _ = np.linalg.qr(rng.standard_normal((n, r)))
            cc.append(np.linalg.svd(Qx.T @ Qy, compute_uv=False) ** 2)
        cc = np.concatenate(cc)
        c1, c2 = p / n, r / n
        ax.hist(cc, bins=50, density=True, color=C0, alpha=0.5, label="simulated")
        xx = np.linspace(1e-4, wachter_edge(c1, c2) * 1.02, 600)
        ax.plot(xx, wachter_density(xx, c1, c2), color=C1, lw=1.4, label="Wachter law")
        ax.set_title(f"$c_1={c1:.2f}$, $c_2={c2:.2f}$", fontsize=8)
        ax.set_xlabel("squared sample canonical correlation")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_wachter)
    plt.close(fig)
    # Maps of kappa and the detection frontier
    g = np.linspace(0.005, 0.6, 240)
    C1g, C2g = np.meshgrid(g, g)
    K = np.full_like(C1g, np.nan)
    F = np.full_like(C1g, np.nan)
    for i in range(len(g)):
        for j in range(len(g)):
            a, b = C1g[i, j], C2g[i, j]
            if a + b < 0.98:
                K[i, j] = kappa(a, b)
                F[i, j] = info_frontier(a, b)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    cs = axes[0].contourf(C1g, C2g, np.log10(K), levels=16, cmap="Blues")
    axes[0].contour(C1g, C2g, np.log10(K), levels=[-3, -2, -1], colors="k", linewidths=0.6)
    fig.colorbar(cs, ax=axes[0], label=r"$\log_{10}\kappa(c_1,c_2)$")
    axes[0].set_title("spurious information per observation", fontsize=8)
    cs = axes[1].contourf(C1g, C2g, F, levels=16, cmap="Oranges")
    axes[1].contour(C1g, C2g, F, levels=[0.05, 0.1, 0.25, 0.5], colors="k", linewidths=0.6)
    fig.colorbar(cs, ax=axes[1], label=r"$I_c(c_1,c_2)$ (nats)")
    axes[1].set_title("minimum detectable rank-one information", fontsize=8)
    for ax in axes:
        ax.set_xlabel("$c_1=p/n$")
        ax.set_ylabel("$c_2=r/n$")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path_maps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX output helpers
# ---------------------------------------------------------------------------

def fmt(x, d=3):
    if isinstance(x, str):
        return x
    if x != x:
        return "--"
    s = f"{x:.{d}f}"
    return r"\ensuremath{-}" + s[1:] if s.startswith("-") else s


def write_tables(out, mc, bt, snap, curves, nd):
    g = out / "generated"
    # Monte Carlo verification table
    groups = {"mp_m2": "Spectral moments and log-determinant",
              "mi_bias": "Spurious information",
              "kl_forward": "Information costs and portfolio risk",
              "bbp_outlier_2": "Spiked covariance (BBP)",
              "mou_kl_rate": "Matrix Ornstein--Uhlenbeck dynamics"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Quantity & Exact or limit & Simulated & MC s.e. & Limit \\"]
    for _, row in mc.iterrows():
        if row["check"] in groups:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{groups[row['check']]}}}}}\\[1pt]")
        lim = row.get("limit", float("nan"))
        lines.append(f"\\quad {row['label']} & {fmt(row['analytic'], 4)} & {fmt(row['simulated'], 4)} & "
                     f"{fmt(row['mc_se'], 4)} & {fmt(lim, 4) if lim == lim else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (g / "tab_mc.tex").write_text("\n".join(lines) + "\n")

    # Practitioner table of exact spurious information
    designs = [(2, 2), (5, 5), (5, 20), (10, 10), (10, 40), (20, 20), (25, 50)]
    lines = [r"\begin{tabular}{r" + "r" * len(designs) + "}", r"\toprule",
             r"$n$ & " + " & ".join(f"$({p},{r})$" for p, r in designs) + r"\\", r"\midrule"]
    for n in (60, 120, 250, 500, 1000, 2500):
        cells = []
        for p, r in designs:
            if n < p + r + 2:
                cells.append("--")
            else:
                cells.append(f"{mi_bias(p, r, n):.3f} \\scriptsize({math.sqrt(mi_null_var(p, r, n)):.3f})")
        lines.append(f"{n} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (g / "tab_practice.tex").write_text("\n".join(lines) + "\n")

    # Backtest
    piv = bt[bt.estimator != "Equal weight"].pivot(index="window", columns="estimator",
                                                    values="realized_vol")
    ratio = bt[bt.estimator == "Sample"].set_index("window")
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             r"$T$ & $q$ & Sample & Clipped & Ledoit--Wolf & RIE & $\hat R_{\rm out}/\hat R_{\rm in}$ & $(1-q)^{-2}$\\",
             r"\midrule"]
    for T in piv.index:
        lines.append(f"{T} & {49 / T:.3f} & " + " & ".join(
            fmt(100 * piv.loc[T, c], 2) for c in ["Sample", "Clipped", "Ledoit-Wolf", "RIE"])
            + f" & {fmt(ratio.loc[T, 'var_ratio'], 2)} & {fmt(ratio.loc[T, 'theory_ratio'], 2)} \\\\")
    ew = bt[bt.estimator == "Equal weight"].realized_vol.iloc[0]
    lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{Equal weight ($1/N$): {100 * ew:.2f}}}\\",
              r"\bottomrule", r"\end{tabular}"]
    (g / "tab_backtest.tex").write_text("\n".join(lines) + "\n")

    # BBP snapshot
    lines = [r"\begin{tabular}{rrrrrl}", r"\toprule",
             r"$k$ & $\lambda_k$ & $\lambda_k/\hat\sigma^2$ & implied $\ell_k$ & predicted $|\langle u_k,v_k\rangle|^2$ & largest loadings\\",
             r"\midrule"]
    for _, row in snap.iterrows():
        lines.append(f"{int(row.k)} & {fmt(row.eigenvalue, 2)} & {fmt(row.scaled, 2)} & "
                     f"{fmt(row.implied_spike, 2)} & {fmt(row.predicted_overlap, 2)} & "
                     f"{row.top_loadings} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (g / "tab_bbp.tex").write_text("\n".join(lines) + "\n")

    # Spurious information curve
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$p=r$ & exact $\E_0\hat I$ & $n\kappa$ & $pr/(2n)$ & simulated & exact sd & simulated sd\\",
             r"\midrule"]
    for _, row in curves.iterrows():
        lines.append(f"{int(row.p)} & {fmt(row.exact)} & {fmt(row.limit)} & {fmt(row.classical)} & "
                     f"{fmt(row.sim_mean)} & {fmt(row.exact_sd)} & {fmt(row.sim_sd)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (g / "tab_spurious.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$m$ & true $I$ & plug-in & predicted & corrected & Ledoit--Wolf & RIE\\", r"\midrule"]
    for _, row in nd.iterrows():
        lines.append(f"{int(row.m)} & {fmt(row.I_true)} & {fmt(row.plug_in)} & "
                     f"{fmt(row.predicted_plug_in)} & {fmt(row.corrected)} & "
                     f"{fmt(row.ledoit_wolf)} & {fmt(row.rie)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (g / "tab_noise.tex").write_text("\n".join(lines) + "\n")


def write_macros(path, nums):
    lines = ["% Generated by reproduction/reproduce.py -- do not edit"]
    for k, v in nums.items():
        v = str(v)
        if v.startswith("-"):
            v = r"\ensuremath{-}" + v[1:]
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reproduction" / "results"))
    ap.add_argument("--paper", default=str(ROOT / "paper"))
    args = ap.parse_args()
    reps = 400 if args.quick else 4000
    res = Path(args.out)
    paper = Path(args.paper)
    for d in (res, paper / "generated", paper / "figures"):
        d.mkdir(parents=True, exist_ok=True)

    nums = {}

    print("Monte Carlo checks ...")
    mc = mc_checks(reps, np.random.default_rng(20261123))
    mc.to_csv(res / "mc_checks.csv", index=False)
    print(mc[["check", "analytic", "simulated", "mc_se", "z"]].to_string())

    print("Spurious information curves ...")
    curves = spurious_curves(reps, np.random.default_rng(11))
    curves.to_csv(res / "spurious_curve.csv", index=False)
    nd = noise_dims(reps, np.random.default_rng(12))
    nd.to_csv(res / "noise_dimensions.csv", index=False)
    cc, edge, thr, dims = cca_phase(reps, np.random.default_rng(13))
    cc.to_csv(res / "cca_phase.csv", index=False)
    print(curves.to_string()); print(nd.to_string()); print(cc.to_string(), edge, thr)

    print("Empirical analysis ...")
    R = load_returns()
    sp = rolling_spectrum(R)
    sp.to_csv(res / "rolling_spectrum.csv", index=False)
    lam, q, s2, T, snap = snapshot(R, "2025-01-01", "2025-12-31")
    snap.to_csv(res / "bbp_snapshot_2025.csv", index=False)
    print(snap.to_string())
    bt = backtest(R)
    bt.to_csv(res / "backtest.csv", index=False)
    print(bt.to_string())
    mi, (p, r, n) = rolling_mi(R, rng=np.random.default_rng(14))
    mi.to_csv(res / "rolling_mi.csv", index=False)

    print("Figures ...")
    fig_mp_snapshot(lam, q, s2, T, paper / "figures" / "mp_snapshot.pdf")
    fig_rolling_spectrum(sp, paper / "figures" / "rolling_spectrum.pdf")
    fig_spurious(curves, nd, paper / "figures" / "spurious.pdf")
    fig_cca(cc, edge, thr, dims, paper / "figures" / "cca_phase.pdf")
    fig_rolling_mi(mi, paper / "figures" / "rolling_mi.pdf")
    fig_dyson(np.random.default_rng(15), paper / "figures" / "dyson.pdf")
    fig_theory(np.random.default_rng(16), paper / "figures" / "mp_family.pdf",
               paper / "figures" / "wachter.pdf", paper / "figures" / "theory_maps.pdf")
    write_tables(paper, mc, bt, snap, curves, nd)

    # ---------------- numbers used in the text
    N = R.shape[1]
    nums["NAssets"] = N
    nums["NDays"] = f"{len(R):,}".replace(",", "{,}")
    nums["SampleStart"] = R.index[0].strftime("%-d %B %Y") if os.name != "nt" else R.index[0].strftime("%#d %B %Y")
    nums["SampleEnd"] = R.index[-1].strftime("%-d %B %Y") if os.name != "nt" else R.index[-1].strftime("%#d %B %Y")
    nums["SnapQ"] = fmt(q, 3)
    nums["SnapT"] = T
    nums["SnapSigma"] = fmt(s2, 3)
    nums["SnapEdge"] = fmt(s2 * (1 + math.sqrt(q)) ** 2, 3)
    nums["SnapLambdaOne"] = fmt(lam[0], 2)
    nums["SnapNOut"] = int(np.sum(lam > s2 * (1 + math.sqrt(q)) ** 2))
    nums["AbsMean"] = fmt(sp.absorption.mean(), 3)
    nums["AbsMax"] = fmt(sp.absorption.max(), 3)
    nums["AbsMaxDate"] = sp.date[sp.absorption.idxmax()].strftime("%B %Y")
    nums["AbsMin"] = fmt(sp.absorption.min(), 3)
    nums["AbsMinDate"] = sp.date[sp.absorption.idxmin()].strftime("%B %Y")
    nums["OutMedian"] = int(sp.n_outliers.median())
    nums["OutMin"] = int(sp.n_outliers.min())
    nums["OutMax"] = int(sp.n_outliers.max())
    nums["OverlapMin"] = fmt(sp.market_overlap.min(), 3)
    nums["NWindows"] = len(sp)
    nums["CorrAbsOut"] = fmt(np.corrcoef(sp.absorption, sp.n_outliers)[0, 1], 2)
    nums["MIp"], nums["MIr"], nums["MIn"] = p, r, n
    nums["MIBias"] = fmt(mi_bias(p, r, n), 4)
    nums["MISd"] = fmt(math.sqrt(mi_null_var(p, r, n)), 4)
    nums["MIPlugMean"] = fmt(mi.plug_in.mean(), 3)
    nums["MICorrMean"] = fmt(mi.corrected.mean(), 3)
    nums["MINullPlugMean"] = fmt(mi.null_plug_in.mean(), 4)
    nums["MINullCorrMean"] = fmt(mi.null_corrected.mean(), 4)
    nums["MINullSd"] = fmt(mi.null_plug_in.std(ddof=1), 4)
    nums["MINullRej"] = fmt(100 * np.mean((mi.null_corrected / math.sqrt(mi_null_var(p, r, n))) > 1.645), 1)
    nums["MIShareSig"] = fmt(100 * np.mean(mi.z > 1.645), 1)
    nums["MICorrMax"] = fmt(mi.corrected.max(), 3)
    nums["MICorrMaxDate"] = mi.date[mi.corrected.idxmax()].strftime("%B %Y")
    nums["MICorrMin"] = fmt(mi.corrected.min(), 3)
    nums["MICorrMinDate"] = mi.date[mi.corrected.idxmin()].strftime("%B %Y")
    for m in (8, 24):
        nums[f"MIPlugNoise{'Eight' if m == 8 else 'TwentyFour'}"] = fmt(mi[f"plug_in_m{m}"].mean(), 3)
        nums[f"MICorrNoise{'Eight' if m == 8 else 'TwentyFour'}"] = fmt(mi[f"corrected_m{m}"].mean(), 3)
        nums[f"MIBiasNoise{'Eight' if m == 8 else 'TwentyFour'}"] = fmt(mi_bias(p, r + m, n), 3)
    nums["MIPartialMean"] = fmt(mi.partial_corrected.mean(), 3)
    nums["MIPartialBias"] = fmt(mi_bias(p, r, n - 1), 4)
    nums["MIPartialShareSig"] = fmt(100 * np.mean(mi.partial_z > 1.645), 1)
    nums["MIPartialMax"] = fmt(mi.partial_corrected.max(), 3)
    nums["MIPartialMaxDate"] = mi.date[mi.partial_corrected.idxmax()].strftime("%B %Y")
    nums["MIPartialMin"] = fmt(mi.partial_corrected.min(), 3)
    nums["MIPartialMinDate"] = mi.date[mi.partial_corrected.idxmin()].strftime("%B %Y")
    nums["MIPartialAbs"] = fmt(np.corrcoef(mi.partial_corrected,
                                           sp.set_index("date").absorption.reindex(mi.date).values)[0, 1], 2)
    nums["MIPartialShare"] = fmt(100 * (mi.partial_corrected / mi.corrected).median(), 1)
    nums["MICorrAbs"] = fmt(np.corrcoef(mi.corrected,
                                        sp.set_index("date").absorption.reindex(mi.date).values)[0, 1], 2)
    nums["KLfwdN"] = fmt(kl_forward_exact(49, 249), 3)
    nums["KLrevN"] = fmt(kl_reverse_exact(49, 249), 3)
    nums["KLfwdSixty"] = fmt(kl_forward_exact(49, 59), 2)
    nums["KLrevSixty"] = fmt(kl_reverse_exact(49, 59), 2)
    nums["CCAEdge"] = fmt(edge, 3)
    nums["CCAThr"] = fmt(thr, 3)
    nums["CCAThrSq"] = fmt(thr ** 2, 3)
    nums["FrontierSim"] = fmt(info_frontier(dims[1] / dims[0], dims[2] / dims[0]), 4)
    nums["FrontierEmp"] = fmt(info_frontier(p / n, r / n), 4)
    nums["FrontierEmpT"] = fmt(cca_threshold(p / n, r / n), 3)
    nums["CCAsixA"] = fmt(cca_outlier(0.6, 0.1, 0.2), 4)
    nums["CCAsixS"] = fmt(cc.loc[(cc.t - 0.6).abs().idxmin(), "top_cc2"], 4)
    nums["CCAeightA"] = fmt(cca_outlier(0.8, 0.1, 0.2), 4)
    nums["CCAeightS"] = fmt(cc.loc[(cc.t - 0.8).abs().idxmin(), "top_cc2"], 4)
    nums["MOUrate"] = fmt(mou_kl_rate(10, 0.5, 1.0), 2)
    ews = bt[bt.estimator == "Equal weight"].realized_vol.iloc[0]
    nums["EWVol"] = fmt(100 * ews, 2)
    for T in (60, 120, 250, 500):
        tag = {60: "Sixty", 120: "OneTwenty", 250: "TwoFifty", 500: "FiveHundred"}[T]
        sub = bt[bt.window == T].set_index("estimator")
        for e, et in (("Sample", "Sample"), ("Clipped", "Clip"), ("Ledoit-Wolf", "LW"), ("RIE", "RIE")):
            nums[f"Vol{et}{tag}"] = fmt(100 * sub.loc[e, "realized_vol"], 2)
        nums[f"Ratio{tag}"] = fmt(sub.loc["Sample", "var_ratio"], 2)
        nums[f"Theory{tag}"] = fmt(sub.loc["Sample", "theory_ratio"], 2)
    for _, row in mc.iterrows():
        key = "MC" + "".join(w.capitalize() for w in row["check"].replace(".", "p").split("_"))
        key = key.replace("2", "Two").replace("4", "Four").replace("3", "Three")
        key = "".join(ch for ch in key if ch.isalpha())
        nums[key + "A"] = fmt(row["analytic"], 4)
        nums[key + "S"] = fmt(row["simulated"], 4)
    nums["MCRepsQuarter"] = f"{reps // 4:,}".replace(",", "{,}")
    nums["MCReps"] = f"{reps:,}".replace(",", "{,}")

    write_macros(paper / "generated" / "numbers.tex", nums)
    (res / "numbers.json").write_text(json.dumps(nums, indent=1, default=str))
    print(json.dumps(nums, indent=1, default=str))


if __name__ == "__main__":
    main()
