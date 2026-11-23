# Verification record

## Mathematical results

| Result in the paper | How it is checked |
| --- | --- |
| Exact log-determinant bias and variance (Bartlett) | Analytic proof (Appendix A). Monte Carlo check at N = 100, n = 400 (Table 1). |
| Second-order term ½ log(1 − q) | Euler–Maclaurin argument (Appendix A). The exact value, the limit and the simulation are reported together in Table 1. |
| Exact Wishart moments N⁻¹E tr S², N⁻¹E tr S³ | Classical Wishart moments. Monte Carlo check in Table 1. |
| Marchenko–Pastur moments as Narayana polynomials; semicircle moments as Catalan numbers | Lean, by exact kernel computation up to order 10 and 12 (`mp_moments_are_narayana`, `semicircle_moments_catalan`). |
| Exact spurious information E Î − I = B_n(p, r), independent of Σ | Analytic proof. Lean proves the telescoping identity, positivity and symmetry for any antitone sequence (`twiceBias_*`). Monte Carlo check with a random Σ of condition number 20. |
| Null law of Î (Wilks Λ as a product of Betas); exact mean and variance | Analytic proof (Appendix A). Monte Carlo check of the mean and variance. |
| Proportional limits κ(c₁, c₂), ν(c₁, c₂) | Riemann-sum proof (Appendix A). Figure 5 and Table 6 compare exact, limiting and simulated values. |
| Logarithmic moment of the Wachter law | Proof by identifying two limits of Î/n (Appendix A). Independent numerical quadrature agrees to 1e-13 on a grid of (c₁, c₂). |
| Monotonicity in uninformative coordinates | Lean (`twiceBias_noise`, `twiceBias_noise_pos`). Simulation (Table 7) and the empirical noise-augmentation experiment. |
| Exact KL costs (Stein loss and entropy loss) | Analytic proof. Lean certifies the trace identity (`stein_trace`). Monte Carlo check. |
| Minimum-variance in- and out-of-sample ratios (n − N + 1)/n and (n − 1)/(n − N) | Analytic proof via the Schur complement and block inverse (Appendix A). Monte Carlo check. |
| BBP outlier location, overlap and inversion | Subordination derivation. Lean certifies the threshold and inversion identities. Monte Carlo check at ℓ = 2 and ℓ = 4. |
| Canonical-correlation threshold t_c and outlier map γ(t) | The threshold and the outlier map are confirmed by simulation (Figure 6), and Lean proves γ(t_c) = ω₊. The general finite-rank theory is due to Bao, Hu, Pan and Zhou (2019). The parametrisation used in the paper is checked numerically, not re-derived. |
| Eigenvector part of the frontier (estimated canonical direction uninformative below t_c) | **Not proved.** It is stated as a conjecture by analogy with the BBP eigenvector transition. |
| CARA value of information = I(v; s)/γ | Analytic proof (Appendix A). Lean certifies the scalar determinant and inverse identities. |
| Matrix OU information rate N(N + 1)(b − a)²/(8a) and Dyson sufficiency | Girsanov argument (Appendix A). Lean proves that the entrywise and eigenvalue rates coincide. Monte Carlo check of the Euler log-likelihood ratio (N = 10). |
| Forced Burgers equation and stationary semicircle | Itô derivation (Appendix A). Figure 4 compares the stationary spectrum with the predicted semicircle. |

## Lean

`lean/RMTChecks.lean` contains 28 theorems. It compiles with `lean RMTChecks.lean` under Lean 4.29.0 using only the core library, with no `sorry`, no `native_decide` and no axioms beyond Lean's standard ones. Kernel computations use `decide +kernel`. Symbolic statements use induction and the `grind` decision procedure.

## Empirical robustness check of the Gaussian theory

In the circular-shift placebo, the plug-in information has a mean of 0.0653 nats against the exact prediction of 0.0660, a standard deviation of 0.0171 against the exact 0.0165, and a corrected-test rejection rate of 6.1% at a nominal 5%. This holds even though returns are non-Gaussian and heteroskedastic.

## Bibliography

`reproduction/audit_bibliography.py` checks all 154 entries in `paper/references.bib` against Crossref and writes `reproduction/results/bibliography_audit.csv`. In the final run, 124 entries matched on title, year, volume and first page, and 2 non-bibliographic entries (data library and arXiv preprint) were skipped. The 28 flagged entries were inspected individually and fall into four groups:

- **Online-first year differs from the issue year.** The bibliography uses the issue year: `anderson2010introduction`, `akemann2011oxford`, `bloemendal2016principal`, `knowles2017anisotropic`, `ledoit2011eigenvectors`, `ledoit2022power`, `engle2019large`, `demiguel2009optimal`, `kelly2024virtue`, `cover2006elements`, `kadan2019estimating`.
- **Crossref title lacks the subtitle or encodes mathematics as markup.** `livan2018introduction`, `potters2020first`, `gavish2014optimal`, `engle2002dynamic`, `mantegna2000introduction`, `bouchaud2003theory`. `laloux2000random` differs only in volume formatting ("03" vs "3").
- **No DOI in the entry, and the Crossref search returned an unrelated record.** These were checked against the publisher or proceedings record: `mehta2004random`, `paul2007asymptotics` (Statistica Sinica 17(4), 1617–1642), `anderson2003introduction`, `james1961estimation` (Fourth Berkeley Symposium, vol. 1, 361–379), `belghazi2018mutual` (PMLR 80, 531–540), `mcallester2020formal` (PMLR 108, 875–884), `li2020scalable` (PMLR 108, 3870–3882), `grossman1980impossibility` (AER 70(3), 393–408), `veldkamp2011information`.
- **Handbook chapter.** For `bouchaud2011financial`, Crossref gives a first page of 823. The page range has been removed from the entry rather than guessed.

Two DOIs found to be wrong in the first audit were corrected: `stiglitz2000contributions` (now 10.1162/003355300555015) and `kadan2019estimating` (now 10.1093/rfs/hhy087).
