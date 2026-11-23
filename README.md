# Spurious Information in Large Markets

**Random Matrix Laws for Entropy, Mutual Information and the Value of Information**

James K Rice, University of Essex — November 2026

This repository contains the manuscript, data, reproduction code, simulation outputs and Lean 4 formalization for the paper. The paper develops the random matrix theory of Gaussian information functionals. It covers exact and proportional-regime laws for plug-in mutual information and Kullback–Leibler costs, a closed form for the logarithmic moment of the Wachter law, a canonical-correlation frontier for detectable information, an exact CARA–Gaussian value-of-information identity, and the informational sufficiency of Dyson eigenvalue dynamics. It applies these results to 36 years of daily returns on the 49 Fama–French industry portfolios.

- Paper: [paper/main.pdf](paper/main.pdf) (51 pages)
- Source: [paper/main.tex](paper/main.tex), [paper/sections/](paper/sections), [paper/references.bib](paper/references.bib)
- Verification record: [VERIFICATION.md](VERIFICATION.md)

## Layout

| Path | Contents |
| --- | --- |
| `paper/` | LaTeX source, bibliography, compiled PDF and checked-in `main.bbl` |
| `paper/generated/` | LaTeX macros (`numbers.tex`) and tables written by the reproduction script |
| `paper/figures/` | All figures, written by the reproduction script |
| `reproduction/reproduce.py` | Single script that produces every number, table and figure |
| `reproduction/audit_bibliography.py` | Crossref audit of `references.bib` |
| `reproduction/results/` | Committed CSV/JSON outputs |
| `data/` | Daily value-weighted returns of the 49 Fama–French industries, 1990–2026 |
| `lean/RMTChecks.lean` | Lean 4 proofs (28 theorems, core Lean only) |

## Reproduce the numbers, tables and figures

Python 3.11 with the pinned packages:

    python -m pip install -r requirements.txt
    python reproduction/reproduce.py            # full run, about 4 minutes
    python reproduction/reproduce.py --quick    # reduced Monte Carlo sizes

The script regenerates `reproduction/results/*`, `paper/generated/*` and `paper/figures/*` from fixed seeds and the committed data file. No network access is needed. Last-digit differences in Monte Carlo quantities can occur across platforms and BLAS libraries.

| Output | Paper |
| --- | --- |
| `mc_checks.csv`, `generated/tab_mc.tex` | Table 1: Monte Carlo verification of all exact and limiting formulas |
| `spurious_curve.csv`, `noise_dimensions.csv`, `figures/spurious.pdf` | Figure 5, Tables 6–7 |
| `cca_phase.csv`, `figures/cca_phase.pdf` | Figure 6: canonical-correlation phase transition |
| `figures/mp_family.pdf`, `figures/wachter.pdf`, `figures/theory_maps.pdf`, `figures/dyson.pdf` | Figures 1–4 |
| `rolling_spectrum.csv`, `figures/rolling_spectrum.pdf`, `figures/mp_snapshot.pdf` | Figures 7–8 |
| `bbp_snapshot_2025.csv`, `generated/tab_bbp.tex` | Table 2: BBP triage of 2025 eigenvalues |
| `backtest.csv`, `generated/tab_backtest.tex` | Table 3: minimum-variance backtest |
| `rolling_mi.csv`, `figures/rolling_mi.pdf` | Figure 9: carbon-intensive/financial information linkage |
| `generated/tab_practice.tex` | Table 5: look-up table of exact spurious information |
| `numbers.json`, `generated/numbers.tex` | Every number quoted in the text |

## Build the paper

With a TeX distribution providing pdfLaTeX, BibTeX, natbib, amsmath, booktabs, listings, placeins and hyperref:

    cd paper
    pdflatex main && bibtex main && pdflatex main && pdflatex main

The checked-in `main.bbl` is sufficient if the citations are unchanged.

## Check the Lean formalization

With Lean installed through [elan](https://github.com/leanprover/elan):

    cd lean
    lean RMTChecks.lean

The toolchain is pinned to Lean 4.29.0 (`lean/lean-toolchain`), and the file uses only the core distribution, without mathlib. It proves the following:

- the Marchenko–Pastur moment recursion yields the Narayana polynomials, and the Wigner recursion yields the Catalan numbers (kernel computation);
- the telescoping, positivity, symmetry and strict monotonicity in uninformative coordinates of the spurious-information formula, for any antitone sequence;
- the BBP and canonical-correlation threshold identities;
- the equality of the entrywise and eigenvalue (Dyson) information rates of the matrix Ornstein–Uhlenbeck process;
- the Stein-loss, portfolio and CARA algebraic identities.

The file contains no `sorry` and no `native_decide`. The measure-theoretic arguments are proved analytically in Appendix A of the paper.

## Data

`data/ff49_vw_daily_1990_2026.csv` contains the value-weighted daily returns, in percent, of the 49 industry portfolios from 2 January 1990 to 30 September 2026. They are extracted from `49_Industry_Portfolios_Daily_CSV.zip` (CRSP 202609 vintage) in the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), downloaded on 23 November 2026, with SHA-256 `b9d36d571e968d71f7b03207e7fed6398554f953719b8c1bf3ee3ad8d00e7520`. All 49 series are complete over this period. Please cite Fama and French (1997) and the Data Library when using these data.

## Citation

See [CITATION.cff](CITATION.cff).
