# rdq-ST2HFS DSS

Data-driven r,d,q-spherical type-2 hesitant fuzzy TOPSIS decision support system for financial asset selection
(no expert grading: all fuzzy grades are computed from market data).
This repository reproduces every number, table and figure of the manuscript
"A hybrid fuzzy similarity framework for financial asset selection: A data-driven decision support system
based on r,d,q-spherical type-2 hesitant fuzzy sets".

## Contents
- `code/rdq_st2hfs.py`: library with data loading, monthly criteria, data-driven construction of r,d,q-ST2HFEs,
  admissibility check and rescaling, CRITIC weights, weighted hybrid similarity measure, TOPSIS (Steps 1-5),
  ablations and benchmarks (SF-TOPSIS, HF-TOPSIS, TOPSIS, WASPAS, EDAS, CoCoSo).
  The simple baselines (rank average, TOPSIS on ranks, low-volatility and low-beta sorts) are defined in `run_experiments.py`.
- `code/run_experiments.py`: the whole study, i.e. design phase (1976-1990), main case, agreement, robustness,
  rank reversal, out-of-sample evaluation (moving-block bootstrap tests, equivalence test), portfolios,
  sensitivity analyses, reliability flag of the secondary grades, decision report, scalability.
- `code/data/ff49_daily_vw.csv`, `code/data/ff_factors_daily.csv`: daily value-weighted returns of the 49 Fama-French
  industry portfolios and daily factors (Mkt-RF, SMB, HML, RMW, CMA, RF), January 1971 - April 2021, in decimals.
  Source: Kenneth R. French Data Library, retrieved through the FFresearch R package (github.com/bautheac/FFresearch).
- `results/`, `tables/`, `figures/`, `numbers.tex`: the outputs of `run_experiments.py` as used in the manuscript
  (CSV results, LaTeX tables, PDF figures and the LaTeX macros that hold the numbers quoted in the text).

## Run
    cd code
    pip install -r requirements.txt
    python run_experiments.py          # about 5 minutes on one CPU core

The script overwrites `results/*.csv`, `tables/*.tex`, `figures/*.pdf` and `numbers.tex` in the repository root.
Random seeds are fixed, so results are reproducible for a given data file.

## Using newer data
Download the two official files from the Kenneth R. French Data Library and place them in `code/data/`:
- `49_Industry_Portfolios_Daily_CSV.zip`
- `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`

They are detected automatically; the hold-out period then extends to the last full calendar year.
