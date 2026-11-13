"""
Reproduces every number, table and figure of the manuscript
"A hybrid fuzzy similarity framework for financial asset selection: a data-driven decision support
system based on r,d,q-spherical type-2 hesitant fuzzy sets".

Application: the 49 Fama-French industry portfolios (daily value-weighted returns).
Protocol   : design period  = decisions 1976-1990 (application settings are chosen here),
             hold-out period = decisions 1991-last full year (all reported comparisons).

Usage:  python run_experiments.py            (about 5 minutes on one CPU core)
Output: ../results/*.csv, ../tables/*.tex, ../figures/*.pdf, ../numbers.tex
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, wilcoxon, friedmanchisquare, studentized_range, ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rdq_st2hfs as L

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES, TAB, FIG = (os.path.join(ROOT, d) for d in ("results", "tables", "figures"))
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)

K = 3                      # years (hesitant elements) per evaluation window
FIRST_DECISION = 1976      # first decision year (window 1973-1975)
DESIGN_END = 1990          # last decision year of the design period
TOPK = 10                  # size of the selected portfolio (top quintile of 49)
B_ROB = 100                # perturbation replications per window and scheme
B_DESIGN = 50              # replications per window and scheme in the design phase
SETTING = dict(r=1, d=1, q=1, lam=0.5)      # selected in the design phase (Section 7.2)
SEED = 2026

PROP = "Proposed"
METHODS = [PROP, "Abl-1 (no secondary grades)", "Abl-2 (no hesitancy)", "Abl-3 (no rescaling)",
           "SF-TOPSIS", "HF-TOPSIS", "TOPSIS", "WASPAS", "EDAS", "CoCoSo",
           "Rank average", "TOPSIS on ranks", "Low volatility", "Low beta"]
SHORT = dict(zip(METHODS, ["Prop", "AOne", "ATwo", "AThree", "SF", "HF", "TOP", "WAS", "EDAS", "CoCo",
                           "RA", "RT", "LV", "LB"]))
ABBR = dict(zip(METHODS, ["Prop.", "A1", "A2", "A3", "SF", "HF", "TOP", "WAS", "EDAS", "CoCo", "RA", "RT", "LV", "LB"]))
FUZZY, CRISP = ["SF-TOPSIS", "HF-TOPSIS"], ["TOPSIS", "WASPAS", "EDAS", "CoCoSo"]
# simple non-fuzzy baselines that use the same rank information as the fuzzification (added in response to review)
RANKB, SORTB = ["Rank average", "TOPSIS on ranks"], ["Low volatility", "Low beta"]
BASELINES = RANKB + SORTB

ACCENT, GREY, DARK, LIGHT = "#1f5fbf", "#9a9a92", "#3b3b38", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4aa", "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "pdf.fonttype": 42})
MAC = {}


# ============================================================================ helpers
def mac(name, val, fmt="{:.3f}"):
    assert name.isalpha(), name
    MAC[name] = val if isinstance(val, str) else fmt.format(val)


def fp(p):
    """p-value as it appears after the letter p in the text."""
    return "<0.001" if p < 0.001 else "={:.3f}".format(p)


def holm(pvals):
    """Holm step-down adjustment of a dict of p-values."""
    keys = [k for k, v in pvals.items() if np.isfinite(v)]
    order = sorted(keys, key=lambda k: pvals[k])
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return {k: adj.get(k, np.nan) for k in pvals}


def safe_wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    return np.nan if np.allclose(d, 0) else wilcoxon(x, y).pvalue


def block_boot(d, block=3, B=20000, seed=SEED):
    """Moving-block bootstrap for the mean of a serially dependent series d (paired annual differences).
    Returns (mean, two-sided p-value for H0: mean = 0, 95% CI, 90% CI). Blocks of `block` years preserve the
    dependence created by overlapping three-year estimation windows."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    m = d[idx].mean(axis=1)
    p = min(1.0, 2 * min(((m - d.mean()) >= abs(d.mean())).mean(), ((m - d.mean()) <= -abs(d.mean())).mean()) + 1 / B)
    return d.mean(), p, np.quantile(m, [0.025, 0.975]), np.quantile(m, [0.05, 0.95])


def nemenyi_cd(k, n, alpha=0.05):
    return studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))


def write_table(path, caption, label, body, size="\\footnotesize", note=None):
    body = body.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}", 1)
    body = body.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
    with open(path, "w") as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n" + size + "\n\\caption{" + caption + "}\\label{" + label + "}\n"
                + body.rstrip() + "\n" + (("\\par\\smallskip\\parbox{\\textwidth}{\\scriptsize " + note + "}\n") if note else "")
                + "\\end{table}\n")


def df_tex(df, fmt="{:.3f}", index=True, colfmt=None):
    df = df.copy()
    df.index.name = None
    df.columns.name = None
    return df.to_latex(index=index, float_format=lambda v: fmt.format(v), escape=False, column_format=colfmt, na_rep="--")


def tex_name(code):
    return L.IND_NAMES[code].replace("&", "\\&")


def enum(names):
    return ", ".join(names[:-1]) + " and " + names[-1]


def cd_plot(avg_ranks, cd, path):
    """Critical-difference diagram (Demsar, 2006)."""
    r = avg_ranks.sort_values()
    k = len(r)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0.6, k + 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.86, 1, k, color=DARK, lw=0.8)
    for t in range(1, k + 1):
        ax.vlines(t, 0.84, 0.88, color=DARK, lw=0.8)
        ax.text(t, 0.90, str(t), ha="center", va="bottom", fontsize=8)
    ax.text((1 + k) / 2, 0.99, "Average rank (1 = best)", ha="center", va="bottom", fontsize=8)
    ax.hlines(0.74, 1, 1 + cd, color=DARK, lw=1.6)
    ax.text(1 + cd / 2, 0.76, f"CD = {cd:.2f}", ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (m, v) in enumerate(r.items()):
        left = i < half
        y = 0.60 - 0.095 * (i if left else (k - 1 - i))
        xt = 0.7 if left else k + 0.3
        col = ACCENT if m == PROP else DARK
        ax.plot([v, v, xt], [0.86, y, y], color=col, lw=1.6 if m == PROP else 0.8)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{m} ({v:.2f})", ha="right" if left else "left", va="center",
                fontsize=7.5, color=DARK, fontweight="bold" if m == PROP else "normal")
    vals, cliques = r.values, []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        ax.hlines(0.82 - 0.03 * c, vals[i] - 0.04, vals[j] + 0.04, color=GREY, lw=3, alpha=0.85)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ data
t_start = time.perf_counter()
R, F, SOURCE = L.load_data(os.path.join(HERE, "data"))
P = L.price_index(R)
MKT = F.mktrf + F.rf
LAST = R.index.max().year - (0 if R.index.max().month == 12 else 1)       # last full calendar year
MONTHLY = L.monthly_criteria(P, MKT, f"{FIRST_DECISION - K}-01", f"{LAST}-12")
DESIGN_YEARS = list(range(FIRST_DECISION, DESIGN_END + 1))
HOLDOUT_YEARS = list(range(DESIGN_END + 1, LAST + 1))
DESIGN_WINDOWS = [list(range(y, y + K)) for y in range(FIRST_DECISION - K, DESIGN_END - K + 2, K)]
HOLDOUT_WINDOWS = [list(range(y, y + K)) for y in range(DESIGN_END + 1, LAST - K + 2, K)]
MAIN_YEARS = HOLDOUT_WINDOWS[-1]
print(f"data: {SOURCE}, {R.index.min().date()} to {R.index.max().date()}; last full year {LAST}")
print("design windows", DESIGN_WINDOWS[0], "...", DESIGN_WINDOWS[-1], "| hold-out windows", HOLDOUT_WINDOWS[0], "...", HOLDOUT_WINDOWS[-1])


def window(years, thresh=1 / 3):
    mon = {m: MONTHLY[m] for m in MONTHLY if m.year in years}
    assets, E = L.build_st2hfe(mon, years, thresh)
    X = L.crisp_matrix(mon, years).loc[assets]
    return mon, assets, E, X, L.critic(X.values)


def mean_ranks(mon):
    """Window-average of the monthly cross-sectional percentile ranks (assets x criteria, oriented so that
    larger is better): exactly the information from which the fuzzy grades are built, without classification."""
    out = []
    for m in mon:
        Xm = mon[m].values.copy()
        Xm[:, ~L.BENEFIT] *= -1
        out.append(pd.DataFrame(Xm).rank(pct=True, method="average").values)
    return pd.DataFrame(np.mean(out, axis=0), index=next(iter(mon.values())).index, columns=L.CRITERIA)


def topsis_benefit(Q, w):
    """Crisp TOPSIS with all criteria treated as benefit criteria (used on oriented percentile ranks)."""
    V = Q / np.sqrt((Q ** 2).sum(0)) * w
    dp, dm = np.sqrt(((V - V.max(0)) ** 2).sum(1)), np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return dm / (dp + dm)


def score_all(E, X, w, Q):
    """Decision scores (larger = better) of the framework, its ablations and the benchmarks.
    Q: window-average oriented percentile ranks (see mean_ranks), rows aligned with X."""
    s = SETTING
    rc = L.st2hfs_topsis(L.rescale(E), w, s["r"], s["d"], s["q"], s["lam"])[0]
    k = L.minimal_common_exponent(E)
    return pd.DataFrame({
        PROP: rc,
        METHODS[1]: L.st2hfs_topsis(L.rescale(L.drop_secondary(E)), w, s["r"], s["d"], s["q"], s["lam"])[0],
        METHODS[2]: L.st2hfs_topsis(L.rescale(L.drop_hesitancy(E)), w, s["r"], s["d"], s["q"], s["lam"])[0],
        METHODS[3]: L.st2hfs_topsis(E, w, k, k, k, s["lam"])[0],
        "SF-TOPSIS": L.sf_topsis(E, w), "HF-TOPSIS": L.hf_topsis(E, w),
        "TOPSIS": L.topsis(X.values, w), "WASPAS": L.waspas(X.values, w),
        "EDAS": L.edas(X.values, w), "CoCoSo": L.cocoso(X.values, w),
        "Rank average": np.asarray(Q.values) @ w, "TOPSIS on ranks": topsis_benefit(np.asarray(Q.values, float), w),
        "Low volatility": -X.iloc[:, 2].values, "Low beta": -X.iloc[:, 5].values}, index=X.index)


def realised(year, assets, w):
    """Out-of-sample outcomes of `assets` in `year` (all oriented so that larger is better):
    Sharpe ratio (excess of the risk-free rate), return, and the criteria attainment U, the CRITIC-weighted
    mean percentile rank over the six criteria that can be measured within the year."""
    r = R.loc[str(year), assets]
    mk = MKT.loc[str(year)]
    px = (1 + r).cumprod()
    crit = pd.DataFrame({"ret": (1 + r).prod() - 1, "rtr": r.mean() / r.std(), "vol": -r.std(),
                         "var": r.quantile(0.05), "mdd": (px / px.cummax() - 1).min(),
                         "beta": -r.apply(lambda c: np.cov(c, mk)[0, 1] / mk.var())})
    w6 = w[:6] / w[:6].sum()
    return dict(sharpe=r.sub(F.rf.loc[str(year)], axis=0).mean() / r.std() * np.sqrt(252), ret=crit.ret,
                attain=(crit.rank(pct=True) * w6).sum(axis=1), lowvol=crit.vol)


def perturb(mon, years, rng, sd):
    """One month-bootstrap and one outlier-contaminated copy of the monthly criteria of a window."""
    months = list(mon)
    boot = {}
    for y in years:
        ms = [m for m in months if m.year == y]
        for kk, i in enumerate(rng.choice(len(ms), len(ms))):
            boot[pd.Period(f"{y}-{kk + 1:02d}", "M")] = mon[ms[i]]
    out = {m: df + (rng.random(df.shape) < 0.10) * rng.choice([-1, 1], size=df.shape) * 5 * sd for m, df in mon.items()}
    # small Gaussian noise on every observation: moves observations across the class boundaries without outliers
    noise = {m: df + rng.normal(0, 0.25, size=df.shape) * sd for m, df in mon.items()}
    return boot, out, noise


def crit_sd(mon):
    return np.array([pd.concat([mon[m][c] for m in mon]).std() for c in L.CRITERIA])


# ============================================================================ 1. admissibility of the raw grades
adm = []
for ty in range(FIRST_DECISION, LAST + 2):
    yrs = list(range(ty - K, ty))
    _, _, E, _, _ = window(yrs)
    adm.append(dict(window=f"{yrs[0]}-{yrs[-1]}", raw_111=L.admissible_share(E, 1, 1, 1),
                    raw_222=L.admissible_share(E, 2, 2, 2), raw_333=L.admissible_share(E, 3, 3, 3),
                    min_k=L.minimal_common_exponent(E), max_sum=L.max_primary_sum(E).max(),
                    share_gt1=float((L.max_primary_sum(E) > 1).mean()),
                    rescaled_111=L.admissible_share(L.rescale(E), 1, 1, 1)))
ADM = pd.DataFrame(adm)
ADM.to_csv(os.path.join(RES, "admissibility.csv"), index=False)
mac("AdmWindows", len(ADM), "{:d}")
mac("AdmRawOne", 100 * ADM.raw_111.mean(), "{:.0f}")
mac("AdmRawTwo", 100 * ADM.raw_222.mean(), "{:.0f}")
mac("AdmRawThree", 100 * ADM.raw_333.mean(), "{:.0f}")
mac("AdmShareGtOne", 100 * ADM.share_gt1.mean(), "{:.0f}")
mac("AdmMinKLow", int(ADM.min_k.min()), "{:d}")
mac("AdmMinKHigh", int(ADM.min_k.max()), "{:d}")
mac("AdmMaxSumLow", ADM.max_sum.min(), "{:.2f}")
mac("AdmMaxSumHigh", ADM.max_sum.max(), "{:.2f}")
mac("AdmRescaledOne", 100 * ADM.rescaled_111.min(), "{:.0f}")

# ============================================================================ 2. design phase (design period only)
print("design phase ...", flush=True)


def setting_score(E, w, rescaled, rdq, lam):
    if rdq == "min":
        k = L.minimal_common_exponent(E)
        rdq = (k, k, k)
    return L.st2hfs_topsis(L.rescale(E) if rescaled else E, w, *rdq, lam)[0]


GRID = [("Raw grades", False, "min"), ("Raw grades", False, (9, 7, 8)),
        ("Rescaled grades", True, (1, 1, 1)), ("Rescaled grades", True, (2, 2, 2)), ("Rescaled grades", True, (3, 3, 3))]
LAMS = [0.0, 0.5, 1.0]


def evaluate_settings(decision_years, windows, B, seed):
    rows = []
    cache_w = {tuple(y): window(y) for y in windows}
    cache_d = {}
    for ty in decision_years:
        mon, a, E, X, w = window(list(range(ty - K, ty)))
        cache_d[ty] = (E, w, realised(ty, a, w))
    for name, resc, rdq in GRID:
        for lam in LAMS:
            rng = np.random.default_rng(seed)
            tb, to = [], []
            for yrs, (mon, a, E, X, w) in cache_w.items():
                base = setting_score(E, w, resc, rdq, lam)
                sd = crit_sd(mon)
                for _ in range(B):
                    bm, om, _ = perturb(mon, list(yrs), rng, sd)
                    tb.append(kendalltau(base, setting_score(L.build_st2hfe(bm, list(yrs))[1], w, resc, rdq, lam)).statistic)
                    to.append(kendalltau(base, setting_score(L.build_st2hfe(om, list(yrs))[1], w, resc, rdq, lam)).statistic)
            ra, rs = [], []
            for ty, (E, w, out) in cache_d.items():
                sc = setting_score(E, w, resc, rdq, lam)
                ra.append(spearmanr(sc, out["attain"]).correlation)
                rs.append(spearmanr(sc, out["sharpe"]).correlation)
            rows.append(dict(grades=name, rdq="minimal admissible $k$" if rdq == "min" else str(rdq).replace(" ", ""),
                             lam=lam, boot=np.mean(tb), outlier=np.mean(to), attain=np.mean(ra), sharpe=np.mean(rs)))
    return pd.DataFrame(rows)


DES = evaluate_settings(DESIGN_YEARS, DESIGN_WINDOWS, B_DESIGN, SEED)
DES.to_csv(os.path.join(RES, "design_phase.csv"), index=False)
DES["stab"] = (DES.boot + DES.outlier) / 2
best = DES.sort_values(["stab", "attain"], ascending=False).iloc[0]
sel = DES[(DES.grades == "Rescaled grades") & (DES.rdq == "(1,1,1)")]
assert best.grades == "Rescaled grades" and best.rdq == "(1,1,1)", best
mac("DesN", len(DESIGN_YEARS), "{:d}")
mac("DesYears", f"{DESIGN_YEARS[0]}--{DESIGN_YEARS[-1]}")
mac("DesWindows", len(DESIGN_WINDOWS), "{:d}")
mac("DesB", B_DESIGN, "{:d}")
mac("DesSelStabMin", sel.stab.min())
mac("DesSelStabMax", sel.stab.max())
mac("DesSelAttainMin", sel.attain.min())
mac("DesSelAttainMax", sel.attain.max())
others = DES.drop(sel.index)
mac("DesOtherStabMax", others.stab.max())
mac("DesOtherAttainMax", others.attain.max())
raw = DES[DES.grades == "Raw grades"]
mac("DesRawStabMin", raw.stab.min())
mac("DesRawStabMax", raw.stab.max())
mac("DesRawAttainMin", raw.attain.min())
mac("DesRawAttainMax", raw.attain.max())
# classification thresholds, also chosen on the design period
thr = []
for th, thname in [(1 / 3, "Terciles"), (1 / 4, "Quartiles"), (1 / 5, "Quintiles")]:
    rng = np.random.default_rng(SEED)
    tb, to, ra = [], [], []
    for yrs in DESIGN_WINDOWS:
        mon_, a_, E_, X_, w_ = window(yrs, th)
        base = L.st2hfs_topsis(L.rescale(E_), w_, **SETTING)[0]
        sd = crit_sd(mon_)
        for _ in range(B_DESIGN):
            bm, om, _ = perturb(mon_, yrs, rng, sd)
            tb.append(kendalltau(base, L.st2hfs_topsis(L.rescale(L.build_st2hfe(bm, yrs, th)[1]), w_, **SETTING)[0]).statistic)
            to.append(kendalltau(base, L.st2hfs_topsis(L.rescale(L.build_st2hfe(om, yrs, th)[1]), w_, **SETTING)[0]).statistic)
    for ty in DESIGN_YEARS:
        mon_, a_, E_, X_, w_ = window(list(range(ty - K, ty)), th)
        ra.append(spearmanr(L.st2hfs_topsis(L.rescale(E_), w_, **SETTING)[0], realised(ty, a_, w_)["attain"]).correlation)
    thr.append(dict(classes=thname, stab=(np.mean(tb) + np.mean(to)) / 2, attain=np.mean(ra)))
THR = pd.DataFrame(thr).set_index("classes")
THR.to_csv(os.path.join(RES, "design_phase_thresholds.csv"))
assert THR.stab.idxmax() == "Terciles" and THR.attain.idxmax() == "Terciles", THR
for c_, key in [("Terciles", "Terc"), ("Quartiles", "Quart"), ("Quintiles", "Quint")]:
    mac("DesThStab" + key, THR.loc[c_, "stab"])
    mac("DesThU" + key, THR.loc[c_, "attain"])
T = DES[["grades", "rdq", "lam", "boot", "outlier", "attain", "sharpe"]].copy()
T["lam"] = T["lam"].map("{:.1f}".format)
T.columns = ["Grades", "$r,d,q$", "$\\lambda$", "Bootstrap $\\tau$", "Outlier $\\tau$", "$\\rho_U$", "$\\rho_S$"]
write_table(os.path.join(TAB, "tab_design.tex"),
            f"Design phase ({DESIGN_YEARS[0]}--{DESIGN_YEARS[-1]} only): ranking stability (Kendall's $\\tau$; {len(DESIGN_WINDOWS)} "
            f"three-year windows, {B_DESIGN} replications per window and scheme) and mean out-of-sample rank correlation of the scores "
            f"with next-year criteria attainment ($\\rho_U$) and Sharpe ratio ($\\rho_S$) over {len(DESIGN_YEARS)} annual decisions, "
            "for alternative application settings. The selected setting is rescaled grades with $r=d=q=1$ and $\\lambda=0.5$.",
            "tab:design", df_tex(T, index=False, colfmt="lllrrrr"))

# ============================================================================ 3. main case (most recent window)
print("main case ...", flush=True)
mon, A, E, X, w = window(MAIN_YEARS)
n_alt = len(A)
mac("NAlt", n_alt, "{:d}")
mac("MainYears", f"{MAIN_YEARS[0]}--{MAIN_YEARS[-1]}")
mac("LastYear", LAST, "{:d}")
mac("DataStart", R.index.min().strftime("%B %Y"))
mac("DataEnd", R.index.max().strftime("%B %Y"))
wt = pd.Series(w, index=L.CRITERIA)
mac("Wmax", wt.max())
mac("Wmin", wt.min())
mac("WmaxName", wt.idxmax().split(" ", 1)[1].lower())
mac("WminName", wt.idxmin().split(" ", 1)[1].lower().replace("max.", "maximum"))
crit = pd.DataFrame({"Type": np.where(L.BENEFIT, "Benefit", "Cost"), "Mean": X.mean().values, "Std. dev.": X.std().values,
                     "Min.": X.min().values, "Max.": X.max().values, "CRITIC weight": w},
                    index=[f"$C_{i + 1}$ " + c.split(" ", 1)[1].replace("%", "\\%") for i, c in enumerate(L.CRITERIA)])
write_table(os.path.join(TAB, "tab_criteria.tex"),
            f"Criteria, cross-sectional descriptive statistics of the window-average monthly values ({MAIN_YEARS[0]}--{MAIN_YEARS[-1]}, "
            f"{n_alt} industries) and CRITIC weights.", "tab:criteria", df_tex(crit, fmt="{:.4f}"))

SC = score_all(E, X, w, mean_ranks(mon))
RK = SC.rank(ascending=False).astype(int)
rc, sp, sm = L.st2hfs_topsis(L.rescale(E), w, **SETTING)
MT = pd.DataFrame({"Industry": [tex_name(a) for a in A], "$\\wp_i^{+}$": sp, "$\\wp_i^{-}$": sm, "$RC_i$": rc,
                   "Rank": RK[PROP].values, "SF-TOPSIS": RK["SF-TOPSIS"].values, "TOPSIS": RK["TOPSIS"].values,
                   "Beta": X[L.CRITERIA[5]].values, "Volatility": X[L.CRITERIA[2]].values}, index=A).sort_values("Rank")
SC.to_csv(os.path.join(RES, "main_scores.csv"))
RK.to_csv(os.path.join(RES, "main_ranks.csv"))
pd.DataFrame([{"criterion": c, "industry": a, "component": comp, "element": l + 1, "primary": E[i, j, k, l, 0],
               "secondary": E[i, j, k, l, 1], "primary_rescaled": L.rescale(E)[i, j, k, l, 0]}
              for j, c in enumerate(L.CRITERIA) for i, a in enumerate(A) for k, comp in enumerate("TIF") for l in range(K)]
             ).to_csv(os.path.join(RES, "decision_matrix_main.csv"), index=False, float_format="%.4f")
sel_rows = pd.concat([MT.head(10), MT.tail(5)])
body = df_tex(sel_rows.drop(columns=[]), fmt="{:.4f}", index=False, colfmt="lrrrrrrrr")
lines = body.split("\n")
i0 = [k for k, l in enumerate(lines) if l.startswith("\\midrule")][0]
lines.insert(i0 + 11, "\\multicolumn{9}{c}{$\\vdots$}\\\\")
write_table(os.path.join(TAB, "tab_main.tex"),
            f"Results of the proposed framework for the {n_alt} industries, {MAIN_YEARS[0]}--{MAIN_YEARS[-1]} "
            "(rescaled grades, $r=d=q=1$, $\\lambda=0.5$): ten best and five worst industries with their similarity ratings, "
            "relative closeness, the ranks assigned by SF-TOPSIS and crisp TOPSIS, and window-average beta and annualised volatility. "
            "The ranks of all methods for all industries are given in \\ref{app:extra}.", "tab:main",
            "\n".join(lines), size="\\scriptsize")
FR = RK.loc[MT.index, METHODS].copy()
FR.columns = [ABBR[m] for m in METHODS]
FR.insert(0, "Industry", [tex_name(a) for a in FR.index])
write_table(os.path.join(TAB, "tab_ranks_full.tex"),
            f"Ranks of the {n_alt} industries, {MAIN_YEARS[0]}--{MAIN_YEARS[-1]}, under the proposed framework (Prop.), its ablations "
            "(A1: no secondary grades; A2: no hesitancy; A3: no rescaling), the six MCDM benchmarks and the four simple baselines "
            "(RA: rank average; RT: TOPSIS on ranks; LV: low volatility; LB: low beta).",
            "tab:ranksfull", df_tex(FR, index=False, colfmt="l" + "r" * len(METHODS)), size="\\scriptsize")
beta, vol = X[L.CRITERIA[5]], X[L.CRITERIA[2]]
top5, bot5 = MT.index[:5].tolist(), MT.index[-5:].tolist()
mac("TopFive", enum([tex_name(a).lower() for a in top5]))
mac("BottomFive", enum([tex_name(a).lower() for a in bot5[::-1]]))
mac("TopFiveBetaLow", beta[top5].min(), "{:.2f}")
mac("TopFiveBetaHigh", beta[top5].max(), "{:.2f}")
mac("BottomFiveBetaLow", beta[bot5].min(), "{:.2f}")
mac("BottomFiveBetaHigh", beta[bot5].max(), "{:.2f}")
mac("MedianBeta", beta.median(), "{:.2f}")
mac("TopFiveBelowMedVol", int((vol[top5] < vol.median()).sum()), "{:d}")
tt = RK["TOPSIS"].sort_values().index[:3].tolist()
mac("TopsisTopThree", enum([tex_name(a).lower() for a in tt]))
mac("TopsisTopThreeRanks", enum([str(int(RK.loc[a, PROP])) for a in tt]))
mac("TopsisTopThreeBetaLow", beta[tt].min(), "{:.2f}")
mac("TopsisTopThreeBetaHigh", beta[tt].max(), "{:.2f}")
mac("TopTenOverlapTopsis", len(set(MT.index[:10]) & set(RK["TOPSIS"].sort_values().index[:10])), "{:d}")
mac("TopTenOverlapSF", len(set(MT.index[:10]) & set(RK["SF-TOPSIS"].sort_values().index[:10])), "{:d}")
mac("RCmax", rc.max(), "{:.4f}")
mac("RCmin", rc.min(), "{:.4f}")

# decision report of the DSS: ranking, reliability flag and linguistic explanation for the main window
CSHORT = ["return", "reward-to-risk", "volatility", "VaR", "drawdown", "beta", "momentum"]
secg = (E[..., 1].mean(axis=(2, 3)) * w).sum(1)
tmean, fmean = E[:, :, 0, :, 0].mean(axis=2), E[:, :, 2, :, 0].mean(axis=2)
rep_rows = []
for a_ in list(MT.index[:5]) + list(MT.index[-5:]):
    i_ = A.index(a_)
    strong = [CSHORT[j] for j in np.argsort(-tmean[i_])[:2] if tmean[i_, j] > 0.4]
    weak = [CSHORT[j] for j in np.argsort(-fmean[i_])[:2] if fmean[i_, j] > 0.4]
    rep_rows.append({"Industry": tex_name(a_), "Rank": int(MT.loc[a_, "Rank"]), "$RC_i$": f"{MT.loc[a_, '$RC_i$']:.4f}",
                     "Rank (RA)": int(RK.loc[a_, "Rank average"]),
                     "Confidence": f"{secg[i_]:.3f} ({'high' if secg[i_] > np.median(secg) else 'low'})",
                     "Mostly good on": ", ".join(strong) if strong else "--", "Mostly bad on": ", ".join(weak) if weak else "--"})
REP = pd.DataFrame(rep_rows)
REP.to_csv(os.path.join(RES, "decision_report_main.csv"), index=False)
body = df_tex(REP, index=False, colfmt="lrrrrll")
lines = body.split("\n")
i0 = [k for k, l in enumerate(lines) if l.startswith("\\midrule")][0]
lines.insert(i0 + 6, "\\midrule")
write_table(os.path.join(TAB, "tab_report.tex"),
            f"Decision report produced by the system for {MAIN_YEARS[0]}--{MAIN_YEARS[-1]} (five best and five worst industries). "
            "Rank (RA): rank under the rank-average baseline. Confidence: CRITIC-weighted mean secondary grade, labelled high or low "
            "relative to the cross-sectional median (Section~\\ref{sec:reliability}). Mostly good / bad on: criteria with the largest "
            "mean truth / falsity grade, shown when that grade exceeds 0.4, i.e.\\ when the industry was in the top / bottom third "
            "of the cross-section in clearly more than one third of the months.", "tab:report", "\n".join(lines), size="\\scriptsize")
mac("RepTopName", tex_name(MT.index[0]).lower())
mac("RepTopStrong", REP.iloc[0]["Mostly good on"])
mac("RepTopWeak", REP.iloc[0]["Mostly bad on"])
mac("RepBotName", tex_name(MT.index[-1]).lower())
mac("RepBotWeak", REP.iloc[-1]["Mostly bad on"])
mac("RepLowConfTop", int(sum("low" in c for c in REP.Confidence[:5])), "{:d}")
mac("RepMaxRankDiff", int(max(abs(r1 - r2) for r1, r2 in zip(REP.Rank, REP["Rank (RA)"]))), "{:d}")

# worked example: best-ranked industry on the beta criterion
ia, jc = A.index(MT.index[0]), 5
cnt = np.zeros((K, 3), int)
for yi, y in enumerate(MAIN_YEARS):
    for m in [m for m in mon if m.year == y]:
        Xm = mon[m].values.copy()
        Xm[:, ~L.BENEFIT] *= -1
        u = pd.DataFrame(Xm).rank(pct=True).values[ia, jc]
        cnt[yi, 0 if u > 2 / 3 else (2 if u <= 1 / 3 else 1)] += 1
prim = (cnt + 1) / (cnt.sum(1, keepdims=True) + 3)
sec = 1 - L.wilson_width(cnt / cnt.sum(1, keepdims=True), cnt.sum(1, keepdims=True))
s_el = (1 + L.EPS) * max(1.0, prim.max(0).sum())
ex = pd.DataFrame({"Year": MAIN_YEARS, "Good": cnt[:, 0], "Neutral": cnt[:, 1], "Bad": cnt[:, 2],
                   "$(t,\\tau)$": [f"({prim[y, 0]:.3f}, {sec[y, 0]:.3f})" for y in range(K)],
                   "$(h,\\eta)$": [f"({prim[y, 1]:.3f}, {sec[y, 1]:.3f})" for y in range(K)],
                   "$(f,\\varphi)$": [f"({prim[y, 2]:.3f}, {sec[y, 2]:.3f})" for y in range(K)],
                   "$\\tilde t$": prim[:, 0] / s_el, "$\\tilde h$": prim[:, 1] / s_el, "$\\tilde f$": prim[:, 2] / s_el})
write_table(os.path.join(TAB, "tab_example.tex"),
            f"Worked example of the construction: {tex_name(MT.index[0]).lower()} on criterion $C_6$ (beta), {MAIN_YEARS[0]}--{MAIN_YEARS[-1]}. "
            "Monthly class counts, (primary, secondary) grades of each yearly scenario by Eqs.~\\eqref{eq:primary}--\\eqref{eq:secondary}, "
            f"and rescaled primary grades by Eq.~\\eqref{{eq:rescale}} with $s_{{ij}}={s_el:.3f}$.", "tab:example",
            df_tex(ex, index=False, colfmt="lrrrcccrrr"), size="\\scriptsize")
mac("ExName", tex_name(MT.index[0]).lower())
mac("ExMaxSum", prim.max(0).sum(), "{:.3f}")
mac("ExScaledSum", (prim.max(0) / s_el).sum(), "{:.3f}")

# agreement between methods (main case) -- heat map
AG = pd.DataFrame([[spearmanr(SC[x], SC[y]).correlation for y in METHODS] for x in METHODS], index=METHODS, columns=METHODS)
AG.to_csv(os.path.join(RES, "agreement.csv"))
for m in METHODS[1:]:
    mac("Agree" + SHORT[m], AG.loc[PROP, m])
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(AG.values, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(METHODS)))
ax.set_yticks(range(len(METHODS)))
ax.set_xticklabels([ABBR[m] for m in METHODS], fontsize=7.5)
ax.set_yticklabels(METHODS, fontsize=7.5)
for i in range(len(METHODS)):
    for j in range(len(METHODS)):
        v = AG.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if abs(v) > 0.6 else DARK)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Spearman's $\\rho$", fontsize=8)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_agreement.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================================ 4. robustness (hold-out windows)
print("robustness ...", flush=True)
rng = np.random.default_rng(SEED)
rows = []
for yrs in HOLDOUT_WINDOWS:
    mon0, a0, E0, X0, w0 = window(yrs)
    base = score_all(E0, X0, w0, mean_ranks(mon0))
    sd = crit_sd(mon0)
    wname = f"{yrs[0]}--{str(yrs[-1])[-2:]}"
    for b in range(B_ROB):
        bm, om, nm = perturb(mon0, yrs, rng, sd)
        for test, pm in (("Bootstrap", bm), ("Outlier", om), ("Noise", nm)):
            sc = score_all(L.build_st2hfe(pm, yrs)[1], L.crisp_matrix(pm, yrs).loc[a0], w0, mean_ranks(pm).loc[a0])
            rows += [dict(window=wname, test=test, rep=b, method=m, tau=kendalltau(base[m], sc[m]).statistic) for m in METHODS]
ROB = pd.DataFrame(rows)
ROB.to_csv(os.path.join(RES, "robustness.csv"), index=False, float_format="%.4f")
RT = ROB.groupby(["test", "method"]).tau.mean().unstack(0).loc[METHODS, ["Bootstrap", "Outlier", "Noise"]]
RT["Overall"] = RT.mean(axis=1)
WB = ROB.groupby(["window", "test", "method"]).tau.mean().unstack("method")[METHODS]
rkW = WB.rank(axis=1, ascending=False)
RT["Avg. rank"] = rkW.mean()
frR = friedmanchisquare(*[WB[m] for m in METHODS])
cdR = nemenyi_cd(len(METHODS), len(WB))
pR = holm({m: safe_wilcoxon(WB[PROP], WB[m]) for m in METHODS[1:]})
RT["Holm $p$"] = [np.nan] + [pR[m] for m in METHODS[1:]]
RT.to_csv(os.path.join(RES, "robustness_summary.csv"))
Tt = RT.copy()
Tt["Holm $p$"] = ["--" if not np.isfinite(v) else ("$<$0.001" if v < 0.001 else f"{v:.3f}") for v in Tt["Holm $p$"]]
Tt["Avg. rank"] = Tt["Avg. rank"].map("{:.2f}".format)
write_table(os.path.join(TAB, "tab_robust.tex"),
            f"Ranking stability on the hold-out windows: mean Kendall's $\\tau$ between unperturbed and perturbed rankings over "
            f"{len(HOLDOUT_WINDOWS)} three-year windows and {B_ROB} replications per window and scheme (month bootstrap; outliers of $\\pm 5$ "
            "standard deviations in 10\\% of the observations; Gaussian noise of 0.25 standard deviations on every observation, which moves "
            "observations across the class boundaries without creating outliers). Average ranks are computed over the "
            f"{len(WB)} window--scheme blocks (Friedman $\\chi^2={frR.statistic:.1f}$, $p{fp(frR.pvalue)}$; Nemenyi CD $={cdR:.2f}$); "
            "Holm $p$: Holm-adjusted Wilcoxon signed-rank test of each method against the proposed framework over the blocks.",
            "tab:robust", df_tex(Tt, colfmt="lrrrrrr"))
mac("RobWindows", len(HOLDOUT_WINDOWS), "{:d}")
mac("RobB", B_ROB, "{:d}")
mac("RobRuns", len(HOLDOUT_WINDOWS) * 3 * B_ROB, "{:,d}")
mac("RobBlocks", len(WB), "{:d}")
mac("RobFriedP", fp(frR.pvalue))
for m in METHODS:
    mac("Rob" + SHORT[m], RT.loc[m, "Overall"])
    mac("RobBoot" + SHORT[m], RT.loc[m, "Bootstrap"])
    mac("RobOut" + SHORT[m], RT.loc[m, "Outlier"])
    mac("RobNoise" + SHORT[m], RT.loc[m, "Noise"])
    if m != PROP and np.isfinite(pR[m]):
        mac("RobP" + SHORT[m], fp(pR[m]))
mac("RobCrispMin", RT.loc[CRISP, "Overall"].min())
mac("RobCrispMax", RT.loc[CRISP, "Overall"].max())
mac("RobOutCrispMin", RT.loc[CRISP, "Outlier"].min())
mac("RobOutCrispMax", RT.loc[CRISP, "Outlier"].max())
mac("RobCrispPMax", fp(max(pR[m] for m in CRISP)))
mac("RobNoiseCrispMin", RT.loc[CRISP, "Noise"].min())
mac("RobNoiseCrispMax", RT.loc[CRISP, "Noise"].max())
mac("RobRankMin", RT.loc[RANKB, "Overall"].min())
mac("RobRankMax", RT.loc[RANKB, "Overall"].max())
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
for ax, t in zip(axes, ["Bootstrap", "Outlier", "Noise"]):
    D = WB.xs(t, level="test")
    xs = range(len(D))
    for m in METHODS[::-1]:
        if m in SORTB:
            continue
        isP, isC, isR = m == PROP, m in CRISP, m in RANKB
        ax.plot(xs, D[m].values, color=ACCENT if isP else ("#c2571a" if isR else (DARK if isC else GREY)), lw=2 if isP else (1.3 if isR else 0.9),
                ls="--" if isC else ("-." if isR else "-"), marker="o" if isP else None, ms=3.5, zorder=3 if isP else (2 if isR else 1),
                alpha=1 if isP else 0.75)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(D.index, rotation=45, fontsize=7)
    ax.set_title({"Bootstrap": "(a) Month bootstrap", "Outlier": "(b) Outliers", "Noise": "(c) Boundary noise"}[t], fontsize=9)
    ax.set_xlabel("Evaluation window")
    ax.grid(axis="y", color=LIGHT, lw=0.6)
axes[0].set_ylabel("Mean Kendall's $\\tau$")
axes[1].plot([], [], color=ACCENT, lw=2, marker="o", ms=3.5, label="Proposed")
axes[1].plot([], [], color=GREY, lw=0.9, label="Ablations, SF- and HF-TOPSIS")
axes[1].plot([], [], color="#c2571a", lw=1.3, ls="-.", label="Rank-based baselines")
axes[1].plot([], [], color=DARK, lw=0.9, ls="--", label="Crisp methods (raw values)")
axes[1].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_robust.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================================ 5. rank reversal (hold-out windows)
print("rank reversal ...", flush=True)


def reversed_pairs(full, reduced):
    s1 = np.sign(full[:, None] - full[None, :])
    s2 = np.sign(reduced[:, None] - reduced[None, :])
    return int((np.triu(s1 != s2, 1)).sum())


rr = []
for yrs in HOLDOUT_WINDOWS:
    mon0, a0, E0, X0, w0 = window(yrs)
    Q0 = mean_ranks(mon0)
    base = score_all(E0, X0, w0, Q0)
    for k in range(len(a0)):
        keep = [i for i in range(len(a0)) if i != k]
        agg = score_all(E0[keep], X0.iloc[keep], w0, Q0.iloc[keep])
        mk = {m: df.drop(index=a0[k]) for m, df in mon0.items()}
        _, Ek = L.build_st2hfe(mk, yrs)
        Xk = L.crisp_matrix(mk, yrs)
        full = score_all(Ek, Xk, L.critic(Xk.values), mean_ranks(mk))
        for m in METHODS:
            rr.append(dict(window=yrs[0], removed=a0[k], method=m, agg=reversed_pairs(base[m].values[keep], agg[m].values),
                           full=reversed_pairs(base[m].values[keep], full[m].values), pairs=len(keep) * (len(keep) - 1) // 2))
RR = pd.DataFrame(rr)
RR.to_csv(os.path.join(RES, "rank_reversal.csv"), index=False)
g = RR.groupby("method")
RRt = pd.DataFrame({"Runs with reversal, \\% (agg.)": 100 * g["agg"].apply(lambda v: (v > 0).mean()),
                    "Reversed pairs, \\% (agg.)": 100 * g["agg"].sum() / g["pairs"].sum(),
                    "Runs with reversal, \\% (full)": 100 * g["full"].apply(lambda v: (v > 0).mean()),
                    "Reversed pairs, \\% (full)": 100 * g["full"].sum() / g["pairs"].sum()}).loc[METHODS]
RRt.to_csv(os.path.join(RES, "rank_reversal_summary.csv"))
n_runs = len(RR) // len(METHODS)
write_table(os.path.join(TAB, "tab_rr.tex"),
            f"Leave-one-out rank-reversal test on the {len(HOLDOUT_WINDOWS)} hold-out windows ({n_runs} removals, "
            f"{int(RR.pairs.sum() / len(METHODS)):,} pairs of remaining alternatives). agg.: grades and weights fixed; "
            "full: grades and CRITIC weights recomputed for the reduced set.", "tab:rr", df_tex(RRt, fmt="{:.2f}"))
mac("RRRuns", n_runs, "{:d}")
mac("RRPropAgg", RRt.loc[PROP].iloc[1], "{:.2f}")
mac("RRPropFull", RRt.loc[PROP].iloc[3], "{:.2f}")
mac("RRAggMax", RRt.loc[[PROP] + FUZZY + CRISP].iloc[:, 1].max(), "{:.2f}")
mac("RRFuzzyFullMin", RRt.loc[FUZZY].iloc[:, 3].min(), "{:.2f}")
mac("RRFuzzyFullMax", RRt.loc[FUZZY].iloc[:, 3].max(), "{:.2f}")
mac("RRCrispFullMin", RRt.loc[CRISP].iloc[:, 3].min(), "{:.2f}")
mac("RRCrispFullMax", RRt.loc[CRISP].iloc[:, 3].max(), "{:.2f}")
mac("RRAThreeFull", RRt.loc[METHODS[3]].iloc[3], "{:.2f}")
mac("RRRAFull", RRt.loc["Rank average"].iloc[3], "{:.2f}")
mac("RRRTFull", RRt.loc["TOPSIS on ranks"].iloc[3], "{:.2f}")

# ============================================================================ 6. out-of-sample evaluation
print("out-of-sample ...", flush=True)
bt, SCORES, SECG, OUTC = [], {}, {}, {}
for ty in range(FIRST_DECISION, LAST + 1):
    mon_, a_, E_, X_, w_ = window(list(range(ty - K, ty)))
    sc = score_all(E_, X_, w_, mean_ranks(mon_))
    SCORES[ty] = sc
    SECG[ty] = pd.Series((E_[..., 1].mean(axis=(2, 3)) * w_).sum(1), index=a_)     # weighted mean secondary grade
    OUTC[ty] = realised(ty, a_, w_)["attain"]
    out = realised(ty, a_, w_)
    for m in METHODS:
        bt.append(dict(year=ty, method=m, attain=spearmanr(sc[m], out["attain"]).correlation,
                       sharpe=spearmanr(sc[m], out["sharpe"]).correlation, ret=spearmanr(sc[m], out["ret"]).correlation,
                       lowvol=spearmanr(sc[m], out["lowvol"]).correlation))
BT = pd.DataFrame(bt)
BT.to_csv(os.path.join(RES, "oos_correlations.csv"), index=False, float_format="%.4f")
HO = BT[BT.year > DESIGN_END]
PmU = HO.pivot(index="year", columns="method", values="attain")[METHODS]
PmS = HO.pivot(index="year", columns="method", values="sharpe")[METHODS]
PmR = HO.pivot(index="year", columns="method", values="ret")[METHODS]
PmV = HO.pivot(index="year", columns="method", values="lowvol")[METHODS]
frU, frS = friedmanchisquare(*[PmU[m] for m in METHODS]), friedmanchisquare(*[PmS[m] for m in METHODS])
cdB = nemenyi_cd(len(METHODS), len(PmU))
# pre-specified primary test: Holm-adjusted moving-block bootstrap of the paired annual differences
BBU = {m: block_boot(PmU[PROP] - PmU[m]) for m in METHODS[1:]}
BBS = {m: block_boot(PmS[PROP] - PmS[m]) for m in METHODS[1:]}
pU = holm({m: BBU[m][1] for m in METHODS[1:]})
pS = holm({m: BBS[m][1] for m in METHODS[1:]})
pUw = holm({m: safe_wilcoxon(PmU[PROP], PmU[m]) for m in METHODS[1:]})      # secondary (assumes independence)
# equivalence (TOST via the 90% block-bootstrap interval) against the rank-based baselines, margin 0.05
EQ_MARGIN = 0.05
for m in RANKB:
    d_, p_, ci95, ci90 = BBU[m]
    mac("EqDiff" + SHORT[m], d_)
    mac("EqLo" + SHORT[m], ci90[0])
    mac("EqHi" + SHORT[m], ci90[1])
    mac("EqIs" + SHORT[m], "within" if (ci90[0] > -EQ_MARGIN and ci90[1] < EQ_MARGIN) else "not within", "{}")
for m in METHODS[1:]:
    mac("BBUDiff" + SHORT[m], BBU[m][0])
    mac("BBULo" + SHORT[m], BBU[m][2][0])
    mac("BBUHi" + SHORT[m], BBU[m][2][1])
    mac("OOSUPW" + SHORT[m], fp(pUw[m]))
mac("EqMargin", EQ_MARGIN, "{:.2f}")
FULL = BT.groupby("method")[["attain", "sharpe"]].mean().loc[METHODS]
OT = pd.DataFrame({"$\\rho_U$": PmU.mean(), "Avg. rank": PmU.rank(axis=1, ascending=False).mean(),
                   "Holm $p$": [np.nan] + [pU[m] for m in METHODS[1:]],
                   "$\\rho_S$": PmS.mean(), "Avg. rank ": PmS.rank(axis=1, ascending=False).mean(),
                   "Holm $p$ ": [np.nan] + [pS[m] for m in METHODS[1:]],
                   "$\\rho_R$": PmR.mean(), "$\\rho_V$": PmV.mean(),
                   "$\\rho_U$, all": FULL.attain, "$\\rho_S$, all": FULL.sharpe}).loc[METHODS]
OT.to_csv(os.path.join(RES, "oos_summary.csv"))
Tt = OT.copy()
for c in ["Holm $p$", "Holm $p$ "]:
    Tt[c] = ["--" if not np.isfinite(v) else ("$<$0.001" if v < 0.001 else f"{v:.3f}") for v in Tt[c]]
for c in ["Avg. rank", "Avg. rank "]:
    Tt[c] = Tt[c].map("{:.2f}".format)
body = df_tex(Tt, colfmt="lrrrrrrrrrr")
body = body.replace("\\toprule", "\\toprule\n & \\multicolumn{3}{c}{Criteria attainment} & \\multicolumn{3}{c}{Sharpe ratio} & Return & "
                    "Low vol. & \\multicolumn{2}{c}{All decisions}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){10-11}", 1)
write_table(os.path.join(TAB, "tab_oos.tex"),
            f"Out-of-sample evaluation on the hold-out period ({HOLDOUT_YEARS[0]}--{HOLDOUT_YEARS[-1]}, {len(HOLDOUT_YEARS)} annual decisions; "
            "year $t$ is decided with the data of years $t-3$ to $t-1$). Mean Spearman correlation between the decision scores and "
            "next-year criteria attainment ($\\rho_U$), Sharpe ratio ($\\rho_S$), return ($\\rho_R$) and negative volatility ($\\rho_V$); "
            f"average ranks over the decisions (Friedman: $\\chi^2={frU.statistic:.1f}$, $p{fp(frU.pvalue)}$ for $\\rho_U$; "
            f"$\\chi^2={frS.statistic:.1f}$, $p{fp(frS.pvalue)}$ for $\\rho_S$; Nemenyi CD $={cdB:.2f}$); Holm $p$: Holm-adjusted moving-block "
            f"bootstrap test (blocks of three years) of the mean paired difference to the proposed framework. All decisions: means over {FIRST_DECISION}--{LAST}, including the design period.",
            "tab:oos", body, size="\\scriptsize")
cd_plot(PmU.rank(axis=1, ascending=False).mean(), cdB, os.path.join(FIG, "fig_cd_attain.pdf"))
mac("OOSn", len(HOLDOUT_YEARS), "{:d}")
mac("OOSYears", f"{HOLDOUT_YEARS[0]}--{HOLDOUT_YEARS[-1]}")
mac("AllN", LAST - FIRST_DECISION + 1, "{:d}")
mac("AllYears", f"{FIRST_DECISION}--{LAST}")
mac("OOSFriedUChi", frU.statistic, "{:.1f}")
mac("OOSFriedUP", fp(frU.pvalue))
mac("OOSFriedSChi", frS.statistic, "{:.1f}")
mac("OOSFriedSP", fp(frS.pvalue))
mac("OOSCD", cdB, "{:.2f}")
for m in METHODS:
    mac("OOSU" + SHORT[m], PmU[m].mean())
    mac("OOSS" + SHORT[m], PmS[m].mean())
    mac("OOSR" + SHORT[m], PmR[m].mean())
    mac("OOSV" + SHORT[m], PmV[m].mean())
    mac("OOSURank" + SHORT[m], OT.loc[m, "Avg. rank"], "{:.2f}")
    if m != PROP:
        mac("OOSUP" + SHORT[m], fp(pU[m]))
        mac("OOSSP" + SHORT[m], fp(pS[m]))
mac("OOSUPMax", fp(max(pU[m] for m in METHODS[1:])))
mac("OOSSPMin", fp(min(pS[m] for m in METHODS[1:])))
mac("OOSUPropTP", fp(ttest_1samp(PmU[PROP], 0).pvalue))
mac("OOSSPropTP", fp(ttest_1samp(PmS[PROP], 0).pvalue))
mac("OOSUWins", int((PmU.rank(axis=1, ascending=False)[PROP] == 1).sum()), "{:d}")
mac("OOSUPropBeatsSF", int((PmU[PROP] > PmU["SF-TOPSIS"]).sum()), "{:d}")
mac("OOSUPropBeatsAOne", int((PmU[PROP] > PmU[METHODS[1]]).sum()), "{:d}")
mac("OOSUPropBeatsATwo", int((PmU[PROP] > PmU[METHODS[2]]).sum()), "{:d}")
mac("OOSUCrispMin", PmU[CRISP].mean().min())
mac("OOSUCrispMax", PmU[CRISP].mean().max())
mac("OOSSMin", PmS.mean().min())
mac("OOSSMax", PmS.mean().max())
mac("OOSRMin", PmR.mean().min())
mac("OOSRMax", PmR.mean().max())
mac("AllUProp", FULL.loc[PROP, "attain"])
mac("AllSProp", FULL.loc[PROP, "sharpe"])
DSG = BT[BT.year <= DESIGN_END].pivot(index="year", columns="method", values="attain")[METHODS]
mac("DesUProp", DSG[PROP].mean())
mac("DesUSF", DSG["SF-TOPSIS"].mean())
fig, ax = plt.subplots(figsize=(7.2, 3.5))
for i, m in enumerate(METHODS):
    v = PmU[m].values
    mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), s=8,
               color=ACCENT if m == PROP else GREY, alpha=0.4, lw=0)
    ax.errorbar(mu, i, xerr=1.96 * se, fmt="o", color=ACCENT if m == PROP else DARK, ms=5, capsize=2.5, lw=1.1)
ax.axvline(0, color="#b5b4aa", lw=0.8)
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels(METHODS, fontsize=8)
ax.set_ylim(len(METHODS) - 0.5, -0.5)
ax.set_xlabel("Spearman correlation between decision score and next-year criteria attainment")
ax.grid(axis="x", color=LIGHT, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_oos.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------- portfolios
print("portfolios ...", flush=True)


def newey_west(y, Xf, lags=5):
    Xc = np.column_stack([np.ones(len(y)), Xf])
    b = np.linalg.lstsq(Xc, y, rcond=None)[0]
    e = y - Xc @ b
    XtXi = np.linalg.inv(Xc.T @ Xc)
    S = (Xc * e[:, None]).T @ (Xc * e[:, None])
    for l in range(1, lags + 1):
        G = (Xc[l:] * e[l:, None]).T @ (Xc[:-l] * e[:-l, None])
        S += (1 - l / (lags + 1)) * (G + G.T)
    return b, b / np.sqrt(np.diag(XtXi @ S @ XtXi))


def portfolio(method, years, k=TOPK):
    parts, turn, prev = [], [], None
    for ty in years:
        sc = SCORES[ty]
        r = R.loc[str(ty), sc.index]
        if method == "1/N":
            parts.append(r.mean(axis=1))
        elif method == "Market":
            parts.append(MKT.loc[str(ty)])
        else:
            top = sc[method].sort_values(ascending=False).index[:k]
            parts.append(r[top].mean(axis=1))
            if prev is not None:
                turn.append(1 - len(set(top) & set(prev)) / k)
            prev = top
    return pd.concat(parts), (np.mean(turn) if turn else 0.0)


def perf(pr):
    ex = pr - F.rf.loc[pr.index]
    f = F.loc[pr.index]
    wealth = (1 + pr).cumprod()
    b1, t1 = newey_west(ex.values, f[["mktrf"]].values)
    b5, t5 = newey_west(ex.values, f[["mktrf", "smb", "hml", "rmw", "cma"]].values)
    return dict(exret=ex.mean() * 252, vol=pr.std() * np.sqrt(252), sharpe=ex.mean() / pr.std() * np.sqrt(252),
                mdd=(wealth / wealth.cummax() - 1).min(), beta=b1[1], a1=b1[0] * 252, t1=t1[0], a5=b5[0] * 252, t5=t5[0])


def sharpe_diff_p(pa, pb, block=21, B=5000, seed=SEED):
    """Two-sided circular block-bootstrap p-value for the difference of two Sharpe ratios (paired daily returns)."""
    rf = F.rf.loc[pa.index].values
    a, b = pa.values - rf, pb.values - rf
    n = len(a)
    sh = lambda x: x.mean(axis=-1) / x.std(axis=-1)
    d0 = sh(a) - sh(b)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    d = np.empty(B)
    for i in range(0, B, 250):
        starts = rng.integers(0, n, size=(250, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(250, -1)[:, :n]
        d[i:i + 250] = sh(a[idx]) - sh(b[idx])
    return d0 * np.sqrt(252), 2 * min((d <= 0).mean(), (d >= 0).mean())


PF, SER = {}, {}
for m in METHODS + ["1/N", "Market"]:
    pr, to = portfolio(m, HOLDOUT_YEARS)
    SER[m] = pr
    PF[m] = dict(perf(pr), turnover=to)
for m in METHODS:
    PF[m]["dS"], PF[m]["pS"] = sharpe_diff_p(SER[m], SER["1/N"])
PFT = pd.DataFrame(PF).T
PFT.to_csv(os.path.join(RES, "portfolios.csv"))
rows_ = []
for m in METHODS + ["1/N", "Market"]:
    s = PFT.loc[m]
    isb = m in ("1/N", "Market")
    rows_.append({"Portfolio": (f"Top {TOPK}: " + m) if not isb else ("Equal weight ($1/N$)" if m == "1/N" else "Market"),
                  "Excess return": f"{s.exret:.3f}", "Volatility": f"{s.vol:.3f}", "Sharpe": f"{s.sharpe:.3f}",
                  "$p$ vs.\\ $1/N$": "--" if isb else ("$<$0.001" if s.pS < 0.001 else f"{s.pS:.3f}"),
                  "Max.\\ drawdown": f"{s.mdd:.3f}", "Beta": f"{s.beta:.2f}",
                  "CAPM $\\alpha$": "--" if m == "Market" else f"{s.a1:.3f} ({s.t1:.2f})",
                  "FF5 $\\alpha$": "--" if m == "Market" else f"{s.a5:.3f} ({s.t5:.2f})",
                  "Turnover": "--" if isb else f"{s.turnover:.2f}"})
write_table(os.path.join(TAB, "tab_portfolio.tex"),
            f"Portfolios formed from the rankings on the hold-out period ({HOLDOUT_YEARS[0]}--{HOLDOUT_YEARS[-1]}): equally weighted portfolio of the "
            f"{TOPK} best-ranked industries, re-formed every January, compared with the equal-weight portfolio of all industries and the market. "
            "Annualised excess return, volatility and Sharpe ratio; maximum drawdown over the whole period; market beta; annualised CAPM and "
            "Fama--French five-factor alphas with Newey--West $t$-statistics in parentheses; average annual turnover (share of constituents replaced). "
            "$p$ vs.\\ $1/N$: circular block-bootstrap test of the Sharpe-ratio difference to the equal-weight portfolio.",
            "tab:portfolio", df_tex(pd.DataFrame(rows_), index=False, colfmt="lrrrrrrrrr"), size="\\scriptsize")
for m in [PROP, "SF-TOPSIS", "TOPSIS", "EDAS", "CoCoSo", "1/N", "Market"] + BASELINES:
    key = {"1/N": "EW", "Market": "Mkt"}.get(m, SHORT.get(m))
    s = PFT.loc[m]
    mac("PFRet" + key, 100 * s.exret, "{:.1f}")
    mac("PFVol" + key, 100 * s.vol, "{:.1f}")
    mac("PFSharpe" + key, s.sharpe, "{:.2f}")
    mac("PFMdd" + key, -100 * s.mdd, "{:.0f}")
    mac("PFBeta" + key, s.beta, "{:.2f}")
    if m not in ("1/N", "Market"):
        mac("PFTurn" + key, 100 * s.turnover, "{:.0f}")
mac("PFSharpePProp", fp(PFT.loc[PROP, "pS"]))
for m in BASELINES + ["SF-TOPSIS"]:
    mac("PFSharpePvsProp" + SHORT[m], fp(sharpe_diff_p(SER[PROP], SER[m])[1]))
    mac("PFAlphaFF" + SHORT[m], 100 * PFT.loc[m, "a5"], "{:.1f}")
    mac("PFAlphaFFT" + SHORT[m], PFT.loc[m, "t5"], "{:.2f}")
mac("PFOverlapRA", np.mean([len(set(SCORES[t][PROP].nlargest(TOPK).index) & set(SCORES[t]["Rank average"].nlargest(TOPK).index))
                            for t in HOLDOUT_YEARS]), "{:.1f}")
mac("PFOverlapLV", np.mean([len(set(SCORES[t][PROP].nlargest(TOPK).index) & set(SCORES[t]["Low volatility"].nlargest(TOPK).index))
                            for t in HOLDOUT_YEARS]), "{:.1f}")
mac("PFAlphaCapmProp", 100 * PFT.loc[PROP, "a1"], "{:.1f}")
mac("PFAlphaCapmTProp", PFT.loc[PROP, "t1"], "{:.2f}")
mac("PFAlphaFFProp", 100 * PFT.loc[PROP, "a5"], "{:.1f}")
mac("PFAlphaFFTProp", PFT.loc[PROP, "t5"], "{:.2f}")
mac("PFSharpeRankProp", int(PFT.loc[METHODS, "sharpe"].rank(ascending=False)[PROP]), "{:d}")
mac("PFTurnCrispMin", 100 * PFT.loc[CRISP, "turnover"].min(), "{:.0f}")
mac("PFTurnCrispMax", 100 * PFT.loc[CRISP, "turnover"].max(), "{:.0f}")
mac("PFSharpeCrispMin", PFT.loc[CRISP, "sharpe"].min(), "{:.2f}")
mac("PFSharpeCrispMax", PFT.loc[CRISP, "sharpe"].max(), "{:.2f}")
mac("PFCostDrag", 100 * 2 * PFT.loc[PROP, "turnover"] * 0.005, "{:.2f}")
mac("TopK", TOPK, "{:d}")
for k_ in (5, 15):
    s = perf(portfolio(PROP, HOLDOUT_YEARS, k_)[0])
    mac("PFSharpeK" + {5: "Five", 15: "Fifteen"}[k_], s["sharpe"], "{:.2f}")
fig, ax = plt.subplots(figsize=(7.2, 3.3))
styles = {PROP: (ACCENT, 2.0, "-"), "TOPSIS": (DARK, 1.0, "--"), "1/N": (GREY, 1.3, "-"), "Market": (GREY, 1.0, ":")}
labels = {PROP: f"Top {TOPK}: proposed", "TOPSIS": f"Top {TOPK}: crisp TOPSIS", "1/N": "Equal weight (1/N)", "Market": "Market"}
for m, (c, lw, ls) in styles.items():
    wlt = (1 + SER[m]).cumprod()
    ax.plot(wlt.index, wlt.values, color=c, lw=lw, ls=ls, label=labels[m], zorder=3 if m == PROP else 2)
ax.set_yscale("log")
ax.set_ylabel("Value of 1 invested (log scale)")
ax.grid(axis="y", color=LIGHT, lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_wealth.pdf"), bbox_inches="tight")
plt.close(fig)

# ============================================================================ 7. parameter sensitivity (hold-out, post hoc)
print("sensitivity ...", flush=True)
PGRID = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 2, 1), (2, 1, 1), (1, 1, 2), (6, 6, 6), (9, 7, 8)]
hold = {ty: window(list(range(ty - K, ty))) for ty in HOLDOUT_YEARS}
outs = {ty: realised(ty, hold[ty][1], hold[ty][4]) for ty in HOLDOUT_YEARS}
sens = []
for rdq in PGRID:
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        agree, ru = [], []
        for ty in HOLDOUT_YEARS:
            En, w_ = L.rescale(hold[ty][2]), hold[ty][4]
            s_ = L.st2hfs_topsis(En, w_, *rdq, lam)[0]
            agree.append(spearmanr(s_, SCORES[ty][PROP]).correlation)
            ru.append(spearmanr(s_, outs[ty]["attain"]).correlation)
        sens.append(dict(rdq=str(rdq).replace(" ", ""), lam=lam, agree=np.mean(agree), attain=np.mean(ru)))
SE = pd.DataFrame(sens)
SE.to_csv(os.path.join(RES, "sensitivity_parameters.csv"), index=False)
order = [str(k).replace(" ", "") for k in PGRID]
Sa = SE.pivot(index="rdq", columns="lam", values="agree").loc[order]
Su = SE.pivot(index="rdq", columns="lam", values="attain").loc[order]
T = pd.concat([Sa, Su], axis=1)
T.columns = [f"{c:.2f}" for c in Sa.columns] * 2
T.index = ["$" + i + "$" for i in T.index]
body = df_tex(T, colfmt="l" + "r" * 10)
body = body.replace("\\toprule", "\\toprule\n & \\multicolumn{5}{c}{Agreement with the selected setting ($\\rho$)} & "
                    "\\multicolumn{5}{c}{$\\rho_U$}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n$(r,d,q)$ $\\backslash$ $\\lambda$", 1)
write_table(os.path.join(TAB, "tab_sens.tex"),
            f"Sensitivity to the exponents $r,d,q$ and the parameter $\\lambda$ on the hold-out period (rescaled grades; means over "
            f"{len(HOLDOUT_YEARS)} annual decisions): Spearman correlation of the ranking with the ranking of the selected setting "
            "$(1,1,1)$, $\\lambda=0.5$, and correlation of the scores with next-year criteria attainment ($\\rho_U$).",
            "tab:sens", body, size="\\scriptsize")
mac("SensLamAgreeMin", Sa.loc["(1,1,1)"].min())
mac("SensLamUMin", Su.loc["(1,1,1)"].min())
mac("SensLamUMax", Su.loc["(1,1,1)"].max())
mac("SensTwoAgree", Sa.loc["(2,2,2)", 0.5])
mac("SensTwoU", Su.loc["(2,2,2)", 0.5])
mac("SensThreeAgree", Sa.loc["(3,3,3)", 0.5])
mac("SensThreeU", Su.loc["(3,3,3)", 0.5])
mac("SensSixAgree", Sa.loc["(6,6,6)", 0.5])
mac("SensSixU", Su.loc["(6,6,6)", 0.5])
mac("SensNineAgree", Sa.loc["(9,7,8)", 0.5])
mac("SensNineU", Su.loc["(9,7,8)", 0.5])
mac("SensMixedAgreeMin", Sa.loc[["(1,2,3)", "(3,2,1)", "(2,1,1)", "(1,1,2)"], 0.5].min())
mac("SensMixedAgreeMax", Sa.loc[["(1,2,3)", "(3,2,1)", "(2,1,1)", "(1,1,2)"], 0.5].max())

# ---------------------------------------------------------------------------- thresholds and window length
print("thresholds ...", flush=True)
keep = [PROP, "SF-TOPSIS", "HF-TOPSIS", "TOPSIS", "Rank average", "TOPSIS on ranks"]
rows = []
for th, thname in [(1 / 3, "Terciles"), (1 / 4, "Quartiles"), (1 / 5, "Quintiles")]:
    for K_ in (3, 5):
        btk = []
        for ty in HOLDOUT_YEARS:
            yrs = list(range(ty - K_, ty))
            mon_, a_, E_, X_, w_ = window(yrs, th)
            sc = score_all(E_, X_, w_, mean_ranks(mon_))
            out = realised(ty, a_, w_)
            btk += [dict(year=ty, method=m, rho=spearmanr(sc[m], out["attain"]).correlation) for m in keep]
        Bk = pd.DataFrame(btk).pivot(index="year", columns="method", values="rho")
        st = {m: [] for m in keep}
        rng = np.random.default_rng(SEED + 1)
        for y0 in (HOLDOUT_WINDOWS[1][0], HOLDOUT_WINDOWS[-3][0]):
            yrs = list(range(y0, y0 + K_))
            mon0, a0, E0, X0, w0 = window(yrs, th)
            base = score_all(E0, X0, w0, mean_ranks(mon0))
            sd = crit_sd(mon0)
            for _ in range(40):
                _, om, _ = perturb(mon0, yrs, rng, sd)
                sc = score_all(L.build_st2hfe(om, yrs, th)[1], L.crisp_matrix(om, yrs).loc[a0], w0, mean_ranks(om).loc[a0])
                for m in keep:
                    st[m].append(kendalltau(base[m], sc[m]).statistic)
        for m in keep:
            rows.append(dict(Classes=thname, K=K_, Method=m, rho=Bk[m].mean(), tau=np.mean(st[m]),
                             p=np.nan if m == PROP else safe_wilcoxon(Bk[PROP], Bk[m])))
TH = pd.DataFrame(rows)
TH.to_csv(os.path.join(RES, "sensitivity_thresholds.csv"), index=False)
order = [(c, k) for c in ("Terciles", "Quartiles", "Quintiles") for k in (3, 5)]
Tr = TH.pivot_table(index=["Classes", "K"], columns="Method", values="rho")[keep].loc[order]
Tu = TH.pivot_table(index=["Classes", "K"], columns="Method", values="tau")[keep].loc[order]


def bold(row):
    return [f"\\textbf{{{v:.3f}}}" if abs(v - row.max()) < 1e-12 else f"{v:.3f}" for v in row]


lines = [f"{c} & {k} & " + " & ".join(bold(r)) + " & " + " & ".join(bold(Tu.loc[(c, k)])) + " \\\\" for (c, k), r in Tr.iterrows()]
hdr = " & ".join(ABBR[m] for m in keep)
body = ("\\begin{tabular}{llrrrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{6}{c}{$\\rho_U$} & \\multicolumn{6}{c}{Outlier stability "
        "(Kendall's $\\tau$)}\\\\\n\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}\nClasses & $K$ & " + hdr + " & " + hdr + " \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
write_table(os.path.join(TAB, "tab_thresholds.tex"),
            f"Sensitivity to the classification thresholds and the number of yearly scenarios $K$ on the hold-out period ({len(HOLDOUT_YEARS)} "
            "annual decisions; outlier stability over two windows with 40 replications each). Best value per row and measure in bold.",
            "tab:thresholds", body, size="\\scriptsize")
for (c, k) in order:
    key = c[:-1].replace("Tercile", "Terc").replace("Quartile", "Quart").replace("Quintile", "Quint") + {3: "Three", 5: "Five"}[k]
    mac("ThU" + key, Tr.loc[(c, k), PROP])
    mac("ThTau" + key, Tu.loc[(c, k), PROP])
mac("ThUSFQuintThree", Tr.loc[("Quintiles", 3), "SF-TOPSIS"])
mac("ThBestU", int(sum(Tr.loc[o].idxmax() == PROP for o in order)), "{:d}")
mac("ThPKFiveSF", fp(TH[(TH.Classes == "Terciles") & (TH.K == 5) & (TH.Method == "SF-TOPSIS")].p.iloc[0]))

# ============================================================================ 8. mean-variance benchmarks and tail risk
print("mean-variance benchmarks ...", flush=True)
from scipy.optimize import minimize


def mv_weights(year, kind):
    """Long-only, fully invested Markowitz portfolios estimated on the decision window of `year`:
    'minvar' = minimum variance, 'maxsharpe' = maximum Sharpe ratio (tangency portfolio)."""
    a = hold[year][1]
    r = R.loc[str(year - K):str(year - 1), a]
    S = r.cov().values * 252
    mu = (r.mean().values - F.rf.loc[r.index].mean()) * 252
    n = len(a)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    if kind == "minvar":
        f = lambda x: x @ S @ x
    else:
        f = lambda x: -(x @ mu) / np.sqrt(x @ S @ x)
    best = None
    for x0 in (np.full(n, 1 / n), np.eye(n)[int(np.argmax(mu / np.sqrt(np.diag(S))))] * 0.9 + 0.1 / n):
        res = minimize(f, x0, method="SLSQP", bounds=[(0, 1)] * n, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    x = np.clip(best.x, 0, None)
    return pd.Series(x / x.sum(), index=a)


def mv_portfolio(kind, years):
    parts, turn, prev = [], [], None
    for ty in years:
        wv = mv_weights(ty, kind)
        parts.append(R.loc[str(ty), wv.index] @ wv)
        if prev is not None:
            turn.append(0.5 * (wv - prev.reindex(wv.index).fillna(0)).abs().sum())
        prev = wv
    return pd.concat(parts), float(np.mean(turn))


def expectile(x, alpha):
    e = x.mean()
    for _ in range(200):
        wgt = np.where(x <= e, 1 - alpha, alpha)
        e_new = (wgt * x).sum() / wgt.sum()
        if abs(e_new - e) < 1e-12:
            break
        e = e_new
    return e


def tail(pr, alpha=0.05):
    """Daily historical-simulation risk measures in per cent: VaR, expected shortfall, expectile VaR."""
    x = pr.values
    q = np.quantile(x, alpha)
    return dict(var=-100 * q, es=-100 * x[x <= q].mean(), evar=-100 * expectile(x, alpha))


MVNAME = {"minvar": "Minimum variance", "maxsharpe": "Maximum Sharpe ratio"}
for kind in MVNAME:
    SER[MVNAME[kind]], to_ = mv_portfolio(kind, HOLDOUT_YEARS)
    PF[MVNAME[kind]] = dict(perf(SER[MVNAME[kind]]), turnover=to_)
# two-stage variant: screening by the framework, then minimum-variance weights within the selected industries
parts_ = []
for ty in HOLDOUT_YEARS:
    top_ = SCORES[ty][PROP].sort_values(ascending=False).index[:TOPK]
    S_ = R.loc[str(ty - K):str(ty - 1), top_].cov().values * 252
    res_ = minimize(lambda x: x @ S_ @ x, np.full(TOPK, 1 / TOPK), method="SLSQP", bounds=[(0, 1)] * TOPK,
                    constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    x_ = np.clip(res_.x, 0, None)
    parts_.append(R.loc[str(ty), top_] @ (x_ / x_.sum()))
TWO = "Two-stage"
SER[TWO] = pd.concat(parts_)
PF[TWO] = dict(perf(SER[TWO]), turnover=np.nan)
RISKROWS = [PROP, TWO, "SF-TOPSIS", "TOPSIS", "1/N", "Minimum variance", "Maximum Sharpe ratio", "Market"]
rk = {}
for m in RISKROWS:
    rk[m] = dict(mean=100 * PF[m]["exret"], std=100 * PF[m]["vol"], sharpe=PF[m]["sharpe"], mdd=-100 * PF[m]["mdd"], **tail(SER[m]))
RK_ = pd.DataFrame(rk).T
for b_ in ("1/N", "Minimum variance", "Maximum Sharpe ratio"):
    d0, p0 = sharpe_diff_p(SER[PROP], SER[b_])
    RK_.loc[b_, "p_sharpe"] = p0
RK_.to_csv(os.path.join(RES, "portfolio_risk.csv"))
lab = {PROP: f"Top {TOPK}: Proposed", TWO: f"Top {TOPK}: Proposed, MV weights", "SF-TOPSIS": f"Top {TOPK}: SF-TOPSIS", "TOPSIS": f"Top {TOPK}: TOPSIS",
       "1/N": "Equal weight ($1/N$)", "Minimum variance": "Minimum variance (MV)", "Maximum Sharpe ratio": "Maximum Sharpe ratio (MV)",
       "Market": "Market"}
rows_ = []
for m in RISKROWS:
    s = RK_.loc[m]
    rows_.append({"Portfolio": lab[m], "Mean": f"{s['mean']:.1f}", "StD": f"{s['std']:.1f}", "Sharpe": f"{s['sharpe']:.2f}",
                  "VaR": f"{s['var']:.2f}", "ES": f"{s['es']:.2f}", "EVaR": f"{s['evar']:.2f}", "Max.\\ drawdown": f"{s['mdd']:.0f}"})
body = df_tex(pd.DataFrame(rows_), index=False, colfmt="lrrrrrrr")
rel = []
for b_ in ("1/N", "Minimum variance", "Maximum Sharpe ratio", "Market"):
    rel.append("vs.\\ " + {"1/N": "equal weight ($1/N$)", "Minimum variance": "minimum variance (MV)", "Maximum Sharpe ratio": "maximum Sharpe ratio (MV)", "Market": "market"}[b_] + " & "
               + " & ".join(f"{100 * (RK_.loc[PROP, c] / RK_.loc[b_, c] - 1):+.0f}".replace("-0", "0").replace("+0", "0") if abs(100 * (RK_.loc[PROP, c] / RK_.loc[b_, c] - 1)) < 0.5 else f"{100 * (RK_.loc[PROP, c] / RK_.loc[b_, c] - 1):+.0f}" for c in ["mean", "std", "sharpe", "var", "es", "evar", "mdd"]) + " \\\\")
body = body.replace("\\bottomrule", "\\midrule\n\\multicolumn{8}{l}{\\emph{Relative difference of the proposed portfolio (\\%)}}\\\\\n" + "\n".join(rel) + "\n\\bottomrule")
write_table(os.path.join(TAB, "tab_risk.tex"),
            f"Comparison with Markowitz mean--variance (MV) portfolios and tail-risk measures on the hold-out period ({HOLDOUT_YEARS[0]}--{HOLDOUT_YEARS[-1]}). "
            "The MV portfolios are long-only, estimated on the three-year decision window and re-estimated every January. Mean: annualised excess return (\\%); "
            "StD: annualised volatility (\\%); VaR, ES and EVaR: daily value-at-risk, expected shortfall and expectile value-at-risk at $\\alpha=5\\%$ "
            "by historical simulation (\\%); maximum drawdown (\\%). MV weights: minimum-variance weights within the ten selected industries. Lower panel: relative difference of the proposed portfolio to each benchmark.",
            "tab:risk", body, size="\\scriptsize")
for b_, key in [("1/N", "EW"), ("Minimum variance", "MinVar"), ("Maximum Sharpe ratio", "MaxSh"), ("Market", "Mkt")]:
    for c, cn in [("std", "Std"), ("var", "Var"), ("es", "ES"), ("evar", "EVar"), ("mean", "Mean")]:
        mac("Rel" + cn + key, 100 * (RK_.loc[PROP, c] / RK_.loc[b_, c] - 1), "{:+.0f}")
for m, key in [(PROP, "Prop"), ("1/N", "EW"), ("Minimum variance", "MinVar"), ("Maximum Sharpe ratio", "MaxSh")]:
    mac("TailES" + key, RK_.loc[m, "es"], "{:.2f}")
    mac("TailVar" + key, RK_.loc[m, "var"], "{:.2f}")
    mac("MVSharpe" + key, RK_.loc[m, "sharpe"], "{:.2f}")
    mac("MVVol" + key, RK_.loc[m, "std"], "{:.1f}")
    mac("MVRet" + key, RK_.loc[m, "mean"], "{:.1f}")
    mac("MVMdd" + key, RK_.loc[m, "mdd"], "{:.0f}")
mac("MVSharpePMinVar", fp(RK_.loc["Minimum variance", "p_sharpe"]))
for c_, cn_ in [("sharpe", "Sharpe"), ("es", "ES"), ("std", "Vol"), ("mean", "Ret"), ("mdd", "Mdd")]:
    mac("Two" + cn_, RK_.loc[TWO, c_], "{:.2f}" if c_ in ("sharpe", "es") else ("{:.0f}" if c_ == "mdd" else "{:.1f}"))
mac("MVSharpePMaxSh", fp(RK_.loc["Maximum Sharpe ratio", "p_sharpe"]))
mac("MVTurnMinVar", 100 * PF["Minimum variance"]["turnover"], "{:.0f}")
mac("MVTurnMaxSh", 100 * PF["Maximum Sharpe ratio"]["turnover"], "{:.0f}")
nheld = [int((mv_weights(ty, "minvar") > 0.01).sum()) for ty in HOLDOUT_YEARS]
mac("MVMinVarHeld", float(np.mean(nheld)), "{:.0f}")

# holding (control) periods of 3, 6 and 12 months after each decision
hz = []
for months, name in [(3, "3 months"), (6, "6 months"), (12, "12 months")]:
    row = {"Holding period": name}
    for m, key in [(PROP, "Proposed"), ("1/N", "$1/N$"), ("Minimum variance", "Min.\\ variance")]:
        s_ = SER[m][SER[m].index.month <= months]
        ex_ = s_ - F.rf.loc[s_.index]
        row[key + ": Sharpe"] = ex_.mean() / s_.std() * np.sqrt(252)
        row[key + ": ES"] = tail(s_)["es"]
    hz.append(row)
HZ = pd.DataFrame(hz).set_index("Holding period")
HZ.to_csv(os.path.join(RES, "holding_periods.csv"))
write_table(os.path.join(TAB, "tab_horizon.tex"),
            "Sharpe ratio and daily expected shortfall (ES, 5\\%, in \\%) of the portfolios when they are held for the first 3, 6 or 12 months "
            f"after each January decision ({len(HOLDOUT_YEARS)} hold-out decisions).", "tab:horizon", df_tex(HZ, fmt="{:.2f}", colfmt="lrrrrrr"))
mac("HzSharpeThreeProp", HZ.iloc[0, 0], "{:.2f}")
mac("HzSharpeThreeEW", HZ.iloc[0, 2], "{:.2f}")
mac("HzSharpeSixProp", HZ.iloc[1, 0], "{:.2f}")
mac("HzSharpeSixEW", HZ.iloc[1, 2], "{:.2f}")
mac("HzAllBeatEW", "all" if (HZ.iloc[:, 0] > HZ.iloc[:, 2]).all() else "not all")
mac("HzAllLowerES", "all" if (HZ.iloc[:, 1] < HZ.iloc[:, 3]).all() else "not all")

# ============================================================================ 9. criteria weights and investor profiles
print("criteria weights ...", flush=True)


def sd_weights(X):
    Xv = np.asarray(X, float)
    lo, hi = Xv.min(0), Xv.max(0)
    N = np.where(L.BENEFIT, (Xv - lo) / (hi - lo), (hi - Xv) / (hi - lo))
    s_ = N.std(axis=0, ddof=1)
    return s_ / s_.sum()


RET_C, RISK_C = np.array([1, 1, 0, 0, 0, 0, 1], bool), np.array([0, 0, 1, 1, 1, 1, 0], bool)


def profile(w, tilt):
    """Investor profiles: CRITIC weights with the risk criteria (C3-C6) multiplied by `tilt` and the
    return-oriented criteria (C1, C2, C7) divided by `tilt`, renormalised."""
    v = w * np.where(RISK_C, tilt, 1 / tilt)
    return v / v.sum()


SCHEMES = {"CRITIC (baseline)": lambda X, w: w,
           "Equal weights": lambda X, w: np.full(len(w), 1 / len(w)),
           "Standard deviation": lambda X, w: sd_weights(X),
           "Conservative profile": lambda X, w: profile(w, 2.0),
           "Aggressive profile": lambda X, w: profile(w, 0.5)}
WM = [PROP, "SF-TOPSIS", "TOPSIS"]


def score3(E, X, w):
    return {PROP: L.st2hfs_topsis(L.rescale(E), w, **SETTING)[0], "SF-TOPSIS": L.sf_topsis(E, w), "TOPSIS": L.topsis(X.values, w)}


wrows, WSER = [], {}
for sname, fw in SCHEMES.items():
    agree = {m: [] for m in WM}
    ru, parts, betas_w = [], [], []
    for ty in HOLDOUT_YEARS:
        mon_, a_, E_, X_, w_ = hold[ty]
        wv = fw(X_, w_)
        sc = score3(E_, X_, wv)
        base = score3(E_, X_, w_)
        for m in WM:
            agree[m].append(spearmanr(sc[m], base[m]).correlation)
        ru.append(spearmanr(sc[PROP], realised(ty, a_, wv)["attain"]).correlation)
        top = pd.Series(sc[PROP], index=a_).sort_values(ascending=False).index[:TOPK]
        parts.append(R.loc[str(ty), top].mean(axis=1))
        betas_w.append(wv[RISK_C].sum())
    pr = pd.concat(parts)
    WSER[sname] = pr
    pf = perf(pr)
    wrows.append({"Weights": sname, "Risk weight": np.mean(betas_w), "Prop.": np.mean(agree[PROP]), "SF": np.mean(agree["SF-TOPSIS"]),
                  "TOPSIS": np.mean(agree["TOPSIS"]), "$\\rho_U$": np.mean(ru), "Return": 100 * pf["exret"], "StD": 100 * pf["vol"],
                  "Sharpe": pf["sharpe"], "Beta": pf["beta"], "ES": tail(pr)["es"]})
WT = pd.DataFrame(wrows).set_index("Weights")
WT.to_csv(os.path.join(RES, "sensitivity_weights.csv"))
body = df_tex(WT, fmt="{:.2f}", colfmt="lrrrrrrrrrr")
body = body.replace("\\toprule", "\\toprule\n & & \\multicolumn{3}{c}{Agreement with baseline ($\\rho$)} & & \\multicolumn{5}{c}{Top-" + str(TOPK)
                    + " portfolio of the proposed framework}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){7-11}", 1)
write_table(os.path.join(TAB, "tab_weights.tex"),
            f"Sensitivity to the criteria weights and investor profiles on the hold-out period ({len(HOLDOUT_YEARS)} annual decisions). Risk weight: mean total "
            "weight of the risk criteria $C_3$--$C_6$; agreement: mean Spearman correlation between the ranking under the given weights and the ranking under "
            "the CRITIC weights, for the proposed framework (Prop.), SF-TOPSIS (SF) and crisp TOPSIS; $\\rho_U$: correlation of the scores of the proposed "
            "framework with next-year criteria attainment computed with the same weights; annualised excess return and volatility (\\%), Sharpe ratio, "
            "market beta and daily expected shortfall (ES, 5\\%, in \\%) of the portfolio of the ten best-ranked industries.", "tab:weights", body, size="\\scriptsize")
for sname, key in [("Equal weights", "Eq"), ("Standard deviation", "SD"), ("Conservative profile", "Cons"), ("Aggressive profile", "Aggr"), ("CRITIC (baseline)", "Base")]:
    s = WT.loc[sname]
    mac("W" + key + "Agree", s["Prop."])
    mac("W" + key + "AgreeTop", s["TOPSIS"])
    mac("W" + key + "U", s["$\\rho_U$"])
    mac("W" + key + "Vol", s["StD"], "{:.1f}")
    mac("W" + key + "Beta", s["Beta"], "{:.2f}")
    mac("W" + key + "Sharpe", s["Sharpe"], "{:.2f}")
    mac("W" + key + "Ret", s["Return"], "{:.1f}")
    mac("W" + key + "ES", s["ES"], "{:.2f}")
    mac("W" + key + "Risk", s["Risk weight"], "{:.2f}")

# Monte Carlo perturbation of the weights
rng = np.random.default_rng(SEED + 2)
NW = 100
mc = {(m, t): [] for m in WM for t in ("mild", "random")}
for ty in HOLDOUT_YEARS:
    mon_, a_, E_, X_, w_ = hold[ty]
    base = score3(E_, X_, w_)
    for _ in range(NW):
        wm_ = w_ * rng.uniform(0.5, 1.5, len(w_))
        for t, wv in (("mild", wm_ / wm_.sum()), ("random", rng.dirichlet(np.ones(len(w_))))):
            sc = score3(E_, X_, wv)
            for m in WM:
                mc[(m, t)].append(kendalltau(sc[m], base[m]).statistic)
MC = pd.Series({k_: np.mean(v) for k_, v in mc.items()}).unstack()
MC.to_csv(os.path.join(RES, "sensitivity_weights_montecarlo.csv"))
mac("WMCn", NW * len(HOLDOUT_YEARS), "{:,d}")
for m, key in [(PROP, "Prop"), ("SF-TOPSIS", "SF"), ("TOPSIS", "TOP")]:
    mac("WMCMild" + key, MC.loc[m, "mild"])
    mac("WMCRandom" + key, MC.loc[m, "random"])

# ============================================================================ 10. reliability flag from the secondary grades
# Within every hold-out decision the industries are split by the extremeness of their score (|score - median|), so that the
# comparison is not driven by range restriction; within the more extreme half, the ranking is evaluated separately for the
# industries with high and low weighted mean secondary grade (classification consistency).  The same split is applied with the
# conventional dispersion measure (weighted standard deviation of the monthly percentile ranks) for comparison.
print("reliability flag ...", flush=True)
rel = []
for ty in HOLDOUT_YEARS:
    mon_ = {m: MONTHLY[m] for m in MONTHLY if m.year in range(ty - K, ty)}
    sc, sec, U = SCORES[ty][PROP].values, SECG[ty].values, OUTC[ty].values
    ranks = []
    for m in mon_:
        Xm = mon_[m].loc[SECG[ty].index].values.copy()
        Xm[:, ~L.BENEFIT] *= -1
        ranks.append(pd.DataFrame(Xm).rank(pct=True).values)
    wv = window(list(range(ty - K, ty)))[4]
    rsd = (np.std(ranks, axis=0) * wv).sum(1)
    h = len(sc) // 2
    sub = np.argsort(-np.abs(sc - np.median(sc)))[:h]
    row = dict(year=ty, corr_sec_ext=spearmanr(sec, np.abs(sc - np.median(sc))).correlation,
               corr_sec_rsd=spearmanr(sec, rsd).correlation)
    for name, conf in (("sec", sec), ("rsd", -rsd)):
        c = conf[sub]
        hi, lo = sub[c > np.median(c)], sub[c <= np.median(c)]
        row[name + "_hi"] = spearmanr(sc[hi], U[hi]).correlation
        row[name + "_lo"] = spearmanr(sc[lo], U[lo]).correlation
    # unconditional split of the whole cross-section (for reference)
    hiA = sec > np.median(sec)
    row["all_hi"], row["all_lo"] = spearmanr(sc[hiA], U[hiA]).correlation, spearmanr(sc[~hiA], U[~hiA]).correlation
    rel.append(row)
REL = pd.DataFrame(rel)
REL.to_csv(os.path.join(RES, "reliability_flag.csv"), index=False)
for name, key in (("sec", "Sec"), ("rsd", "Rsd")):
    d_, p_, ci, _ = block_boot(REL[name + "_hi"] - REL[name + "_lo"])
    mac("Rel" + key + "Hi", REL[name + "_hi"].mean())
    mac("Rel" + key + "Lo", REL[name + "_lo"].mean())
    mac("Rel" + key + "P", fp(p_))
    mac("Rel" + key + "Wins", int((REL[name + "_hi"] > REL[name + "_lo"]).sum()), "{:d}")
mac("RelAllHi", REL.all_hi.mean())
mac("RelAllLo", REL.all_lo.mean())
mac("RelCorrSecExt", REL.corr_sec_ext.mean(), "{:.2f}")
mac("RelCorrSecRsd", REL.corr_sec_rsd.mean(), "{:.2f}")
TRel = pd.DataFrame({"High": [REL.all_hi.mean(), REL.sec_hi.mean(), REL.rsd_hi.mean()],
                     "Low": [REL.all_lo.mean(), REL.sec_lo.mean(), REL.rsd_lo.mean()],
                     "Decisions with High $>$ Low": [int((REL.all_hi > REL.all_lo).sum()), int((REL.sec_hi > REL.sec_lo).sum()),
                                                     int((REL.rsd_hi > REL.rsd_lo).sum())],
                     "Block-bootstrap $p$": [block_boot(REL.all_hi - REL.all_lo)[1], block_boot(REL.sec_hi - REL.sec_lo)[1],
                                             block_boot(REL.rsd_hi - REL.rsd_lo)[1]]},
                    index=["Secondary grade, all industries", "Secondary grade, extreme half", "Rank dispersion, extreme half"])
TRel["Decisions with High $>$ Low"] = TRel["Decisions with High $>$ Low"].astype(int).astype(str) + f"/{len(REL)}"
TRel["Block-bootstrap $p$"] = ["$<$0.001" if v < 0.001 else f"{v:.3f}" for v in TRel["Block-bootstrap $p$"]]
write_table(os.path.join(TAB, "tab_reliability.tex"),
            f"Reliability flag on the hold-out period ({len(HOLDOUT_YEARS)} decisions): mean Spearman correlation between the scores of the "
            "proposed framework and next-year criteria attainment, computed separately for the industries with high and low confidence "
            "(split at the median within each decision). Confidence is the CRITIC-weighted mean secondary grade, or the negative weighted "
            "standard deviation of the monthly percentile ranks (conventional dispersion). ``Extreme half'': only the half of the industries "
            "whose scores lie farthest from the median, which removes the advantage that high-confidence industries have from being ranked "
            "near the top or bottom. $p$: moving-block bootstrap test (blocks of three years) of the mean difference High $-$ Low.",
            "tab:reliability", df_tex(TRel, colfmt="lrrrr"))

# ============================================================================ 11. scalability
rng = np.random.default_rng(0)
scal = []
for n in (100, 1000, 5000):
    for m_ in (7, 20):
        Pr = rng.dirichlet([1, 1, 1], size=(n, m_, 5)).transpose(0, 1, 3, 2)
        Es = np.stack([Pr, rng.uniform(0.5, 1, Pr.shape)], axis=-1)
        Es = np.take_along_axis(Es, np.lexsort((Es[..., 1], Es[..., 0]), axis=-1)[..., None], axis=3)
        t0 = time.perf_counter()
        L.st2hfs_topsis(L.rescale(Es), np.full(m_, 1 / m_), **SETTING)
        scal.append(dict(n=n, m=m_, seconds=time.perf_counter() - t0))
SCL = pd.DataFrame(scal)
SCL.to_csv(os.path.join(RES, "scalability.csv"), index=False)
mac("ScalThousand", SCL[(SCL.n == 1000) & (SCL.m == 20)].seconds.iloc[0], "{:.2f}")
mac("ScalFiveK", SCL[(SCL.n == 5000) & (SCL.m == 20)].seconds.iloc[0], "{:.2f}")

with open(os.path.join(ROOT, "numbers.tex"), "w") as f:
    f.write("% generated by code/run_experiments.py -- do not edit by hand\n")
    for k_, v in sorted(MAC.items()):
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(f"done in {time.perf_counter() - t_start:.0f} s; {len(MAC)} macros written")
