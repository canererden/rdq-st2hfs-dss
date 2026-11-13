"""
rdq_st2hfs -- expert-free r,d,q-spherical type-2 hesitant fuzzy TOPSIS for financial asset selection.

Core library used by run_experiments.py:

  * data loading (Kenneth R. French Data Library: 49 industry portfolios and factors, daily);
  * monthly crisp criteria;
  * data-driven construction of r,d,q-ST2HFEs (three-way classification, Dirichlet-smoothed
    primary grades, Wilson-precision secondary grades, yearly scenarios as hesitant elements);
  * admissibility check and admissibility-preserving rescaling;
  * CRITIC weights;
  * weighted hybrid (Dice/cosine) similarity measure and the five TOPSIS steps;
  * benchmark methods (spherical fuzzy TOPSIS, hesitant fuzzy TOPSIS, TOPSIS, WASPAS, EDAS, CoCoSo).

Array convention for a decision matrix E:
    E[alternative, criterion, component (0 = truth, 1 = indeterminacy, 2 = falsity),
      hesitant element (ascending order), grade (0 = primary, 1 = secondary)]
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

CRITERIA = ["C1 Monthly return", "C2 Reward-to-risk", "C3 Volatility", "C4 VaR (95%)",
            "C5 Max. drawdown", "C6 Beta", "C7 Momentum (12-1)"]
BENEFIT = np.array([1, 1, 0, 0, 0, 0, 1], bool)      # False = cost criterion
EPS = 1e-6                                           # slack of the rescaling

IND49 = ("Agric Food Soda Beer Smoke Toys Fun Books Hshld Clths Hlth MedEq Drugs Chems Rubbr Txtls BldMt "
         "Cnstr Steel FabPr Mach ElcEq Autos Aero Ships Guns Gold Mines Coal Oil Util Telcm PerSv BusSv "
         "Hardw Softw Chips LabEq Paper Boxes Trans Whlsl Rtail Meals Banks Insur RlEst Fin Other").split()

IND_NAMES = {
    "Agric": "Agriculture", "Food": "Food products", "Soda": "Candy and soda", "Beer": "Beer and liquor",
    "Smoke": "Tobacco products", "Toys": "Recreation", "Fun": "Entertainment", "Books": "Printing and publishing",
    "Hshld": "Consumer goods", "Clths": "Apparel", "Hlth": "Healthcare", "MedEq": "Medical equipment",
    "Drugs": "Pharmaceutical products", "Chems": "Chemicals", "Rubbr": "Rubber and plastic products",
    "Txtls": "Textiles", "BldMt": "Construction materials", "Cnstr": "Construction", "Steel": "Steel works",
    "FabPr": "Fabricated products", "Mach": "Machinery", "ElcEq": "Electrical equipment",
    "Autos": "Automobiles and trucks", "Aero": "Aircraft", "Ships": "Shipbuilding and railroad equipment",
    "Guns": "Defense", "Gold": "Precious metals", "Mines": "Non-metallic and industrial metal mining",
    "Coal": "Coal", "Oil": "Petroleum and natural gas", "Util": "Utilities", "Telcm": "Communication",
    "PerSv": "Personal services", "BusSv": "Business services", "Hardw": "Computers",
    "Softw": "Computer software", "Chips": "Electronic equipment", "LabEq": "Measuring and control equipment",
    "Paper": "Business supplies", "Boxes": "Shipping containers", "Trans": "Transportation",
    "Whlsl": "Wholesale", "Rtail": "Retail", "Meals": "Restaurants, hotels and motels", "Banks": "Banking",
    "Insur": "Insurance", "RlEst": "Real estate", "Fin": "Trading", "Other": "Other"}


# ============================================================================ data
def _read_french_zip(path):
    """First table (value-weighted daily returns / daily factors) of an official French Data Library CSV zip."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, line in enumerate(raw)
                 if line.split(",")[0].strip() == "" and len(line.split(",")) > 3)
    header = [x.strip() for x in raw[start].split(",")][1:]
    rows = []
    for line in raw[start + 1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows]), columns=header)
    return df.astype(float)


def load_data(datadir="data"):
    """Returns (R, F, source): daily value-weighted returns of the 49 industry portfolios (decimals),
    daily factors with columns mktrf, smb, hml, rmw, cma, rf (decimals), and a description of the source.

    If the official files 49_Industry_Portfolios_Daily_CSV.zip and
    F-F_Research_Data_5_Factors_2x3_daily_CSV.zip are present in `datadir`, they are used; otherwise the
    bundled extract (ff49_daily_vw.csv, ff_factors_daily.csv) is used."""
    z_ind = os.path.join(datadir, "49_Industry_Portfolios_Daily_CSV.zip")
    z_fac = os.path.join(datadir, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    if os.path.exists(z_ind) and os.path.exists(z_fac):
        R = _read_french_zip(z_ind).replace([-99.99, -999], np.nan) / 100
        F = _read_french_zip(z_fac) / 100
        F.columns = [c.lower().replace("mkt-rf", "mktrf") for c in F.columns]
        source = "official files"
    else:
        R = pd.read_csv(os.path.join(datadir, "ff49_daily_vw.csv"), index_col=0, parse_dates=True)
        F = pd.read_csv(os.path.join(datadir, "ff_factors_daily.csv"), index_col=0, parse_dates=True)
        source = "bundled extract"
    R = R[IND49].loc["1971-01-01":]
    F = F[["mktrf", "smb", "hml", "rmw", "cma", "rf"]].reindex(R.index)
    return R, F, source


def price_index(R):
    """Total-return index levels (NaN before the first observation of a series)."""
    return (1 + R.fillna(0)).cumprod().where(R.notna().cummax())


# ============================================================================ criteria
def monthly_criteria(P, market, start, end):
    """dict: month (Period) -> DataFrame (assets x 7 crisp criteria), from daily index levels P and
    the daily market return series `market`."""
    r = P.pct_change()
    per = r.index.to_period("M")
    out = {}
    for m in pd.period_range(start, end, freq="M"):
        idx = per == m
        if idx.sum() < 10:
            continue
        rr, mm, px = r[idx], market[idx], P[idx]
        ret = (1 + rr).prod() - 1                                   # C1 monthly return
        rtr = rr.mean() / rr.std()                                  # C2 reward-to-risk of daily returns
        vol = rr.std() * np.sqrt(252)                               # C3 annualised volatility
        var = -rr.quantile(0.05)                                    # C4 historical 95% value-at-risk
        dd = (px / px.cummax() - 1).min().abs()                     # C5 maximum drawdown within the month
        beta = rr.apply(lambda c: np.cov(c, mm)[0, 1] / mm.var())   # C6 beta against the market
        t0 = rr.index[0]                                            # C7 momentum from month t-12 to t-1
        hist = P[P.index < t0]
        mom = hist.iloc[-1] / hist[hist.index <= t0 - pd.DateOffset(months=12)].iloc[-1] - 1
        out[m] = pd.DataFrame(dict(zip(CRITERIA, [ret, rtr, vol, var, dd, beta, mom])))
    return out


def crisp_matrix(monthly, years):
    """Window-average crisp decision matrix (used by CRITIC and by the crisp benchmarks)."""
    ms = [m for m in monthly if m.year in years]
    return sum(monthly[m] for m in ms) / len(ms)


# ============================================================================ fuzzification
def wilson_width(p, n, z=1.96):
    """Width of the Wilson score interval of a proportion p estimated from n observations."""
    return 2 * z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)


def build_st2hfe(monthly, years, thresh=1 / 3):
    """Data-driven r,d,q-ST2HFEs.  Each month every asset is classified on every criterion as good
    (percentile rank > 1 - thresh), bad (<= thresh) or neutral.  For each year, primary grades are the
    Dirichlet(1,1,1) posterior means of the class probabilities and secondary grades are one minus the
    width of the 95% Wilson interval of the class frequency.  The years of the window are the hesitant
    elements, ordered by (primary, secondary)."""
    assets = list(next(iter(monthly.values())).index)
    A, C, Y = len(assets), len(CRITERIA), len(years)
    E = np.zeros((A, C, 3, Y, 2))
    for yi, y in enumerate(years):
        ms = [m for m in monthly if m.year == y]
        n = len(ms)
        cnt = np.zeros((A, C, 3))
        for m in ms:
            X = monthly[m].values.copy()
            X[:, ~BENEFIT] *= -1                                    # orient: larger = better
            u = pd.DataFrame(X).rank(pct=True, method="average").values
            cnt[:, :, 0] += u > 1 - thresh
            cnt[:, :, 1] += (u > thresh) & (u <= 1 - thresh)
            cnt[:, :, 2] += u <= thresh
        E[:, :, :, yi, 0] = (cnt + 1) / (n + 3)
        E[:, :, :, yi, 1] = 1 - wilson_width(cnt / n, n)
    order = np.lexsort((E[..., 1], E[..., 0]), axis=-1)             # ascending by primary, then secondary
    E = np.take_along_axis(E, order[..., None], axis=3)
    return assets, E


def max_primary_sum(E):
    """Sum of the maximal truth, indeterminacy and falsity primary grades of every element (A x C)."""
    return E[..., 0].max(axis=3).sum(axis=2)


def admissible_share(E, r, d, q):
    """Share of elements satisfying (0,0) <= (T+)^r (+) (I+)^d (+) (F+)^q <= (1,1), where T+, I+, F+ are
    the maximal pairs and <= is the lexicographic relation of the type-2 hesitant fuzzy elements."""
    mx = E[:, :, :, -1, :]                                          # maximal pair of each component
    p = mx[..., 0, 0] ** r + mx[..., 1, 0] ** d + mx[..., 2, 0] ** q
    s = mx[..., 0, 1] ** r + mx[..., 1, 1] ** d + mx[..., 2, 1] ** q
    return float(((p < 1) | ((p == 1) & (s <= 1))).mean())


def minimal_common_exponent(E, kmax=30):
    """Smallest k such that every element is admissible for r = d = q = k (None if k > kmax)."""
    for k in range(1, kmax + 1):
        if admissible_share(E, k, k, k) == 1.0:
            return k
    return None


def rescale(E):
    """Admissibility-preserving rescaling: the primary grades of each element are divided by
    (1 + EPS) * max{1, sum of its maximal primary grades}; secondary grades are unchanged."""
    E = E.copy()
    s = (1 + EPS) * np.maximum(max_primary_sum(E), 1.0)
    E[..., 0] /= s[:, :, None, None]
    return E


# ============================================================================ weights
def critic(X, benefit=BENEFIT):
    """CRITIC weights (Diakoulaki et al., 1995) from a crisp decision matrix."""
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    N = np.where(benefit, (X - lo) / (hi - lo), (hi - X) / (hi - lo))
    c = N.std(axis=0, ddof=1) * (1 - np.corrcoef(N, rowvar=False)).sum(axis=0)
    return c / c.sum()


# ============================================================================ proposed method
def ideal_solutions(E):
    """Step 2 (all criteria oriented as benefit criteria): positive ideal = (max T, min I, min F),
    negative ideal = (min T, max I, max F), taken over the alternatives pair by pair."""
    Ap, Am = np.empty(E.shape[1:]), np.empty(E.shape[1:])
    Ap[:, 0], Ap[:, 1], Ap[:, 2] = E[:, :, 0].max(0), E[:, :, 1].min(0), E[:, :, 2].min(0)
    Am[:, 0], Am[:, 1], Am[:, 2] = E[:, :, 0].min(0), E[:, :, 1].max(0), E[:, :, 2].max(0)
    return Ap, Am


def weighted_hybrid_similarity(Xa, Xb, w, r, d, q, lam):
    """Weighted hybrid vector similarity measure Hyb_{r,d,q-WST2HFS} between two r,d,q-ST2HFSs given as
    arrays (criteria x 3 x L x 2): lam * weighted Dice + (1 - lam) * weighted cosine.
    Primary and secondary grades of T, I and F are raised to the powers r, d and q."""
    a, b = Xa.copy(), Xb.copy()
    for k, p in enumerate((r, d, q)):
        a[:, k] **= p
        b[:, k] **= p
    L = a.shape[2]
    num = 2 * (a * b).sum(axis=(1, 2, 3))
    den = (a ** 2).sum(axis=(1, 2, 3)) + (b ** 2).sum(axis=(1, 2, 3))
    dice = (w * num / den).sum()
    ip = (w[:, None] * (a * b).sum(axis=(2, 3)) / L).sum()
    na = (w[:, None] * (a ** 2).sum(axis=(2, 3)) / L).sum()
    nb = (w[:, None] * (b ** 2).sum(axis=(2, 3)) / L).sum()
    cosine = ip / np.sqrt(na * nb)
    return lam * dice + (1 - lam) * cosine


def st2hfs_topsis(E, w, r=1, d=1, q=1, lam=0.5):
    """Steps 2-4: returns (RC, S+, S-) with RC = S+ / (S+ + S-); alternatives are ranked by decreasing RC."""
    Ap, Am = ideal_solutions(E)
    Sp = np.array([weighted_hybrid_similarity(E[i], Ap, w, r, d, q, lam) for i in range(E.shape[0])])
    Sm = np.array([weighted_hybrid_similarity(E[i], Am, w, r, d, q, lam) for i in range(E.shape[0])])
    return Sp / (Sp + Sm), Sp, Sm


# ---------------------------------------------------------------------------- ablations
def drop_secondary(E):
    """Abl-1: remove the secondary grades (type-1 hesitant elements)."""
    return E[..., :1]


def drop_hesitancy(E):
    """Abl-2: replace the hesitant elements of each component by their average pair."""
    return E.mean(axis=3, keepdims=True)


# ============================================================================ benchmarks
def sf_topsis(E, w):
    """Spherical fuzzy TOPSIS on the year-averaged primary grades (truth, indeterminacy, falsity):
    weighted normalised Euclidean distances to the best and worst observed grades."""
    M = E[..., 0].mean(axis=3)
    mu, pi, nu = M[..., 0], M[..., 1], M[..., 2]

    def dist(m, n_, p):
        return np.sqrt(0.5 * (w * ((mu - m) ** 2 + (nu - n_) ** 2 + (pi - p) ** 2)).sum(1))
    dp = dist(mu.max(0), nu.min(0), pi.min(0))
    dm = dist(mu.min(0), nu.max(0), pi.max(0))
    return dm / (dp + dm)


def hf_topsis(E, w):
    """Hesitant fuzzy TOPSIS (Xu & Zhang, 2013): the yearly truth grades form the hesitant element;
    hesitant normalised Euclidean distances to the best and worst observed elements."""
    H = np.sort(E[:, :, 0, :, 0], axis=2)
    dp = np.sqrt((w[:, None] * (H - H.max(0)) ** 2).mean(2).sum(1))
    dm = np.sqrt((w[:, None] * (H - H.min(0)) ** 2).mean(2).sum(1))
    return dm / (dp + dm)


def _shift_positive(X):
    X = np.asarray(X, float).copy()
    for j in range(X.shape[1]):
        lo = X[:, j].min()
        if lo <= 0:
            X[:, j] = X[:, j] - lo + 1e-3 * (X[:, j].max() - lo + 1e-9)
    return X


def topsis(X, w):
    X = np.asarray(X, float)
    V = X / np.sqrt((X ** 2).sum(0)) * w
    best = np.where(BENEFIT, V.max(0), V.min(0))
    worst = np.where(BENEFIT, V.min(0), V.max(0))
    dp, dm = np.sqrt(((V - best) ** 2).sum(1)), np.sqrt(((V - worst) ** 2).sum(1))
    return dm / (dp + dm)


def waspas(X, w, lam=0.5):
    X = _shift_positive(X)
    N = np.where(BENEFIT, X / X.max(0), X.min(0) / X)
    return lam * (N * w).sum(1) + (1 - lam) * np.prod(N ** w, axis=1)


def edas(X, w):
    X = np.asarray(X, float)
    av = X.mean(0)
    sgn = np.where(BENEFIT, 1, -1)
    den = np.where(np.abs(av) > 0, np.abs(av), 1)
    sp = (np.maximum(0, sgn * (X - av)) / den * w).sum(1)
    sn = (np.maximum(0, -sgn * (X - av)) / den * w).sum(1)
    return (sp / sp.max() + 1 - sn / sn.max()) / 2


def cocoso(X, w, lam=0.5):
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    N = np.where(BENEFIT, (X - lo) / (hi - lo), (hi - X) / (hi - lo))
    S, P = (N * w).sum(1), (N ** w).sum(1)
    ka = (S + P) / (S + P).sum()
    kb = S / max(S.min(), 1e-9) + P / max(P.min(), 1e-9)
    kc = (lam * S + (1 - lam) * P) / (lam * S.max() + (1 - lam) * P.max())
    return (ka * kb * kc) ** (1 / 3) + (ka + kb + kc) / 3
