"""STEP 1 — Feature selection FROM SCRATCH (no hardcoded winners).
Faithful to comprehensive_search.py: includes rolling + missing-indicator
features in the pool so the Stage-1 funnel reproduces the original 42 ->
original gain top-12 -> original 10-feature winner.
Input : dengue_with_population.csv  (same folder as this script)
Output: features_pick/selected_features.json"""
import warnings; warnings.filterwarnings("ignore")
import itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed

BASE = Path(__file__).resolve().parent
CSV  = BASE / "dengue_with_population.csv"
OUT  = BASE / "features_pick"
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    sys.exit(f"ERROR: put dengue_with_population.csv next to this script.\nMissing: {CSV}")

TARGET, SEED, THREADS, N_JOBS = "cases", 42, 1, 4
TRAIN_END = pd.Timestamp("2025-01-01")
TOP_K     = 12
LAGS      = (1, 2, 3, 4, 8, 12)
ROLL_WINDOWS = (2, 4)

# ---------------- load ----------------
df = pd.read_csv(CSV, low_memory=False)
df["province"]   = df["province"].astype(str).str.strip()
df["week_start"] = pd.to_datetime(df["week_start"])
df = df.sort_values(["province", "week_start"]).reset_index(drop=True)
df["year"] = df["week_start"].dt.year
if "zone" not in df.columns: df["zone"] = "NA"
df["zone"] = df["zone"].astype(str).str.strip()
for c in df.columns:
    if c.startswith(("w_", "mos_")) or c in ("cos_week", "sin_week", "sin2_week", "cases_per_100k"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------- past-only feature engineering ----------------
for L in LAGS:
    df[f"cases_L{L}"] = df.groupby("province")[TARGET].shift(L)
    if "cases_per_100k" in df.columns:
        df[f"rate_L{L}"] = df.groupby("province")["cases_per_100k"].shift(L)
train_mask = (df["week_start"] < TRAIN_END).to_numpy()
base = df.loc[train_mask].groupby("province")[TARGET].mean()
df["prov_base"] = df["province"].map(base).fillna(df.loc[train_mask, TARGET].mean())
for L in LAGS:
    df[f"cases_L{L}_relative"] = df[f"cases_L{L}"] / (df["prov_base"] + 1.0)
df["outbreak_momentum"] = df["cases_L1_relative"] / (df["cases_L4_relative"] + 0.1)

# <<< EDIT 1: rolling features, PAST-shifted so leakage-safe >>>
def add_rolling_features(df, windows=ROLL_WINDOWS):
    exclude = {"w_sunrise", "w_sunset", "w_weather_code", "w_days_observed"}
    bases = [c for c in df.columns
             if c.startswith("w_") and c not in exclude
             and not c.endswith(("_L1", "_L2", "_L3", "_L4"))
             and not c.startswith("miss_")
             and not c.endswith(("_roll2", "_roll4"))
             and pd.api.types.is_numeric_dtype(df[c])]
    for b in bases:
        for w in windows:
            df[f"{b}_roll{w}"] = df.groupby("province")[b].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
    print(f"  Added rolling features on {len(bases)} base columns x {len(windows)} windows")
    return df
df = add_rolling_features(df)

# ---------------- candidate pool + leakage guard ----------------
EXCLUDE = {TARGET, "cases_per_100k", "week_start", "year", "prov_base",
           "w_sunrise", "w_sunset", "prov_lat", "prov_lon", "year_ce"}
pool = [c for c in df.columns if c not in EXCLUDE and (
        c.startswith(("w_", "mos_")) or c in ("cos_week", "sin_week", "sin2_week",
        "population", "log_population", "province", "zone")
        or c.startswith(("cases_L", "rate_L")) or c == "outbreak_momentum")]

CAT = [c for c in ("province", "zone") if c in pool]
for c in CAT: df[c] = df[c].astype("category")
CAT_DT = {c: df[c].dtype for c in CAT}

# <<< EDIT 2: missing indicators (matches original 0.5%-99.5% rule) >>>
new_miss = []
for c in list(pool):
    if c in CAT: continue
    r = df[c].isna().mean()
    if 0.005 < r < 0.995:
        nc = f"miss_{c}"
        df[nc] = df[c].isna().astype("int8")
        new_miss.append(nc)
pool = pool + new_miss
print(f"  Added {len(new_miss)} missing indicators")

pool = [c for c in pool if df[c].nunique(dropna=False) > 1]
assert "cases_per_100k" not in pool and TARGET not in pool, "LEAKAGE in pool!"
print(f"Candidate pool: {len(pool)} features | categorical: {CAT}")

# <<< EDIT 3: group_of checks miss_/roll FIRST so they don't mis-group >>>
def group_of(c):
    if c.startswith("miss_"): return "missing_indicators"
    if c.endswith(("_roll2", "_roll4")): return "rolling"
    if c in CAT: return "geo"
    if c in ("cos_week", "sin_week", "sin2_week"): return "seasonality"
    if c.startswith("mos_"): return "mosquito"
    if c in ("population", "log_population"): return "population"
    if c.startswith("rate_L"): return "rate_lags"
    if c == "outbreak_momentum": return "momentum"
    if c.endswith("_relative"): return "relative_lags"
    if c == "cases_L12": return "lag12"
    if c == "cases_L8": return "lag8"
    if c in (f"cases_L{L}" for L in (1, 2, 3, 4)): return f"lag{c[-1]}"
    if "soil_temperature" in c: return "soil_temperature"
    if "soil_moisture" in c: return "soil_moisture"
    if any(k in c for k in ("temperature", "dew_point", "wet_bulb")): return "temperature"
    if any(k in c for k in ("precipitation", "rain")): return "precipitation"
    if any(k in c for k in ("humidity", "vapour")): return "humidity"
    if "wind" in c: return "wind"
    if any(k in c for k in ("cloud", "radiation", "sunshine", "daylight")): return "radiation"
    if "pressure" in c: return "pressure"
    if "et0" in c: return "et0"
    return "other"

GROUPS = {}
for c in pool: GROUPS.setdefault(group_of(c), []).append(c)

# ---------------- rolling-origin CV folds (train window only) ----------------
search = df[train_mask].sort_values("week_start")
t0, t1 = search["week_start"].min(), search["week_start"].max()
total = (t1 - t0).days; n_val = max(1, int(total * 0.55 / 5)); min_tr = int(total * 0.45)
FOLDS = []
for i in range(5):
    ve = t0 + pd.Timedelta(days=min_tr + n_val * (i + 1)); vs = ve - pd.Timedelta(days=n_val)
    tc = vs - pd.Timedelta(weeks=4)
    tr = search.index[search["week_start"] <= tc].to_numpy()
    va = search.index[(search["week_start"] > vs) & (search["week_start"] <= ve)].to_numpy()
    if len(tr) > 500 and len(va) > 100: FOLDS.append((tr, va))
Y = df[TARGET].astype(float).clip(lower=0).to_numpy()
print(f"CV folds: {len(FOLDS)} (all inside train window)")

FAST = dict(objective="poisson", metric="None", learning_rate=0.06, num_leaves=31,
            min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1, seed=SEED,
            num_threads=THREADS, n_jobs=THREADS)

def mat(idx, feats):
    F = df.iloc[idx][list(feats)].copy()
    for c in F.columns:
        F[c] = F[c].astype(CAT_DT[c]) if c in CAT else F[c].to_numpy(np.float32)
    return F

def cv_deviance(feats, rounds=120):
    devs = []
    for tr, va in FOLDS:
        d = lgb.Dataset(mat(tr, feats), label=Y[tr],
                        categorical_feature=[c for c in feats if c in CAT], free_raw_data=False)
        m = lgb.train(FAST, d, num_boost_round=rounds)
        p = np.clip(m.predict(mat(va, feats)), 1e-9, None); y = Y[va]
        devs.append(2 * np.mean(np.where(y > 0, y * np.log(y / p), 0.0) - (y - p)))
    return float(np.mean(devs))

# ---------------- Stage 1: group-level greedy forward ----------------
t = time.time(); chosen_g, chosen, best = [], [], np.inf
for _ in range(len(GROUPS)):
    cands = [g for g in GROUPS if g not in chosen_g]
    if not cands: break
    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(cv_deviance)(list(dict.fromkeys(chosen + GROUPS[g]))) for g in cands)
    g_best, s_best = min(zip(cands, res), key=lambda x: x[1])
    if s_best >= best - 1e-9: break
    best = s_best; chosen_g.append(g_best); chosen = list(dict.fromkeys(chosen + GROUPS[g_best]))
    print(f"  Stage1 +{g_best:<18} -> {len(chosen):>3} feats | CV dev {best:.4f}")
print(f"Stage 1 done {time.time()-t:.0f}s: {len(chosen)} features  groups={chosen_g}")

# ---------------- Stage 2: single-feature greedy forward ----------------
t = time.time(); best = cv_deviance(chosen)
for _ in range(len(pool)):
    rem = [c for c in pool if c not in chosen]
    if not rem: break
    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(cv_deviance)(chosen + [c]) for c in rem)
    c_best, s_best = min(zip(rem, res), key=lambda x: x[1])
    if s_best >= best - 1e-9: break
    best = s_best; chosen.append(c_best)
    print(f"  Stage2 +{c_best:<28} -> {len(chosen):>3} feats | CV dev {best:.4f}")
print(f"Stage 2 done {time.time()-t:.0f}s: {len(chosen)} features")

# ---------------- Stage 3: backward elimination ----------------
t = time.time(); best = cv_deviance(chosen)
while len(chosen) > 1:
    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(cv_deviance)([c for c in chosen if c != d]) for d in chosen)
    d_best, s_best = min(zip(chosen, res), key=lambda x: x[1])
    if s_best >= best - 1e-9: break
    best = s_best; chosen.remove(d_best)
    print(f"  Stage3 -{d_best:<28} -> {len(chosen):>3} feats | CV dev {best:.4f}")
print(f"Stage 3 done {time.time()-t:.0f}s: {len(chosen)} survivors")

# ---------------- Stage 4: gain shortlist -> exhaustive wrapper ----------------
# rolling + missing_indicators EXCLUDED here (matches original SHORT_GROUPS)
t = time.time()
SHORT_GROUPS = {"geo", "seasonality", "mosquito", "temperature", "precipitation",
                "humidity", "soil_temperature", "et0", "lag1", "lag2", "lag8", "lag12",
                "population", "rate_lags", "relative_lags", "momentum"}
cand = list(dict.fromkeys(chosen + [c for c in pool if group_of(c) in SHORT_GROUPS]))
d = lgb.Dataset(mat(np.where(train_mask)[0], cand), label=Y[train_mask],
                categorical_feature=[c for c in cand if c in CAT], free_raw_data=False)
imp = pd.Series(lgb.train(FAST, d, num_boost_round=400).feature_importance("gain"),
                index=cand).sort_values(ascending=False)
shortlist = imp.head(TOP_K).index.tolist()
print("Shortlist by split-gain:", shortlist)

combos = [c for k in range(1, TOP_K + 1) for c in itertools.combinations(shortlist, k)]
print(f"Exhaustive wrapper search: {len(combos):,} subsets ...")
scores = Parallel(n_jobs=N_JOBS, backend="threading", verbose=5)(
    delayed(cv_deviance)(list(cb)) for cb in combos)
i = int(np.argmin(scores))
BEST, DEV = list(combos[i]), scores[i]

print(f"\nDISCOVERED WINNER ({len(BEST)} features) | CV deviance = {DEV:.4f}")
for c in BEST: print(f"  - {c}")
json.dump({"features": BEST, "cv_deviance": DEV,
           "domain_feature": "w_apparent_temperature_mean",
           "train_end": str(TRAIN_END.date())},
          open(OUT / "selected_features.json", "w"), indent=2)
print(f"Saved -> {OUT / 'selected_features.json'}  (stage 4 took {time.time()-t:.0f}s)")