"""V1 CHAMPION — 11-feature super combo, trained and evaluated standalone.
Input : dengue_with_population.csv  (same folder as this script)
Output: v1_champion/  (model, predictions, importance, metrics)
Guarantees: Nov-2026 dropped, time-ordered early stop, master province order,
            leakage guard, seasonal-naive null, single-threaded reproducibility."""
import warnings; warnings.filterwarnings("ignore")
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
CSV  = BASE / "dengue_with_population.csv"
OUT  = BASE / "v1_champion"; OUT.mkdir(exist_ok=True)

if not CSV.exists():
    sys.exit(f"ERROR: put dengue_with_population.csv next to this script.\nMissing: {CSV}")

# ============================================================
# CONFIG
# ============================================================
TARGET      = "cases"
TRAIN_END   = pd.Timestamp("2025-01-01")     # train = week_start < this
EVAL_CUTOFF = pd.Timestamp("2026-11-01")     # ALWAYS drop incomplete Nov-2026
CASE_LAGS   = (1, 2, 3, 4, 8, 12)
SEED, THREADS = 42, 1                        # 1 thread => bit-reproducible
FINAL_EARLY_STOP = 150

# V1 champion feature set (the 11 that won every stress test)
FEATURES = [
    "cases_L1", "cases_L2", "cases_L3", "cases_L4", "cases_L8",
    "outbreak_momentum", "cases_L1_relative", "cases_L2_relative",
    "cos_week", "sin_week",
    "w_apparent_temperature_mean",
]

# Embedded champion hyperparameters (from the validated Optuna run)
CHAMPION_PARAMS = dict(
    learning_rate=0.0788, num_leaves=105, min_child_samples=67,
    feature_fraction=0.4475, bagging_fraction=0.5303, bagging_freq=2,
    reg_alpha=0.0392, reg_lambda=0.2535, max_depth=3, path_smooth=4.8471,
)
CHAMPION_ROUNDS = 1327

# ============================================================
# LOAD + DROP NOV-2026 + LOCK MASTER ORDER
# ============================================================
t0 = time.time()
df = pd.read_csv(CSV, low_memory=False)
df["province"]   = df["province"].astype(str).str.strip()
df["week_start"] = pd.to_datetime(df["week_start"])
df = df.dropna(subset=["province", "week_start", TARGET]).copy()

PROVINCE_ORDER = list(dict.fromkeys(df["province"]))   # น่าน-first, before any sort
print(f"Master order (first 3): {PROVINCE_ORDER[:3]}")

n_before = len(df)
df = df[df["week_start"] < EVAL_CUTOFF].copy()          # unconditional Nov-2026 drop
print(f"Dropped {n_before - len(df)} incomplete rows >= {EVAL_CUTOFF.date()} (now {len(df)})")

df = df.sort_values(["province", "week_start"]).reset_index(drop=True)   # required for .shift
df["year"] = df["week_start"].dt.year
for c in df.columns:
    if isinstance(c, str) and (c.startswith("w_") or c in ("cos_week", "sin_week", "sin2_week")):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# FEATURE ENGINEERING (past-only, leakage-safe)
# ============================================================
for L in CASE_LAGS:
    df[f"cases_L{L}"] = df.groupby("province")[TARGET].shift(L)

train_mask = (df["week_start"] < TRAIN_END).to_numpy()
base = df.loc[train_mask].groupby("province")[TARGET].mean()
df["prov_base"] = df["province"].map(base).fillna(df.loc[train_mask, TARGET].mean())
for L in CASE_LAGS:
    df[f"cases_L{L}_relative"] = df[f"cases_L{L}"] / (df["prov_base"] + 1.0)
df["outbreak_momentum"] = df["cases_L1_relative"] / (df["cases_L4_relative"] + 0.1)

# leakage guard
assert "cases_per_100k" not in FEATURES and TARGET not in FEATURES, "LEAKAGE!"
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    sys.exit(f"ERROR: features missing after engineering: {missing}")
print(f"✓ {len(FEATURES)} features ready | leakage guard passed")

Y = df[TARGET].astype(float).clip(lower=0).to_numpy()
tr_pos = np.where(train_mask)[0]
te_pos = np.where(~train_mask)[0]
print(f"train={len(tr_pos)}  holdout={len(te_pos)}  "
      f"({df.iloc[te_pos]['week_start'].min().date()} -> {df.iloc[te_pos]['week_start'].max().date()})")

def frame(pos): return df.iloc[pos][FEATURES].to_numpy(np.float32)
def ds(pos, lab): return lgb.Dataset(frame(pos), label=lab, free_raw_data=False)

# ============================================================
# SEASONAL-NAIVE NULL
# ============================================================
trd = df.iloc[tr_pos].copy(); trd["woy"] = trd["week_start"].dt.isocalendar().week.astype(int)
lut = trd.groupby(["province", "woy"])[TARGET].mean()
gm  = trd.groupby("province")[TARGET].mean()
ted = df.iloc[te_pos].copy(); ted["woy"] = ted["week_start"].dt.isocalendar().week.astype(int)
naive = np.array([lut.get((p, w), gm.get(p, trd[TARGET].mean()))
                  for p, w in zip(ted.province, ted.woy)], float)

# ============================================================
# TIME-ORDERED EARLY STOP (date quantile, not row slice)
# ============================================================
tr_times = df.loc[tr_pos, "week_start"].values
cut_date = np.quantile(tr_times, 0.9)
tr_in = tr_pos[tr_times <  cut_date]
va_in = tr_pos[tr_times >= cut_date]
print(f"Early-stop: val starts {pd.Timestamp(cut_date).date()} (tr_in={len(tr_in)} va_in={len(va_in)})")

BASE_P = dict(objective="poisson", verbose=-1, seed=SEED,
              num_threads=THREADS, n_jobs=THREADS)

es = dict(BASE_P, **CHAMPION_PARAMS, metric="poisson")
m = lgb.train(es, ds(tr_in, Y[tr_in]), num_boost_round=CHAMPION_ROUNDS,
              valid_sets=[ds(va_in, Y[va_in])],
              callbacks=[lgb.early_stopping(FINAL_EARLY_STOP, verbose=False)])
iters = m.best_iteration or CHAMPION_ROUNDS
print(f"Best iteration: {iters}")

fp = dict(BASE_P, **CHAMPION_PARAMS, metric="None")
m = lgb.train(fp, ds(tr_pos, Y[tr_pos]), num_boost_round=iters)

# ============================================================
# PREDICT + METRICS
# ============================================================
pred = np.clip(m.predict(frame(te_pos)), 0, None)
y = Y[te_pos]

EPS = 1e-9
def pdev(yy, pp):
    yy = np.asarray(yy, float); pp = np.clip(np.asarray(pp, float), EPS, None)
    return float(2 * np.mean(np.where(yy > 0, yy * np.log((yy + EPS) / pp), 0.0) - (yy - pp)))

ss = np.sum((y - y.mean()) ** 2)
mae   = float(np.mean(np.abs(y - pred)))
rmse  = float(np.sqrt(np.mean((y - pred) ** 2)))
wape  = float(np.sum(np.abs(y - pred)) / np.sum(np.abs(y)))
r2    = float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else np.nan
nm    = float(np.mean(np.abs(y - naive)))
skill = (nm - mae) / nm * 100

print(f"\n{'='*60}")
print(f"V1 CHAMPION — holdout metrics")
print(f"{'='*60}")
print(f"  MAE    = {mae:.4f}")
print(f"  RMSE   = {rmse:.4f}")
print(f"  WAPE   = {wape:.4f}")
print(f"  R2     = {r2:.4f}")
print(f"  PDev   = {pdev(y, pred):.4f}")
print(f"  Naive  = {nm:.4f}")
print(f"  Skill  = {skill:.1f}%")
print(f"{'='*60}")

# ============================================================
# SAVE (master province order)
# ============================================================
out = df.iloc[te_pos][["province", "week_start", "year", TARGET]].copy()
out["pred"]     = pred
out["naive"]    = naive
out["residual"] = out[TARGET] - out["pred"]
out["province"] = pd.Categorical(out["province"], categories=PROVINCE_ORDER, ordered=True)
out = (out.sort_values(["province", "week_start"])
          .assign(province=lambda d: d["province"].astype(str))
          .reset_index(drop=True))
out.to_csv(OUT / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

imp = pd.DataFrame({"feature": FEATURES, "gain": m.feature_importance("gain")})
imp["gain_pct"] = imp.gain / imp.gain.sum() * 100
imp.sort_values("gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

json.dump({"features": FEATURES, "params": {**BASE_P, **CHAMPION_PARAMS},
           "rounds": iters, "mae": mae, "rmse": rmse, "wape": wape,
           "r2": r2, "naive_mae": nm, "skill_pct": skill,
           "train_end": str(TRAIN_END.date()), "eval_cutoff": str(EVAL_CUTOFF.date()),
           "n_train": len(tr_pos), "n_holdout": len(te_pos),
           "holdout_min": str(df.iloc[te_pos]["week_start"].min().date()),
           "holdout_max": str(df.iloc[te_pos]["week_start"].max().date())},
          open(OUT / "metrics.json", "w"), indent=2, default=str)

m.save_model(str(OUT / "model.txt"))

print(f"\nFirst 3 provinces: {out['province'].head(3).tolist()}")
print(f"Saved -> {OUT}")
print(f"DONE. ({time.time()-t0:.0f}s)")