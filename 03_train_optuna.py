"""STEP 3 — Tuning + training from the SLIM CSV (single file, all rows).
Input : features_pick/selected_features.json
        split_data/dengue_slime.csv  (column-strip output of 02)
Output: trained_data/  (model, predictions, importance, metrics)
The train/test row boundary is derived from TRAIN_END (date), NOT from
separate files. The slim CSV already contains engineered features, so
the engineering block below is a safe no-op (guarded by 'if col not in')."""
import warnings; warnings.filterwarnings("ignore")
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

BASE = Path(__file__).resolve().parent
P1   = BASE / "features_pick"
P2   = BASE / "split_data"              # <<< was "split_data"
OUT  = BASE / "trained_data"
OUT.mkdir(parents=True, exist_ok=True)

SLIM = P2 / "dengue_slime.csv"          # <<< single file, was train.csv + test.csv
for f in [P1 / "selected_features.json", SLIM]:
    if not f.exists():
        sys.exit(f"ERROR: missing {f}\nRun 01 and 02 first.")

TARGET, SEED, THREADS = "cases", 42, 1
ADD_DOMAIN_FEATURE = True
USE_SAVED_PARAMS   = True

sel = json.load(open(P1 / "selected_features.json"))
FEATURES = list(sel["features"])
if ADD_DOMAIN_FEATURE and sel.get("domain_feature") not in FEATURES:
    FEATURES.append(sel["domain_feature"])
TRAIN_END = pd.Timestamp(sel.get("train_end", "2025-01-01"))
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Train end: {TRAIN_END.date()}")

# ---------- parse needed lag levels dynamically ----------
lag_nums = set()
for f in FEATURES:
    m = re.match(r"(?:cases|rate)_L(\d+)", f)
    if m: lag_nums.add(int(m.group(1)))
    m = re.match(r"cases_L(\d+)_relative", f)
    if m: lag_nums.add(int(m.group(1)))
if "outbreak_momentum" in FEATURES:
    lag_nums.update([1, 4])
lag_nums = sorted(lag_nums)
print(f"Lag levels needed: {lag_nums}")

# ============================================================
# >>> THE ONLY CHANGED BLOCK: read ONE file, derive split by date <<<
# ============================================================
df = pd.read_csv(SLIM, encoding="utf-8-sig", low_memory=False)
df["province"]   = df["province"].astype(str).str.strip()
df["week_start"] = pd.to_datetime(df["week_start"])

PROVINCE_ORDER = list(dict.fromkeys(df["province"]))   # master order (น่าน first)
print(f"Master order (first 5): {PROVINCE_ORDER[:5]}")
print(f"Rows: {len(df)} | {df['week_start'].min().date()} -> {df['week_start'].max().date()}")

# train/test mask from DATE, not from a _split column
tmask = (df["week_start"] < TRAIN_END).to_numpy()
print(f"Train rows: {tmask.sum()} | Test rows: {(~tmask).sum()}")

# sort by time within province (required for .shift if any col is missing)
df = df.sort_values(["province", "week_start"]).reset_index(drop=True)
tmask = (df["week_start"] < TRAIN_END).to_numpy()      # recompute after sort

# ---------- numeric cleanup ----------
for c in df.columns:
    if isinstance(c, str) and (c.startswith(("w_", "mos_"))
                               or c in ("cos_week", "sin_week", "sin2_week")):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- engineering (no-op if slim already has the columns) ----------
for L in lag_nums:
    col = f"cases_L{L}"
    if col not in df.columns:
        df[col] = df.groupby("province")[TARGET].shift(L)

if "prov_base" not in df.columns:
    base = df.loc[tmask].groupby("province")[TARGET].mean()
    df["prov_base"] = df["province"].map(base).fillna(df.loc[tmask, TARGET].mean())
for L in lag_nums:
    col = f"cases_L{L}_relative"
    if col not in df.columns and (col in FEATURES or ("outbreak_momentum" in FEATURES and L in (1, 4))):
        df[col] = df[f"cases_L{L}"] / (df["prov_base"] + 1.0)
if "outbreak_momentum" in FEATURES and "outbreak_momentum" not in df.columns:
    df["outbreak_momentum"] = df["cases_L1_relative"] / (df["cases_L4_relative"] + 0.1)

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    sys.exit(f"ERROR: features missing after engineering: {missing}")

tr_i = np.where(tmask)[0]; te_i = np.where(~tmask)[0]
Y = df[TARGET].to_numpy(float)
def mat(i): return df.iloc[i][FEATURES].to_numpy(np.float32)

# ---------- rolling-origin folds inside train (Optuna objective) ----------
s = df.iloc[tr_i].sort_values("week_start")
t0, t1 = s["week_start"].min(), s["week_start"].max()
total = (t1 - t0).days; n_val = max(1, int(total * 0.55 / 5)); min_tr = int(total * 0.45)
FOLDS = []
for i in range(5):
    ve = t0 + pd.Timedelta(days=min_tr + n_val * (i + 1)); vs = ve - pd.Timedelta(days=n_val)
    tc = vs - pd.Timedelta(weeks=4)
    a = s.index[s["week_start"] <= tc].to_numpy()
    b = s.index[(s["week_start"] > vs) & (s["week_start"] <= ve)].to_numpy()
    if len(a) > 500 and len(b) > 100: FOLDS.append((a, b))
print(f"CV folds: {len(FOLDS)}")

def cv_dev(p, rounds):
    devs = []
    for a, b in FOLDS:
        d = lgb.Dataset(mat(a), label=Y[a], free_raw_data=False)
        m = lgb.train(p, d, num_boost_round=rounds)
        pr = np.clip(m.predict(mat(b)), 1e-9, None); y = Y[b]
        devs.append(2 * np.mean(np.where(y > 0, y * np.log(y / pr), 0.0) - (y - pr)))
    return float(np.mean(devs))

# ---------- hyperparameters ----------
SAVED = dict(learning_rate=0.0788, num_leaves=105, min_child_samples=67, feature_fraction=0.4475,
             bagging_fraction=0.5303, bagging_freq=2, reg_alpha=0.0392, reg_lambda=0.2535,
             max_depth=3, path_smooth=4.8471, rounds=1327)
if USE_SAVED_PARAMS:
    best = SAVED
else:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(t):
        p = dict(objective="poisson", metric="None", verbose=-1, seed=SEED,
                 num_threads=THREADS, n_jobs=THREADS,
                 learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                 num_leaves=t.suggest_int("num_leaves", 15, 127),
                 min_child_samples=t.suggest_int("min_child_samples", 10, 300),
                 feature_fraction=t.suggest_float("feature_fraction", 0.4, 1.0),
                 bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 1.0),
                 bagging_freq=t.suggest_int("bagging_freq", 1, 10),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 12),
                 path_smooth=t.suggest_float("path_smooth", 1e-3, 100, log=True))
        return cv_dev(p, t.suggest_int("rounds", 300, 1500))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=100, show_progress_bar=True)
    best = dict(st.best_params); print("Optuna best:", best)

params = dict(objective="poisson", verbose=-1, seed=SEED, num_threads=THREADS, n_jobs=THREADS,
              **{k: v for k, v in best.items() if k != "rounds"})
ROUNDS = best.get("rounds", 1300)

# ---------- time-ordered early stopping ----------
tr_times = df.loc[tr_i, "week_start"].values
cut_date = np.quantile(tr_times, 0.9)
tr_in = tr_i[tr_times <  cut_date]
va_in = tr_i[tr_times >= cut_date]
print(f"Early-stop split by date: val starts {pd.Timestamp(cut_date).date()}")

es = dict(params); es["metric"] = "poisson"
m = lgb.train(es, lgb.Dataset(mat(tr_in), label=Y[tr_in], free_raw_data=False),
              num_boost_round=ROUNDS,
              valid_sets=[lgb.Dataset(mat(va_in), label=Y[va_in], free_raw_data=False)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
iters = m.best_iteration or ROUNDS
m = lgb.train({**params, "metric": "None"},
              lgb.Dataset(mat(tr_i), label=Y[tr_i], free_raw_data=False),
              num_boost_round=iters)

pred = np.clip(m.predict(mat(te_i)), 0, None); y = Y[te_i]
ss = np.sum((y - y.mean()) ** 2)
mae = float(np.mean(np.abs(y - pred)))
r2  = float(1 - np.sum((y - pred) ** 2) / ss)
print(f"HOLDOUT  MAE = {mae:.4f}")
print(f"HOLDOUT  R2  = {r2:.4f}")

# ---------- outputs in MASTER province order ----------
out = df.iloc[te_i][["province", "week_start", TARGET]].copy()
out["pred"] = pred
out["residual"] = out[TARGET] - out["pred"]
out["province"] = pd.Categorical(out["province"], categories=PROVINCE_ORDER, ordered=True)
out = (out.sort_values(["province", "week_start"])
          .assign(province=lambda d: d["province"].astype(str))
          .reset_index(drop=True))
out.to_csv(OUT / "final_holdout_predictions.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({"feature": FEATURES, "gain": m.feature_importance("gain")}) \
  .sort_values("gain", ascending=False).to_csv(OUT / "final_importance.csv", index=False)
json.dump({"features": FEATURES, "params": params, "rounds": iters,
           "mae": mae, "r2": r2, "train_end": str(TRAIN_END.date()),
           "input_file": str(SLIM)},
          open(OUT / "final_metrics.json", "w"), indent=2)
m.save_model(str(OUT / "final_model.txt"))

print(f"Saved -> {OUT}")