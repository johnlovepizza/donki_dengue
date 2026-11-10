"""STEP 2 — COLUMN strip ONLY. No train/test row split here.
Reads features dynamically from features_pick/selected_features.json.
Output: split_data/dengue_slime.csv  (ALL rows, used columns only, master order, Nov-2026 dropped)
The train/test row split is done downstream in 03_train_optuna.py via TRAIN_END."""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
CSV  = BASE / "dengue_with_population.csv"
PICK = BASE / "features_pick" / "selected_features.json"
OUT  = BASE / "split_data"
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    sys.exit(f"ERROR: put dengue_with_population.csv next to this script.\nMissing: {CSV}")
if not PICK.exists():
    sys.exit(f"ERROR: run 01_pick_features.py first.\nMissing: {PICK}")

# ============================================================
# READ FEATURE LIST FROM PICKER (dynamic)
# ============================================================
sel = json.load(open(PICK))
FEATURES = list(sel["features"])
DOMAIN   = sel.get("domain_feature")
if DOMAIN and DOMAIN not in FEATURES:
    FEATURES.append(DOMAIN)
TRAIN_END = pd.Timestamp(sel.get("train_end", "2025-01-01"))   # used ONLY for baseline calc
EVAL_CUTOFF = pd.Timestamp("2026-11-01")                       # ALWAYS drop incomplete Nov-2026
TARGET = "cases"

print(f"Features from picker ({len(FEATURES)}): {FEATURES}")
print(f"Baseline train_end: {TRAIN_END.date()} | Eval cutoff (drop >=): {EVAL_CUTOFF.date()}")

# ============================================================
# PARSE FEATURE NAMES -> what to engineer
# ============================================================
lag_nums = set()
for f in FEATURES:
    m = re.match(r"(?:cases|rate)_L(\d+)", f)
    if m: lag_nums.add(int(m.group(1)))
    m = re.match(r"cases_L(\d+)_relative", f)
    if m: lag_nums.add(int(m.group(1)))
needs_momentum = "outbreak_momentum" in FEATURES
if needs_momentum:
    lag_nums.update([1, 4])
lag_nums = sorted(lag_nums)
print(f"Lag levels needed: {lag_nums}")

RAW_COLS, CAT_COLS, DERIVED = [], [], []
for f in FEATURES:
    if f in ("province", "zone"):                       CAT_COLS.append(f)
    elif f.startswith("cases_L") and "_relative" not in f: DERIVED.append(f)
    elif f.startswith("rate_L"):                        DERIVED.append(f)
    elif f.endswith("_relative"):                       DERIVED.append(f)
    elif f == "outbreak_momentum":                      DERIVED.append(f)
    elif f == "log_population":                         DERIVED.append(f)
    else:                                               RAW_COLS.append(f)
print(f"Raw (in CSV): {RAW_COLS}")
print(f"Categorical:  {CAT_COLS}")
print(f"Derived:      {DERIVED}")

# ============================================================
# LOAD + LOCK MASTER ORDER (before any sort/drop)
# ============================================================
df = pd.read_csv(CSV, low_memory=False)
df["_master_row"] = np.arange(len(df))
df["province"]    = df["province"].astype(str).str.strip()
df["week_start"]  = pd.to_datetime(df["week_start"])
df["year"]        = df["week_start"].dt.year
df["week"]        = df["week_start"].dt.isocalendar().week.astype(int)

PROVINCE_ORDER = list(dict.fromkeys(df["province"]))
print(f"Master order (first 5): {PROVINCE_ORDER[:5]}")

# >>> UNCONDITIONAL DROP of incomplete November 2026 <<<
n_before = len(df)
df = df[df["week_start"] < EVAL_CUTOFF].copy()
n_dropped = n_before - len(df)
print(f"Dropped {n_dropped} incomplete rows with week_start >= {EVAL_CUTOFF.date()} (now {len(df)} rows)")

# Sort by time within province (required for .shift)
df = df.sort_values(["province", "week_start"]).reset_index(drop=True)

# ============================================================
# NUMERIC CONVERSION
# ============================================================
for c in df.columns:
    if isinstance(c, str) and (c.startswith("w_") or c.startswith("mos_")
                               or c in ("cos_week", "sin_week", "sin2_week")):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# ENGINEER: case lags
# ============================================================
for L in lag_nums:
    col = f"cases_L{L}"
    if col not in df.columns:
        df[col] = df.groupby("province")[TARGET].shift(L)

# ============================================================
# ENGINEER: rate lags
# ============================================================
if any(f.startswith("rate_L") for f in FEATURES):
    if "cases_per_100k" not in df.columns:
        sys.exit("ERROR: rate_L features requested but cases_per_100k not in CSV")
    for L in lag_nums:
        col = f"rate_L{L}"
        if col not in df.columns:
            df[col] = df.groupby("province")["cases_per_100k"].shift(L)

# ============================================================
# ENGINEER: train-only baseline + relative lags
# (baseline uses < TRAIN_END only; this is the ONLY place TRAIN_END is used here)
# ============================================================
train_mask = (df["week_start"] < TRAIN_END).to_numpy()
base = df.loc[train_mask].groupby("province")[TARGET].mean()
df["prov_base"] = df["province"].map(base).fillna(df.loc[train_mask, TARGET].mean())
for L in lag_nums:
    col = f"cases_L{L}_relative"
    if col in FEATURES or (needs_momentum and L in (1, 4)):
        df[col] = df[f"cases_L{L}"] / (df["prov_base"] + 1.0)

# ============================================================
# ENGINEER: outbreak_momentum
# ============================================================
if needs_momentum:
    df["outbreak_momentum"] = df["cases_L1_relative"] / (df["cases_L4_relative"] + 0.1)

# ============================================================
# ENGINEER: log_population
# ============================================================
if "log_population" in FEATURES:
    if "population" not in df.columns:
        sys.exit("ERROR: log_population requested but population not in CSV")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["log_population"] = np.log1p(df["population"])

# ============================================================
# CATEGORICAL HANDLING
# ============================================================
for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("missing").astype("category")

# ============================================================
# ASSEMBLE OUTPUT COLUMNS
# ============================================================
KEYS   = ["province", "week_start", "year", "week"]
OUTPUT = KEYS + [TARGET] + FEATURES
missing = [c for c in OUTPUT if c not in df.columns]
if missing:
    sys.exit(f"ERROR: columns missing after engineering: {missing}")

# LEAKAGE GUARD
assert "cases_per_100k" not in OUTPUT, "NEVER ship the answer column!"
assert TARGET not in FEATURES, "target must not appear as a feature!"

# ============================================================
# RESTORE MASTER ORDER
# ============================================================
slim = df[OUTPUT + ["_master_row"]].copy()
slim = slim.sort_values("_master_row").drop(columns=["_master_row"]).reset_index(drop=True)
for c in CAT_COLS:
    slim[c] = slim[c].astype(str)

# sanity: no Nov-2026 row survived
assert slim["week_start"].max() < EVAL_CUTOFF, "Nov-2026 leaked into slim output!"

# ============================================================
# SAVE (ONE file, ALL rows, NO train/test split)
# ============================================================
slim.to_csv(OUT / "dengue_slime.csv", index=False, encoding="utf-8-sig")
json.dump({"train_end_for_baseline": str(TRAIN_END.date()),
           "eval_cutoff_drop_ge": str(EVAL_CUTOFF.date()),
           "rows_dropped_incomplete": int(n_dropped),
           "features": FEATURES,
           "n_features": len(FEATURES),
           "columns": OUTPUT,
           "n_columns": len(OUTPUT),
           "lag_levels": lag_nums,
           "date_min": str(slim["week_start"].min().date()),
           "date_max": str(slim["week_start"].max().date()),
           "n_rows": int(len(slim)),
           "province_order_first5": PROVINCE_ORDER[:5],
           "note": "column-strip only; train/test row split done in 03 via TRAIN_END"},
          open(OUT / "slim_config.json", "w"), indent=2, ensure_ascii=False)

print(f"\nSaved -> {OUT / 'dengue_slime.csv'}")
print(f"dengue_slime.csv: {len(slim)} rows x {len(OUTPUT)} cols  "
      f"{slim['week_start'].min().date()} -> {slim['week_start'].max().date()}")
print(f"First 3 provinces: {slim['province'].head(3).tolist()}")
print(f"\nColumns ({len(OUTPUT)}): {OUTPUT}")
print("\n✓ Column strip only — NO train/test split. All rows kept, master order, Nov-2026 dropped.")