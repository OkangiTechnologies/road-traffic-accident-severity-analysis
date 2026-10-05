"""
STEP 2 — CLEAN THE DATA
========================
Dataset: Road Traffic Accident (RTA) severity — Addis Ababa, Ethiopia
Source : https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents
Rows   : 12,316 | Raw columns: 32

Problem type : Multi-class CLASSIFICATION
Target       : Accident_severity  ->  {Fatal injury, Serious Injury, Slight Injury}

This script documents every cleaning decision with a comment explaining *why*,
per the task's requirement for a "short summary of what you found and fixed."
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/RTA_Dataset.csv"
OUT_PATH = "data/rta_clean.csv"

df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2a. DUPLICATES
# ---------------------------------------------------------------------------
n_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dupes}")   # 0 in the raw 32-col file
df = df.drop_duplicates()

# ---------------------------------------------------------------------------
# 2b. LEAKAGE COLUMNS  <-- most important cleaning decision
# ---------------------------------------------------------------------------
# These columns describe facts that are only known AFTER the crash has
# happened and its severity is already determined (casualty outcomes,
# how many people/vehicles ended up involved, etc). A real dashboard user
# would fill in the form BEFORE a crash to get a risk prediction, so none
# of this information is legitimately available at prediction time.
# Keeping them would let the model "cheat" (e.g. Casualty_severity is
# almost the same information as Accident_severity itself).
leakage_cols = [
    "Number_of_casualties",       # outcome count
    "Number_of_vehicles_involved",# only fully known post-crash
    "Casualty_class",             # describes the casualty AFTER the crash
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",          # near-duplicate of the target
    "Work_of_casuality",
    "Fitness_of_casuality",
]
df = df.drop(columns=leakage_cols)
print(f"Dropped {len(leakage_cols)} leakage columns.")

# ---------------------------------------------------------------------------
# 2c. IDENTIFIER / TOO-GRANULAR COLUMNS
# ---------------------------------------------------------------------------
# 'Time' is a free-text HH:MM:SS with ~1100 unique values -- not usable raw.
# We keep it but engineer a coarse 'time_of_day' bucket in the feature step,
# then drop the raw string later. Nothing dropped here yet.

# ---------------------------------------------------------------------------
# 2d. MISSING VALUES
# ---------------------------------------------------------------------------
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing values per column:")
print(missing)

# Columns that are majority-missing (>30%) carry too little signal to impute
# reliably and risk introducing a fake pattern if we do. Drop them.
mostly_missing = missing[missing / len(df) > 0.30].index.tolist()
print(f"\nDropping (>30% missing): {mostly_missing}")
df = df.drop(columns=mostly_missing)

# For the remaining categorical columns with a modest amount of missing data,
# missingness is itself informative in this dataset (e.g. a blank
# "Educational_level" often means "unknown/not recorded" on the police
# report) so instead of imputing a mode (which would invent facts), we
# encode missing as an explicit category "Unknown". This preserves the
# signal that the field was left blank.
remaining_missing_cols = df.columns[df.isna().any()].tolist()
print(f"Filling remaining missing categoricals with 'Unknown': {remaining_missing_cols}")
for c in remaining_missing_cols:
    df[c] = df[c].fillna("Unknown")

# ---------------------------------------------------------------------------
# 2e. ODD / MESSY ENTRIES
# ---------------------------------------------------------------------------
# Strip stray whitespace picked up from the raw CSV (e.g. "  Recreational areas")
obj_cols = df.select_dtypes(include="object").columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()

# Vehicle type strings contain garbled encoding artifacts, e.g.
# "Lorry (41?100Q)" should read "Lorry (41-100Q)". Fix known mojibake.
if "Type_of_vehicle" in df.columns:
    df["Type_of_vehicle"] = df["Type_of_vehicle"].str.replace("?", "-", regex=False)

# Some columns already contain a genuine lowercase "unknown" category recorded
# by the police (distinct from blank/NaN), which is now colliding with our
# filled-in "Unknown" for missing cells. Normalize casing so they merge into
# one consistent "Unknown" category instead of silently splitting the signal
# in two ("unknown" vs "Unknown" showing up as different classes).
for c in obj_cols:
    if c in df.columns:
        df[c] = df[c].replace({v: "Unknown" for v in df[c].unique() if str(v).lower() == "unknown"})

# Pedestrian_movement has a corrupted find/replace artifact from the original
# source file: the substring "stationary" was mangled into
# "statioNot a Pedestrianry" (someone's broken search/replace of "station...").
# Restore the intended word so the category is readable.
if "Pedestrian_movement" in df.columns:
    df["Pedestrian_movement"] = df["Pedestrian_movement"].str.replace(
        "statioNot a Pedestrianry", "stationary", regex=False
    )

print(f"\nFinal cleaned shape: {df.shape}")
print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
