"""
STEP 4 — ENGINEER FEATURES
============================
Creates >=3 new features from raw columns, then encodes everything into a
model-ready matrix. Encoding is fit ONLY on categories seen in the data and
saved with joblib so the Streamlit app can reuse the exact same encoding at
inference time (no leakage from refitting on new input).
"""
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/rta_clean.csv")

# ---------------------------------------------------------------------------
# New feature 1: time_of_day  (from the raw "Time" HH:MM:SS string)
# ---------------------------------------------------------------------------
# Reason: raw Time has ~1,100 unique values -> useless for a model directly
# and impossible for a dashboard user to "enter" meaningfully. Bucketing
# into 4 human-meaningful periods keeps the signal (rush hour vs night)
# while making it usable both for the model and for a simple UI dropdown.
def bucket_time(t):
    try:
        hour = int(str(t).split(":")[0])
    except (ValueError, IndexError):
        return "Unknown"
    if 5 <= hour < 10:
        return "Morning Rush (5-10)"
    elif 10 <= hour < 16:
        return "Midday (10-16)"
    elif 16 <= hour < 21:
        return "Evening Rush (16-21)"
    else:
        return "Night (21-5)"

df["time_of_day"] = df["Time"].apply(bucket_time)
df = df.drop(columns=["Time"])

# ---------------------------------------------------------------------------
# New feature 2: is_weekend  (from Day_of_week)
# ---------------------------------------------------------------------------
# Reason: weekday/weekend driving patterns (commute vs leisure/alcohol-related
# trips) are a well-known accident-severity factor, and this is a much lower-
# cardinality signal than the 7-value raw day for a small dataset.
df["is_weekend"] = df["Day_of_week"].isin(["Friday", "Saturday", "Sunday"]).astype(int)

# ---------------------------------------------------------------------------
# New feature 3: junction_risk  (combines Types_of_Junction + Road_allignment)
# ---------------------------------------------------------------------------
# Reason: neither column alone captures it well, but junctions on a bend/slope
# ("Road_allignment" mentioning grade/curve) are intuitively higher risk than
# a straight tangent road, whether or not there's a junction. This combined
# flag surfaces that compound condition explicitly for the model.
risky_alignment = df["Road_allignment"].astype(str).str.contains(
    "curve|slope|grade|hill|steep", case=False, na=False
)
has_junction = df["Types_of_Junction"] != "No junction"
df["junction_risk"] = (risky_alignment | has_junction).astype(int)

# ---------------------------------------------------------------------------
# New feature 4: is_vulnerable_road_user_involved
# ---------------------------------------------------------------------------
# Reason: EDA showed pedestrian-involved collisions have ~2x the fatal share
# of vehicle-vehicle collisions. Collapsing pedestrian-related signals from
# two columns (Type_of_collision, Pedestrian_movement) into a single binary
# flag makes that risk explicit and easy for a dashboard user to toggle.
df["is_vulnerable_road_user_involved"] = (
    df["Type_of_collision"].str.contains("pedestrian", case=False, na=False)
    | (df["Pedestrian_movement"] != "Not a Pedestrian")
).astype(int)

print("New engineered features: time_of_day, is_weekend, junction_risk, "
      "is_vulnerable_road_user_involved")

# ---------------------------------------------------------------------------
# Drop columns that became redundant after engineering, or that add
# near-zero signal (very high cardinality, mostly "Unknown", or fully
# summarized by an engineered feature).
# ---------------------------------------------------------------------------
drop_after_fe = ["Day_of_week"]  # info now captured by is_weekend
df = df.drop(columns=drop_after_fe)
print(f"Dropped (redundant after engineering): {drop_after_fe}")

# ---------------------------------------------------------------------------
# Encode target
# ---------------------------------------------------------------------------
target_order = ["Fatal injury", "Serious Injury", "Slight Injury"]  # 0,1,2
target_map = {name: i for i, name in enumerate(target_order)}
df["target"] = df["Accident_severity"].map(target_map)
df = df.drop(columns=["Accident_severity"])

# ---------------------------------------------------------------------------
# Encode categorical features (One-Hot). Numbers here are just the two
# binary flags (already 0/1), everything else is categorical text.
# ---------------------------------------------------------------------------
cat_cols = [c for c in df.columns if df[c].dtype.name in ("object", "str")]
num_cols = [c for c in df.columns if c not in cat_cols + ["target"]]
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric/binary columns ({len(num_cols)}): {num_cols}")

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

feature_cols = [c for c in df_encoded.columns if c != "target"]
print(f"\nFinal feature matrix: {df_encoded.shape[0]} rows x {len(feature_cols)} features")

df_encoded.to_csv("data/rta_features.csv", index=False)

# Save the raw (pre-one-hot) cleaned+engineered frame too -- the Streamlit
# app's "Data Explorer" tab reads human-readable categories from this, and
# the "Live Prediction" tab uses it to know each dropdown's valid options.
df.to_csv("data/rta_readable.csv", index=False)

joblib.dump(
    {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "target_map": target_map,
        "target_order": target_order,
    },
    "models/feature_meta.joblib",
)

print("\nSaved -> data/rta_features.csv, data/rta_readable.csv, models/feature_meta.joblib")
