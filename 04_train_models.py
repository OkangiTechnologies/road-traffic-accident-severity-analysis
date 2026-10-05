"""
STEP 5 — TRAIN AND COMPARE MODELS
====================================
Split -> train 3 models -> tune the best one -> save comparison table.
Uses macro-F1 as the primary comparison metric since classes are heavily
imbalanced (accuracy would be misleadingly high just by predicting
"Slight Injury" every time -> 84.6% "accuracy" for a useless model).
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/rta_features.csv")
X = df.drop(columns=["target"])
y = df["target"]
feature_cols = X.columns.tolist()

# ---------------------------------------------------------------------------
# Split: 70% train / 15% validation / 15% test, stratified so every split
# keeps the same (very skewed) class proportions.
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scale for the linear model (tree models don't need it, but a shared
# scaled copy keeps the code simple and doesn't hurt trees).
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

results = []

# ---------------------------------------------------------------------------
# Model 1: Logistic Regression (simple, interpretable baseline)
# ---------------------------------------------------------------------------
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_s, y_train)
pred = lr.predict(X_val_s)
results.append({
    "model": "Logistic Regression",
    "val_accuracy": accuracy_score(y_val, pred),
    "val_macro_f1": f1_score(y_val, pred, average="macro"),
    "val_recall_fatal": recall_score(y_val, pred, labels=[0], average="macro"),
})

# ---------------------------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
results.append({
    "model": "Random Forest",
    "val_accuracy": accuracy_score(y_val, pred),
    "val_macro_f1": f1_score(y_val, pred, average="macro"),
    "val_recall_fatal": recall_score(y_val, pred, labels=[0], average="macro"),
})

# ---------------------------------------------------------------------------
# Model 3: Gradient Boosting
# ---------------------------------------------------------------------------
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
)
# GradientBoostingClassifier has no class_weight param -> pass sample_weight
sw = y_train.map({0: 8.0, 1: 2.0, 2: 1.0}).values  # inverse-ish of class freq
gb.fit(X_train, y_train, sample_weight=sw)
pred = gb.predict(X_val)
results.append({
    "model": "Gradient Boosting",
    "val_accuracy": accuracy_score(y_val, pred),
    "val_macro_f1": f1_score(y_val, pred, average="macro"),
    "val_recall_fatal": recall_score(y_val, pred, labels=[0], average="macro"),
})

results_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
print("\n=== Model comparison (validation set) ===")
print(results_df.to_string(index=False))
results_df.to_csv("models/model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Tune the best-performing model (by val macro-F1) with a small grid search
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_name} -> tuning with GridSearchCV (3-fold, macro-F1)")

if best_name == "Random Forest":
    base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [10, 16, None],
        "min_samples_leaf": [1, 3],
    }
    gs = GridSearchCV(base, grid, scoring="f1_macro", cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    uses_scaled = False
elif best_name == "Gradient Boosting":
    base = GradientBoostingClassifier(random_state=42)
    grid = {
        "n_estimators": [150, 250],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
    }
    gs = GridSearchCV(base, grid, scoring="f1_macro", cv=3, n_jobs=-1)
    gs.fit(X_train, y_train, sample_weight=sw)
    best_model = gs.best_estimator_
    uses_scaled = False
else:
    base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    grid = {"C": [0.01, 0.1, 1, 10]}
    gs = GridSearchCV(base, grid, scoring="f1_macro", cv=3, n_jobs=-1)
    gs.fit(X_train_s, y_train)
    best_model = gs.best_estimator_
    uses_scaled = True

print(f"Best params: {gs.best_params_}")

# Validate tuned model
Xv = X_val_s if uses_scaled else X_val
pred_tuned = best_model.predict(Xv)
tuned_val_f1 = f1_score(y_val, pred_tuned, average="macro")
print(f"Tuned {best_name} val macro-F1: {tuned_val_f1:.4f} "
      f"(vs {results_df.iloc[0]['val_macro_f1']:.4f} untuned)")

# ---------------------------------------------------------------------------
# Persist everything the app / evaluation step needs
# ---------------------------------------------------------------------------
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(feature_cols, "models/feature_cols.joblib")
joblib.dump(uses_scaled, "models/uses_scaled.joblib")

with open("models/best_model_name.json", "w") as f:
    json.dump({"name": best_name, "params": gs.best_params_}, f, indent=2)

# save split indices so evaluation step uses the EXACT same untouched test set
np.save("models/test_idx.npy", X_test.index.to_numpy())

print("\nSaved best_model.joblib, scaler.joblib, feature_cols.joblib, "
      "uses_scaled.joblib, best_model_name.json, test_idx.npy")
