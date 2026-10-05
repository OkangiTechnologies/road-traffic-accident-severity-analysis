"""
STEP 6 — EVALUATE THE BEST MODEL
===================================
Touches the test set exactly once, here. Reports macro-F1 (fits an
imbalanced multi-class problem far better than accuracy) plus a confusion
matrix, and gives a direct answer to "is it good enough?".
"""
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

df = pd.read_csv("data/rta_features.csv")
X = df.drop(columns=["target"])
y = df["target"]

test_idx = np.load("models/test_idx.npy")
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")
uses_scaled = joblib.load("models/uses_scaled.joblib")

Xt = scaler.transform(X_test) if uses_scaled else X_test
pred = model.predict(Xt)

target_names = ["Fatal", "Serious", "Slight"]  # order 0,1,2

acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")

print("=== TEST SET performance (touched once) ===")
print(f"Accuracy   : {acc:.4f}")
print(f"Macro-F1   : {macro_f1:.4f}   <- primary metric (imbalanced classes)")
print()
print(classification_report(y_test, pred, target_names=target_names, digits=3))

report_dict = classification_report(
    y_test, pred, target_names=target_names, digits=3, output_dict=True
)
with open("models/test_report.json", "w") as f:
    json.dump({"accuracy": acc, "macro_f1": macro_f1, "report": report_dict}, f, indent=2)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(target_names)
ax.set_yticks(range(3)); ax.set_yticklabels(target_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix (row-normalized)\nTest macro-F1 = {macro_f1:.3f}")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})", ha="center", va="center",
                 color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("figures/07_confusion_matrix.png")
plt.close()

np.save("models/cm.npy", cm)
print("\nSaved figures/07_confusion_matrix.png, models/test_report.json")
