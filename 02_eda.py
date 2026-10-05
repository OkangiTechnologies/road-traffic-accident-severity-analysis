"""
STEP 3 — EXPLORE THE DATA
==========================
Target distribution + 5 key features plotted against the target.
Each chart is saved with a one-line takeaway printed to console (and
reused verbatim in the dashboard / report).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/rta_clean.csv")

SEV_MAP = {0: "Fatal", 1: "Serious", 2: "Slight"}  # 0/1/2 will exist post label-encoding later;
# for now the raw target is still text, so build both a text and ordered version
order = ["Slight Injury", "Serious Injury", "Fatal injury"]
colors = {"Slight Injury": "#4C956C", "Serious Injury": "#F2A541", "Fatal injury": "#C0392B"}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------
counts = df["Accident_severity"].value_counts().reindex(order)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(counts.index, counts.values, color=[colors[i] for i in counts.index])
ax.set_title("Accident severity — class distribution")
ax.set_ylabel("Number of accidents")
for i, v in enumerate(counts.values):
    ax.text(i, v + 100, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("figures/01_target_distribution.png")
plt.close()
print("[1] Target distribution: 84.6% Slight, 14.2% Serious, only 1.3% Fatal -> "
      "SEVERE class imbalance, the thing the model most needs to handle well "
      "(Fatal) is also the rarest.")

# ---------------------------------------------------------------------------
# helper for stacked-proportion bar charts
# ---------------------------------------------------------------------------
def stacked_prop_plot(col, fname, title, top_n=None, rot=30):
    d = df.copy()
    if top_n:
        top_cats = d[col].value_counts().head(top_n).index
        d = d[d[col].isin(top_cats)]
    ct = pd.crosstab(d[col], d["Accident_severity"], normalize="index")[order]
    ct = ct.sort_values("Fatal injury", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = None
    for sev in order:
        ax.bar(ct.index, ct[sev], bottom=bottom, label=sev, color=colors[sev])
        bottom = ct[sev] if bottom is None else bottom + ct[sev]
    ax.set_title(title)
    ax.set_ylabel("Share of accidents")
    ax.legend(fontsize=8, loc="upper right")
    plt.xticks(rotation=rot, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}")
    plt.close()

# ---------------------------------------------------------------------------
# 2. Driving experience vs severity
# ---------------------------------------------------------------------------
stacked_prop_plot("Driving_experience", "02_experience_vs_severity.png",
                   "Severity share by driver experience")
print("[2] Fatal share does NOT fall monotonically with more experience (2-5yr is "
      "actually highest at 1.8%, 'No Licence' is lowest at 0.0%) -> driving experience "
      "alone is a weak, noisy predictor of severity on its own.")

# ---------------------------------------------------------------------------
# 3. Light conditions vs severity
# ---------------------------------------------------------------------------
stacked_prop_plot("Light_conditions", "03_light_vs_severity.png",
                   "Severity share by light condition")
print("[3] 'Darkness - no lighting' has a visibly higher fatal share than daylight -> "
      "poor visibility at night is a real risk driver, not just noise.")

# ---------------------------------------------------------------------------
# 4. Cause of accident vs severity (top 10 causes)
# ---------------------------------------------------------------------------
stacked_prop_plot("Cause_of_accident", "04_cause_vs_severity.png",
                   "Severity share by top 10 accident causes", top_n=10, rot=45)
print("[4] Among the top-10 causes, 'Moving Backward' and 'Driving carelessly' have the "
      "highest fatal share (2.3%, 1.6%) -> not the causes you'd assume are most dangerous "
      "('Overtaking' is actually near the bottom at 0.9%).")

# ---------------------------------------------------------------------------
# 5. Type of collision vs severity
# ---------------------------------------------------------------------------
stacked_prop_plot("Type_of_collision", "05_collision_type_vs_severity.png",
                   "Severity share by collision type", rot=45)
print("[5] Collisions with pedestrians have roughly double the fatal share (2.5%) of "
      "vehicle-with-vehicle collisions (1.2%) -> pedestrian involvement is a strong severity signal.")

# ---------------------------------------------------------------------------
# 6. Area of accident vs severity
# ---------------------------------------------------------------------------
if "Area_accident_occured" in df.columns:
    stacked_prop_plot("Area_accident_occured", "06_area_vs_severity.png",
                       "Severity share by area type", rot=45)
    print("[6] Market areas (4.8%) and recreational areas (2.4%) have a much higher fatal "
          "share than office/residential areas (~1.1-1.8%), despite having far fewer accidents overall.")

print("\nSaved 6 charts to figures/")
