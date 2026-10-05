# Road Traffic Accident Severity — Project Report

## 1. Problem definition & data

**Question:** Given the conditions surrounding a road accident (driver profile,
road type, junction, lighting, weather, and cause), how severe is it likely to
be — **Slight**, **Serious**, or **Fatal** injury?

**Dataset:** [Road Traffic Accidents — Kaggle](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents)
(12,316 real police-recorded accidents, Addis Ababa, Ethiopia, 2017–2020; 32
raw columns).

**Problem type:** Multi-class **classification** (3 classes, target =
`Accident_severity`).

**Raw column list (32):** `Time, Day_of_week, Age_band_of_driver,
Sex_of_driver, Educational_level, Vehicle_driver_relation,
Driving_experience, Type_of_vehicle, Owner_of_vehicle,
Service_year_of_vehicle, Defect_of_vehicle, Area_accident_occured,
Lanes_or_Medians, Road_allignment, Types_of_Junction, Road_surface_type,
Road_surface_conditions, Light_conditions, Weather_conditions,
Type_of_collision, Number_of_vehicles_involved, Number_of_casualties,
Vehicle_movement, Casualty_class, Sex_of_casualty, Age_band_of_casualty,
Casualty_severity, Work_of_casuality, Fitness_of_casuality,
Pedestrian_movement, Cause_of_accident, Accident_severity`

---

## 2. Cleaning summary (what was found & fixed)

| Finding | Fix | Why |
|---|---|---|
| 0 exact duplicate rows | none needed | — |
| **8 leakage columns** (`Number_of_casualties`, `Number_of_vehicles_involved`, `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Casualty_severity`, `Work_of_casuality`, `Fitness_of_casuality`) | dropped | These describe facts only known *after* the crash / casualty outcome — `Casualty_severity` is nearly a copy of the target. Keeping them would let the model "cheat." |
| `Defect_of_vehicle` (36% missing), `Service_year_of_vehicle` (32% missing) | dropped | Too sparse to impute reliably without inventing a pattern. |
| 12 remaining columns with 1–7% missing | filled with `"Unknown"` | Missingness looked informative here (a blank field on a police report), not random — an explicit "Unknown" category preserves that signal instead of a fabricated mode/mean. |
| Inconsistent casing (`"unknown"` vs filled `"Unknown"` as two different categories) | normalized to one `"Unknown"` | Otherwise the same real-world value was silently split across two categories, diluting its signal. |
| Corrupted text artifact in `Pedestrian_movement` (`"statioNot a Pedestrianry"` instead of `"stationary"` — a broken find/replace in the source file) | restored to `"stationary"` | Made the category human-readable and correctly grouped. |
| Stray leading/trailing whitespace (e.g. `"  Recreational areas"`) | stripped | Prevented duplicate categories that differ only by whitespace. |
| Mojibake in `Type_of_vehicle` (e.g. `"Lorry (41?100Q)"`) | `?` → `-` | Restored the intended range notation. |

Final cleaned shape: **12,316 rows × 22 columns** (21 features + target).

---

## 3. Exploration — key takeaways

1. **Class distribution:** 84.6% Slight, 14.2% Serious, only 1.3% Fatal — severe
   imbalance, and the rarest class is the one that matters most.
2. **Driving experience** does *not* predict severity monotonically (2–5yr
   drivers have the highest fatal share at 1.8%; "No Licence" is lowest at
   0.0%) — a weak, noisy signal alone.
3. **Darkness with no lighting** has a clearly higher fatal share than
   daylight or lit darkness — visibility is a real risk driver.
4. Among the top-10 accident causes, **"Moving Backward"** and **"Driving
   carelessly"** carry the highest fatal share (2.3%, 1.6%) — not
   "Overtaking," which is actually near the bottom (0.9%).
5. **Pedestrian collisions** carry roughly double the fatal share (2.5%) of
   vehicle-vehicle collisions (1.2%).
6. **Market areas** (4.8%) and **recreational areas** (2.4%) have far higher
   fatal shares than office/residential areas (~1–2%), despite having fewer
   accidents overall.

Charts: `figures/01_target_distribution.png` through `06_area_vs_severity.png`.

---

## 4. Feature engineering

4 new features created from raw columns:

| Feature | Built from | Reason |
|---|---|---|
| `time_of_day` | `Time` (HH:MM:SS, ~1,100 unique values) | Bucketed into 4 human-meaningful periods (Morning Rush, Midday, Evening Rush, Night) — usable by both the model and a dashboard dropdown. |
| `is_weekend` | `Day_of_week` | Collapses 7 values into the commute-vs-leisure signal that matters most, and removes a higher-cardinality column. |
| `junction_risk` | `Types_of_Junction` + `Road_allignment` | Neither column alone captures "compound" risk; flags junctions or curved/graded roads together. |
| `is_vulnerable_road_user_involved` | `Type_of_collision` + `Pedestrian_movement` | EDA showed pedestrian involvement roughly doubles fatal share — made explicit as one binary signal. |

`Day_of_week` was dropped afterward (fully captured by `is_weekend`).
Categorical columns (20) were one-hot encoded; the two engineered flags plus
`is_weekend` stayed binary numeric. Final feature matrix: **12,316 rows ×
169 columns**.

---

## 5. Model comparison (validation set)

| Model | Val accuracy | Val macro-F1 | Val Fatal recall |
|---|---|---|---|
| Random Forest | 0.836 | **0.328** | 0.042 |
| Gradient Boosting | 0.839 | 0.323 | 0.000 |
| Logistic Regression | 0.453 | 0.301 | 0.542 |

**Macro-F1** (not accuracy) is the comparison metric — with 84.6% of
accidents "Slight," always predicting "Slight" already scores 84.6% accuracy
while being useless. Random Forest won on macro-F1 and was tuned with a
3-fold grid search (`n_estimators`, `max_depth`, `min_samples_leaf`),
improving macro-F1 from 0.328 → **0.355** on validation.

Interesting trade-off: Logistic Regression has by far the best *Fatal
recall* (0.54 vs ~0.04) because `class_weight="balanced"` pushes it to guess
rare classes far more often — at the cost of tanking overall accuracy and
Serious-class precision. This is a real design choice a deployed system
would need to make explicitly (see Reflection).

## 6. Final evaluation (test set, touched once)

**Model:** Random Forest, tuned (`max_depth=10, min_samples_leaf=3,
n_estimators=400`)

| Metric | Value |
|---|---|
| Test accuracy | 0.811 |
| **Test macro-F1** | **0.367** |
| Fatal recall | 0.087 |
| Fatal precision | 0.083 |
| Slight recall | 0.944 |

Confusion matrix: `figures/07_confusion_matrix.png`

**Is it good enough?** Not for real deployment as-is. It clearly beats a
naive "always Slight" baseline on macro-F1, and it's genuinely useful for
exploring which conditions correlate with severity — but it catches only
~9% of truly Fatal accidents. With just 158 Fatal examples in the whole
12,316-row dataset, there isn't enough signal for any model trained this way
to reliably flag the rarest, highest-stakes class. It's a solid working
baseline and a complete, honest pipeline — not a system to act on alone.

---

## 7. Streamlit dashboard

`app_bundle/app.py` — run with `streamlit run app.py` from inside
`app_bundle/`. Four tabs: **Overview**, **Data Explorer** (filterable charts
over the cleaned data), **Model Performance** (comparison table, tuned test
metrics, confusion matrix, honest "is it good enough" verdict), and **Live
Prediction** (a form matching every raw feature the model uses; predicts
from the saved `best_model.joblib` — no retraining happens in the app).

---

## 8. Reflection

**What could break this model in the real world?** The training data comes
from one city (Addis Ababa) over a few years, so it would likely generalize
poorly to a different country's road layout, vehicle mix, or enforcement
culture. It also relies on categories a police officer fills in *after*
arriving at the scene (e.g. weather, light, road surface) — for a true
pre-crash early-warning use case, some of those may not be knowable in
advance either. And because Fatal accidents are only 1.3% of the data, the
model's read on that class is built on very few examples and would be
fragile to any shift in what "fatal" conditions look like over time (new
road infrastructure, speed limit changes, vehicle safety improvements).

**How would you know if it needs retraining?** Track the model's macro-F1
and per-class recall on a rolling window of new accident records with known
outcomes, and watch for drift in the *input* distribution too (e.g. a new
road type or vehicle category appearing far more often than in training).
A sustained drop in Fatal/Serious recall, a rise in predictions the traffic
authority's own reviewers disagree with, or a visible shift in feature
distributions (new junction types, changed weather patterns) would all be
signals to retrain — ideally on a regular cadence (e.g. yearly) rather than
waiting for a visible failure, given how few Fatal examples exist to
detect drift from in the first place.
