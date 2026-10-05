"""
STEP 7 — STREAMLIT DASHBOARD
==============================
Run with:  streamlit run app.py

Tabs:
  Overview          - what the project is about
  Data Explorer      - filters/charts over the cleaned data
  Model Performance  - metrics + confusion matrix + model comparison
  Live Prediction    - user fills a form -> saved model returns a prediction

The model is loaded once from disk (joblib) and NEVER retrained here.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Road Traffic Accident Severity",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load everything once, cached
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    joblib.dump(best_model, "models/best_model.joblib")
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    uses_scaled = joblib.load("models/uses_scaled.joblib")
    meta = joblib.load("models/feature_meta.joblib")
    with open("models/best_model_name.json") as f:
        model_info = json.load(f)
    with open("models/test_report.json") as f:
        test_report = json.load(f)
    comparison = pd.read_csv("models/model_comparison.csv")
    return model, scaler, feature_cols, uses_scaled, meta, model_info, test_report, comparison


@st.cache_data
def load_data():
    df = pd.read_csv("data/rta_readable.csv")
    sev_names = {0: "Fatal", 1: "Serious", 2: "Slight"}
    df["Accident_severity"] = df["target"].map(sev_names)
    return df


(model, scaler, feature_cols, uses_scaled, meta,
 model_info, test_report, comparison) = load_model_artifacts()
df = load_data()

SEV_COLORS = {"Fatal": "#C0392B", "Serious": "#F2A541", "Slight": "#4C956C"}
SEV_ORDER = ["Fatal", "Serious", "Slight"]

st.title("🚦 Road Traffic Accident Severity")
st.caption("Addis Ababa RTA dataset — 12,316 accidents · predicting whether a crash "
           "will be Slight, Serious, or Fatal, from pre-crash conditions only.")

tab_overview, tab_explore, tab_perf, tab_predict = st.tabs(
    ["📋 Overview", "🔎 Data Explorer", "📈 Model Performance", "🔮 Live Prediction"]
)

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_overview:
    st.header("What this project is about")
    st.markdown(
        """
**The question:** given the conditions around a road accident *before or as it
happens* — driver profile, road, junction, light, weather, and cause — how
severe is it likely to be: **Slight**, **Serious**, or **Fatal** injury?

**Why it matters:** traffic authorities and road-safety planners can use a
model like this to flag which combinations of conditions (e.g. pedestrian
collisions at night on curved roads) deserve the most attention — targeted
lighting, speed bumps, signage, enforcement — rather than treating every
crash-prone spot the same way.

**The data:** 12,316 real police-recorded road traffic accidents from Addis
Ababa, Ethiopia (2017-2020), originally 32 columns. Columns that only exist
*after* a crash is over (casualty counts, casualty details) were removed —
using them would leak the answer, since they're near-duplicates of the
severity label itself.

**Type of problem:** multi-class **classification** — 3 severity classes,
heavily imbalanced (84.6% Slight, 14.2% Serious, only 1.3% Fatal).
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents", f"{len(df):,}")
    c2.metric("Features used", len(meta["cat_cols"]) + len(meta["num_cols"]))
    c3.metric("Best model", model_info["name"])
    c4.metric("Test macro-F1", f"{test_report['macro_f1']:.3f}")

    st.subheader("Class balance")
    fig = px.bar(
        df["Accident_severity"].value_counts().reindex(SEV_ORDER).reset_index(),
        x="Accident_severity", y="count", color="Accident_severity",
        color_discrete_map=SEV_COLORS,
        labels={"count": "Number of accidents", "Accident_severity": "Severity"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 — DATA EXPLORER
# ===========================================================================
with tab_explore:
    st.header("Explore the raw accident data")

    with st.sidebar:
        st.subheader("Filters")
        light_filter = st.multiselect(
            "Light conditions", sorted(df["Light_conditions"].unique()),
            default=sorted(df["Light_conditions"].unique())
        )
        weekend_filter = st.multiselect(
            "Weekend?", [0, 1], default=[0, 1],
            format_func=lambda x: "Weekend" if x == 1 else "Weekday"
        )
        sev_filter = st.multiselect(
            "Severity", SEV_ORDER, default=SEV_ORDER
        )

    fdf = df[
        df["Light_conditions"].isin(light_filter)
        & df["is_weekend"].isin(weekend_filter)
        & df["Accident_severity"].isin(sev_filter)
    ]
    st.caption(f"Showing {len(fdf):,} of {len(df):,} accidents matching filters")

    col1, col2 = st.columns(2)
    with col1:
        feature_choice = st.selectbox(
            "Feature to break down by",
            ["Light_conditions", "Weather_conditions", "Cause_of_accident",
             "Type_of_collision", "Area_accident_occured", "time_of_day",
             "Age_band_of_driver", "Driving_experience"],
        )
        ct = pd.crosstab(fdf[feature_choice], fdf["Accident_severity"], normalize="index")
        ct = ct.reindex(columns=SEV_ORDER).fillna(0)
        ct = ct.loc[fdf[feature_choice].value_counts().index]  # order by frequency
        fig2 = px.bar(
            ct.reset_index().melt(id_vars=feature_choice, var_name="Severity", value_name="Share"),
            x=feature_choice, y="Share", color="Severity", barmode="stack",
            color_discrete_map=SEV_COLORS,
        )
        fig2.update_layout(xaxis_tickangle=-40, height=450)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.write("Raw filtered rows")
        st.dataframe(
            fdf[["Accident_severity", feature_choice, "time_of_day",
                 "Age_band_of_driver", "Driving_experience"]].head(300),
            height=450,
        )

# ===========================================================================
# TAB 3 — MODEL PERFORMANCE
# ===========================================================================
with tab_perf:
    st.header("How the model performs")

    st.subheader("Model comparison (validation set)")
    st.dataframe(comparison.style.format({
        "val_accuracy": "{:.3f}", "val_macro_f1": "{:.3f}", "val_recall_fatal": "{:.3f}"
    }), use_container_width=True)
    st.caption(
        "Macro-F1 is the metric that matters here, not accuracy — with 84.6% of "
        "accidents being 'Slight', a model that always predicts 'Slight' scores "
        "84.6% accuracy while being useless. Macro-F1 weights all 3 classes equally."
    )

    st.subheader(f"Final model: {model_info['name']} (tuned)")
    st.json(model_info["params"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{test_report['accuracy']:.3f}")
    c2.metric("Test macro-F1", f"{test_report['macro_f1']:.3f}")
    fatal_recall = test_report["report"]["Fatal"]["recall"]
    c3.metric("Fatal-class recall", f"{fatal_recall:.3f}",
              help="Of accidents that were actually Fatal, what share did the model catch?")

    st.image("figures/07_confusion_matrix.png",
              caption="Confusion matrix on the held-out test set (touched once)")

    st.warning(
        "**Is it good enough?** Not for real-world deployment as-is. The model "
        f"catches only {fatal_recall:.0%} of truly Fatal accidents — with only 158 "
        "Fatal examples in the whole dataset (1.3%), there's too little signal for "
        "any model trained this way to reliably flag the rare, highest-stakes class. "
        "It's a solid *baseline* that beats guessing, but a real deployment would need "
        "more Fatal-class data, likely combined with resampling (SMOTE) or a cost-"
        "sensitive threshold tuned specifically for recall on Fatal/Serious cases."
    )

# ===========================================================================
# TAB 4 — LIVE PREDICTION
# ===========================================================================
with tab_predict:
    st.header("Get a live prediction")
    st.caption("Fill in the conditions below — the saved model predicts severity instantly, no retraining.")

    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    target_order = meta["target_order"]  # ["Fatal injury","Serious Injury","Slight Injury"]

    input_vals = {}
    cols = st.columns(3)
    for i, c in enumerate(cat_cols):
        options = sorted(df[c].dropna().unique())
        with cols[i % 3]:
            input_vals[c] = st.selectbox(c.replace("_", " "), options, key=f"in_{c}")

    st.markdown("**Situational flags**")
    fcols = st.columns(3)
    with fcols[0]:
        input_vals["is_weekend"] = int(st.checkbox("Weekend", value=False))
    with fcols[1]:
        input_vals["junction_risk"] = int(st.checkbox(
            "At/near a risky junction or curve/grade", value=False))
    with fcols[2]:
        input_vals["is_vulnerable_road_user_involved"] = int(st.checkbox(
            "Pedestrian / vulnerable road user involved", value=False))

    if st.button("Predict severity", type="primary"):
        row = pd.DataFrame([input_vals])
        row_encoded = pd.get_dummies(row, columns=cat_cols)
        # align to the exact training feature set (missing dummy cols -> 0)
        row_encoded = row_encoded.reindex(columns=feature_cols, fill_value=0)

        X_in = scaler.transform(row_encoded) if uses_scaled else row_encoded
        pred_class = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]

        pred_label = target_order[pred_class].replace(" injury", "").replace(" Injury", "")
        color = SEV_COLORS.get(pred_label, "#333")

        st.markdown(
            f"### Predicted severity: <span style='color:{color}'>{pred_label}</span>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({
            "Severity": [t.replace(" injury", "").replace(" Injury", "") for t in target_order],
            "Probability": proba,
        })
        fig3 = px.bar(
            proba_df, x="Severity", y="Probability", color="Severity",
            color_discrete_map=SEV_COLORS, range_y=[0, 1],
        )
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Remember: the model's overall Fatal-class recall is low (see Model "
            "Performance tab) — treat a 'Fatal' or 'Serious' prediction as a signal "
            "to investigate further, not a certainty, and don't treat a 'Slight' "
            "prediction as a guarantee of safety."
        )
