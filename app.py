"""
Explainable AI Cardiovascular Risk Assessment
-----------------------------------------------
Streamlit demo app for the MS thesis project:
"Enhancing Heart Disease Prediction using Explainable Artificial Intelligence"

HOW TO PLUG IN YOUR REAL MODEL
================================
This app runs out of the box in DEMO MODE using a small synthetic model so you
can test the UI immediately. To use your real trained model:

1. In your training notebook, after training your final model, save:
       import pickle, json
       pickle.dump(model,  open("model.pkl", "wb"))
       pickle.dump(scaler, open("scaler.pkl", "wb"))
       json.dump(FEATURE_NAMES, open("features.json", "w"))   # exact column order used in training

2. Put model.pkl, scaler.pkl, and features.json in the same folder as this app.py.

3. Edit the FEATURE_NAMES list below (~line 40) so it EXACTLY matches the order
   and names of the columns your scaler/model were trained on. This is the most
   common source of bugs -- a mismatched order will silently produce garbage
   predictions instead of an error.

4. If your model is a Keras/TensorFlow model instead of a pickled sklearn model,
   see the `load_artifacts()` function below -- there's a commented-out branch
   for `tf.keras.models.load_model("model.h5")`.

Run locally with:  streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# SHAP is optional at import time so the app doesn't crash if it's missing
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Cardiovascular Risk Assessment (XAI)",
    page_icon="❤️",
    layout="wide",
)

MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")
FEATURES_PATH = Path("features.json")

# TODO: replace with the exact feature order your model/scaler were trained on.
# This default list is a reasonable NHANES-style subset for the demo/UI.
FEATURE_NAMES = [
    "Age",
#    "sex",
    "Body-Mass-Index",
    "Systolic",
    "Diastolic",
    "Total-Cholesterol",
    "Cholesterol",
    "Triglycerides",
    "Glucose",
#    "waist_circumference",
#    "current_smoker",
    "Diabetes",
#    "family_history_chd",
#    "physical_activity",
] 

# Human-readable labels + input widgets for each feature.
# type: "slider" | "select"
FEATURE_CONFIG = {
    "Age":                  {"label": "Age (years)",                    "type": "slider", "min": 18,  "max": 90,  "default": 45,  "step": 1},
#    "sex":                  {"label": "Sex",                             "type": "select", "options": {"Female": 0, "Male": 1}, "default": "Male"},
    "Body-Mass-Index":                  {"label": "BMI (kg/m²)",                     "type": "slider", "min": 15.0, "max": 55.0, "default": 27.0, "step": 0.1},
    "Systolic":             {"label": "Systolic Blood Pressure (mmHg)",  "type": "slider", "min": 80,  "max": 220, "default": 125, "step": 1},
    "Diastolic":         {"label": "Diastolic Blood Pressure (mmHg)", "type": "slider", "min": 40,  "max": 140, "default": 80,  "step": 1},
    "Total-Cholesterol":    {"label": "Total Cholesterol (mg/dL)",       "type": "slider", "min": 100, "max": 400, "default": 200, "step": 1},
    "Cholesterol":      {"label": "HDL Cholesterol (mg/dL)",         "type": "slider", "min": 20,  "max": 100, "default": 50,  "step": 1},
    "Triglycerides":        {"label": "Triglycerides (mg/dL)",           "type": "slider", "min": 50,  "max": 500, "default": 150, "step": 1},
    "Glucose":      {"label": "Fasting Glucose (mg/dL)",         "type": "slider", "min": 60,  "max": 300, "default": 95,  "step": 1},
#    "waist_circumference":  {"label": "Waist Circumference (cm)",       "type": "slider", "min": 60,  "max": 160, "default": 90,  "step": 1},
#    "current_smoker":       {"label": "Current Smoker",                 "type": "select", "options": {"No": 0, "Yes": 1}, "default": "No"},
    "Diabetes":             {"label": "Diagnosed Diabetes",             "type": "select", "options": {"No": 0, "Yes": 1}, "default": "No"},
#    "family_history_chd":   {"label": "Family History of Heart Disease","type": "select", "options": {"No": 0, "Yes": 1}, "default": "No"},
#    "physical_activity":    {"label": "Regular Physical Activity",      "type": "select", "options": {"No": 0, "Yes": 1}, "default": "Yes"},
} 

# --------------------------------------------------------------------------------------
# Artifact loading (model, scaler, feature order) with a synthetic DEMO fallback
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """
    Loads model.pkl / scaler.pkl / features.json if present.
    Otherwise trains a small synthetic logistic-regression model on the fly so
    the app is fully functional out of the box for demo/testing purposes.

    Returns: (model, scaler, feature_names, is_demo_mode: bool)
    """
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, "r") as f:
            feature_names = json.load(f)
            #st.write(f"Type of the feature is: {type(feature_names)}")
            #my_var = {"Feature Names are": feature_names }
            #st.write("Debug info:", my_var)
            #st.write("Debug info:", type(my_var)) 
        
        # --- If you're using a Keras/TensorFlow model instead, use this instead: ---
        # import tensorflow as tf
        # model = tf.keras.models.load_model("model.h5")
        # (Keras models don't have predict_proba; see predict_risk() below for the branch)

        return model, scaler, feature_names, False


    # ---------------- DEMO MODE: synthetic fallback model ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n_samples = 2000
    n_features = len(FEATURE_NAMES)
    X_demo = rng.normal(size=(n_samples, n_features))

    # Hand-picked weights so higher age/BP/cholesterol/smoking push risk up,
    # just to make the demo behave sensibly -- not a real trained model.
    weights = np.array([1.4, 0.2, 0.6, 1.0, 0.5, 0.7, -0.8, 0.4, 0.6, 0.5, 0.9, 1.1, 0.5, -0.6])
    logits = X_demo @ weights + rng.normal(scale=0.5, size=n_samples)
    y_demo = (logits > np.median(logits)).astype(int)

    scaler = StandardScaler().fit(X_demo)
    model = LogisticRegression().fit(scaler.transform(X_demo), y_demo)

    return model, scaler, FEATURE_NAMES, True


def predict_risk(model, scaler, X_row: pd.DataFrame):
    """Scales a single-row DataFrame and returns (predicted_class, probability_of_risk)."""
    X_scaled = scaler.transform(X_row)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        print(f"The prob value is: {proba}")
        prob_positive = proba[1] if len(proba) > 1 else proba[0]
    else:
        # Keras-style model: model.predict returns raw probabilities/logits
        prob_positive = float(model.predict(X_scaled)[0][0])
    
    predicted_class = int(prob_positive >= 0.6)

    return predicted_class, prob_positive


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.title("❤️ Explainable AI Cardiovascular Risk Assessment")
st.caption(
    "Based on the MS thesis research *\"Enhancing Heart Disease Prediction using "
    "Explainable Artificial Intelligence\"* (NHANES dataset, MLP model, SHAP explainability)."
)

model, scaler, feature_names, is_demo_mode = load_artifacts()

if is_demo_mode:
    st.warning(
        "⚠️ **Demo mode:** no trained model files were found (`model.pkl`, `scaler.pkl`, "
        "`features.json`), so this app is running on a small synthetic stand-in model just "
        "to demonstrate the interface. Drop your real trained artifacts into the app folder "
        "to replace it with the actual research model — see the comment block at the top of "
        "`app.py` for exact instructions.",
        icon="⚠️",
    )

st.divider()



left_col, right_col = st.columns([1, 1.3])

# By Default Data:
user_inputs = feature_names
    


with left_col:
    st.subheader("Patient Health Profile")
    st.caption("Adjust the values below to reflect the profile you want to assess.")

    with st.form("risk_form"):
        for feat in FEATURE_NAMES:
            cfg = FEATURE_CONFIG.get(feat)
            if cfg is None:
                # Feature has no widget defined -- fall back to a plain number input
                user_inputs[feat] = st.number_input(feat, value=0.0)
                continue

            if cfg["type"] == "slider":
                user_inputs[feat] = st.slider(
                    cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                    value=cfg["default"], step=cfg["step"],
                )
            elif cfg["type"] == "select":
                choice = st.selectbox(cfg["label"], options=list(cfg["options"].keys()),
                                       index=list(cfg["options"].keys()).index(cfg["default"]))
                user_inputs[feat] = cfg["options"][choice]

        submitted = st.form_submit_button("Assess Risk", use_container_width=True, type="primary")

with right_col:
    st.subheader("Prediction & Explanation")

    if not submitted:
        st.write("Fill in the patient profile on the left and click **Assess Risk** to see results.")

    else: 
        X_row = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)
        
        predicted_class, prob_positive = predict_risk(model, scaler, X_row)
        risk_pct = prob_positive * 100
        #st.write(f"The Predicted Class is: {predicted_class}, and risk is: {risk_pct}, ProbibalityPositive: {prob_positive}")
        
        if risk_pct < 40:
            risk_label, risk_color = "Low Risk", "green"
        elif risk_pct < 80:
            risk_label, risk_color = "Moderate Risk", "orange"
        else:
            risk_label, risk_color = "High Risk", "red"

        m1, m2 = st.columns(2)
        m1.metric("Predicted Risk Probability", f"{risk_pct:.1f}%")
        m2.markdown(f"### :{risk_color}[{risk_label}]")
        st.progress(min(max(prob_positive, 0.0), 1.0))

        st.divider()
        st.markdown("#### Why did the model predict this?")
 

        if not SHAP_AVAILABLE:
            st.caption(
                "Install the `shap` package (see requirements.txt) to enable per-prediction "
                "explanations here."
            )
        else:
            try:
                X_scaled = scaler.transform(X_row)
                explainer = shap.Explainer(model, scaler.transform(pd.DataFrame([[FEATURE_CONFIG.get(f, {}).get("default", 0)
                if not isinstance(FEATURE_CONFIG.get(f, {}).get("default"), str)
                else 0 for f in feature_names]], columns=feature_names)
                    ))
                shap_values = explainer(X_scaled)
                # For binary classifiers shap_values may have a class dimension
                sv = shap_values[0, :, 1] if shap_values.values.ndim == 3 else shap_values[0]
                #st.write(f"The Shape Values: {sv}.")


                if sv is not None:
                    sv_named = shap.Explanation(
                        values=sv.values,
                        base_values=sv.base_values,
                        data=sv.data,
                        feature_names=feature_names,
                    )
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    shap.plots.waterfall(sv_named, show=False, max_display=12)
                    st.pyplot(fig, clear_figure=True)
                    st.caption(
                        "Red bars push the risk prediction higher, blue bars push it lower. "
                        "Bar length shows the magnitude of each feature's contribution for "
                        "*this specific patient*."
                    )
                 
            except Exception as e:
                st.caption(f"Could not generate SHAP explanation for this model/input: {e}")
                

st.divider()
st.caption(
    "Source code & full research notebooks: "
    "[github.com/Mubshr07/eXplainableAI_Cardiovascular_Risk_Assessment]"
    "(https://github.com/Mubshr07/eXplainableAI_Cardiovascular_Risk_Assessment)"
)

# --------------------------------------------------------------------------------------
# Personal Branding / About Me
# --------------------------------------------------------------------------------------
profile_col, intro_col = st.columns([1, 3])

with profile_col:
    st.image("https://avatars.githubusercontent.com/u/34352213?v=4", width=160)   # place profile.jpg in the same folder as app.py

with intro_col:
    st.markdown("## Mubashir Iqbal")
    st.markdown("##### AI/ML Engineer | 12+ Years in Systems & Embedded Software (Qt/C++) → Applied Machine Learning & Explainable AI | System Automation")
    st.write(
        "I build interpretable machine learning systems for high-stakes domains like "
        "healthcare, where understanding *why* a model makes a prediction matters as much "
        "as the prediction itself. This app is a live demo of my MS thesis research, "
        "**\"Enhancing Heart Disease Prediction using Explainable Artificial Intelligence,\"** "
        "which combines a neural network trained on the NHANES dataset with SHAP-based "
        "explainability to make cardiovascular risk predictions transparent and clinically "
        "meaningful."
    )
    st.markdown(
        "[GitHub](https://github.com/Mubshr07) &nbsp;|&nbsp; "
        "[LinkedIn](https://www.linkedin.com/in/mubshr07/) &nbsp;|&nbsp; "
        "[Email](mailto:mubshr07@gmail.com)"
    )

st.divider()

st.info(
    "**Disclaimer:** This tool is a research/portfolio demonstration only. It is **not** "
    "medical advice and must not be used for actual diagnosis or treatment decisions. "
    "No input data entered here is stored or transmitted anywhere.",
    icon="ℹ️",
)
