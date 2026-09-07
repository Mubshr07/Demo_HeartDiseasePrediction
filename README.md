# ❤️ Explainable AI Cardiovascular Risk Assessment — Live Demo

**🔗 Live app: [heartdiseasedemo.streamlit.app](https://heartdiseasedemo.streamlit.app/)**

An interactive Streamlit web app that demonstrates the MS thesis research **"Enhancing Heart Disease Prediction using Explainable Artificial Intelligence."** Enter a patient health profile and the app returns a cardiovascular risk prediction along with a **SHAP-based explanation** showing exactly which factors drove that specific prediction — making the model's reasoning transparent rather than a black box.

This repo is the deployable web-app companion to the full research codebase, notebooks, and experiments here:
👉 [eXplainableAI_Cardiovascular_Risk_Assessment](https://github.com/Mubshr07/eXplainableAI_Cardiovascular_Risk_Assessment)

## About the research

The underlying model was trained on the **NHANES dataset** (37,079 records, 40–50 clinical/demographic features) using a neural network classifier, with **SHAP (SHapley Additive exPlanations)** applied to make individual predictions interpretable. This app exposes a subset of the most influential features as an interactive form, so anyone can generate a live prediction and see the explanation behind it — without needing to run the original notebooks.

## What the app does

1. **Collects a patient health profile** — age, blood pressure, cholesterol, BMI, glucose, smoking status, and other key risk factors — through sliders and dropdowns.
2. **Predicts cardiovascular risk** using the pre-trained model, shown as a risk probability and a Low / Moderate / High category.
3. **Explains the prediction** with a SHAP waterfall plot, showing which features pushed the risk score up or down for that specific patient.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application — UI, prediction logic, and SHAP explanation rendering |
| `model.pkl` | Pre-trained classifier, saved from the research notebooks |
| `scaler.pkl` | Fitted feature scaler/preprocessor used at training time |
| `features.json` | Feature names (and reference values) used to build the input form, in the exact order the model expects |
| `requirements.txt` | Pinned Python dependencies for reproducible local runs and cloud deployment |

## Running locally

```bash
git clone https://github.com/Mubshr07/Demo_HeartDiseasePrediction.git
cd Demo_HeartDiseasePrediction
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment

This app is deployed for free on **Streamlit Community Cloud**, connected directly to this GitHub repo — any push to `main` automatically redeploys the live app at [heartdiseasedemo.streamlit.app](https://heartdiseasedemo.streamlit.app/).

## ⚠️ Disclaimer

This tool is a **research and portfolio demonstration only**. It is **not medical advice** and must not be used for actual diagnosis, screening, or treatment decisions. No data entered into the app is stored or transmitted anywhere — all predictions run in-session and are discarded.

## Author

**Mubashir Iqbal**
AI Researcher | Machine Learning Engineer | Explainable AI
[GitHub](https://github.com/Mubshr07) · [Full research repo](https://github.com/Mubshr07/eXplainableAI_Cardiovascular_Risk_Assessment)
