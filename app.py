"""
🏥 Hospital Readmission Prediction — Streamlit Dashboard
Phase 5: Interactive Web App

Run locally:
    streamlit run app.py

Deploy:
    1. Push project to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → set main file = app.py → Deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1E3A5F; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #6B7280; margin-bottom: 1.5rem;
    }
    .risk-high {
        background: #FEF2F2; border-left: 5px solid #DC2626;
        padding: 1rem 1.5rem; border-radius: 8px;
    }
    .risk-medium {
        background: #FFFBEB; border-left: 5px solid #D97706;
        padding: 1rem 1.5rem; border-radius: 8px;
    }
    .risk-low {
        background: #F0FDF4; border-left: 5px solid #16A34A;
        padding: 1rem 1.5rem; border-radius: 8px;
    }
    .metric-card {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700;
        color: #1E3A5F; border-bottom: 2px solid #2563EB;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem 0;
    }
    .insight-box {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-size: 0.92rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    artifact = joblib.load('models/best_model.pkl')
    return artifact

@st.cache_resource
def load_explainer(_model):
    from sklearn.pipeline import Pipeline
    if isinstance(_model, Pipeline):
        base = _model.named_steps[list(_model.named_steps.keys())[-1]]
    else:
        base = _model
    return shap.TreeExplainer(base)

try:
    artifact  = load_model()
    model     = artifact['model']
    threshold = artifact['threshold']
    feat_names= artifact['feature_names']
    model_name= artifact['model_name']
    explainer = load_explainer(model)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    load_error   = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DIAG_CATEGORIES = [
    'circulatory', 'respiratory', 'digestive', 'diabetes',
    'injury', 'musculoskeletal', 'genitourinary',
    'neoplasms', 'mental', 'other', 'external'
]

MED_OPTIONS = ['No', 'Steady', 'Down', 'Up']
MED_MAP     = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}

def build_feature_vector(inputs: dict, feature_names: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the training feature matrix.
    All features default to 0; we fill in what the user provided.
    """
    row = {f: 0 for f in feature_names}

    # ── Numeric features ──────────────────────────────────────────────────────
    numeric_map = {
        'time_in_hospital'   : inputs['time_in_hospital'],
        'num_lab_procedures' : inputs['num_lab_procedures'],
        'num_procedures'     : inputs['num_procedures'],
        'num_medications'    : inputs['num_medications'],
        'number_outpatient'  : inputs['number_outpatient'],
        'number_emergency'   : inputs['number_emergency'],
        'number_inpatient'   : inputs['number_inpatient'],
        'number_diagnoses'   : inputs['number_diagnoses'],
        'total_visits'       : inputs['number_outpatient'] + inputs['number_emergency'] + inputs['number_inpatient'],
        'age_numeric'        : inputs['age_numeric'],
        'comorbidity_tier'   : inputs['comorbidity_tier'],
        'med_change_flag'    : inputs['med_change_flag'],
        'insulin_adjusted'   : inputs['insulin_adjusted'],
        'gender'             : inputs['gender'],
        'diabetesMed'        : inputs['diabetesMed'],
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    # ── Medication ordinal features ───────────────────────────────────────────
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'insulin',
                'glyburide-metformin', 'tolbutamide', 'miglitol']
    for col in med_cols:
        if col in row and col in inputs:
            row[col] = MED_MAP.get(inputs[col], 0)

    # ── One-hot: diagnosis categories ─────────────────────────────────────────
    for d_num in [1, 2, 3]:
        cat = inputs.get(f'diag_{d_num}_cat', 'other')
        col = f'diag_{d_num}_cat_{cat}'
        if col in row:
            row[col] = 1

    # ── One-hot: race ─────────────────────────────────────────────────────────
    race_col = f"race_{inputs.get('race', 'Unknown')}"
    if race_col in row:
        row[race_col] = 1

    # ── One-hot: admission / discharge ────────────────────────────────────────
    for field in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        col = f'{field}_{inputs.get(field, 1)}'
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])


def get_risk_level(prob: float, threshold: float):
    if prob >= threshold * 1.5:
        return 'HIGH', '#DC2626', 'risk-high', '🔴'
    elif prob >= threshold:
        return 'MODERATE', '#D97706', 'risk-medium', '🟡'
    else:
        return 'LOW', '#16A34A', 'risk-low', '🟢'


def plot_waterfall(sv_row, max_display=12):
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(sv_row, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_vals, feat_names, top_n=12):
    vals  = pd.Series(shap_vals, index=feat_names)
    top   = vals.abs().nlargest(top_n)
    top_v = vals[top.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#DC2626' if v > 0 else '#2563EB' for v in top_v.values[::-1]]
    bars    = ax.barh(top_v.index[::-1], top_v.values[::-1],
                      color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('SHAP value (impact on readmission risk)')
    ax.set_title('Feature Contributions for This Patient', fontweight='bold')

    red_p  = mpatches.Patch(color='#DC2626', label='Increases risk')
    blue_p = mpatches.Patch(color='#2563EB', label='Decreases risk')
    ax.legend(handles=[red_p, blue_p], fontsize=9, loc='lower right')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Details")
    st.markdown("Fill in the clinical information below.")
    st.divider()

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown("**Demographics**")
    age = st.selectbox("Age group", [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ], index=6)
    age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
               '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
               '[80-90)':85,'[90-100)':95}
    age_numeric = age_map[age]

    gender = st.selectbox("Gender", ['Female', 'Male'])
    race   = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'])

    st.divider()

    # ── Visit Information ─────────────────────────────────────────────────────
    st.markdown("**Current Visit**")
    time_in_hospital   = st.slider("Days in hospital", 1, 14, 4)
    num_lab_procedures = st.slider("Lab procedures", 1, 100, 42)
    num_procedures     = st.slider("Other procedures", 0, 6, 1)
    num_medications    = st.slider("Medications prescribed", 1, 30, 15)
    number_diagnoses   = st.slider("Number of diagnoses coded", 1, 16, 7)

    st.divider()

    # ── Prior Visits ──────────────────────────────────────────────────────────
    st.markdown("**Prior Year Visits**")
    number_inpatient  = st.number_input("Inpatient visits",  0, 20, 0)
    number_outpatient = st.number_input("Outpatient visits", 0, 30, 0)
    number_emergency  = st.number_input("Emergency visits",  0, 20, 0)

    st.divider()

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    st.markdown("**Diagnoses**")
    diag_1 = st.selectbox("Primary diagnosis",   DIAG_CATEGORIES, index=0)
    diag_2 = st.selectbox("Secondary diagnosis", DIAG_CATEGORIES, index=9)
    diag_3 = st.selectbox("Tertiary diagnosis",  DIAG_CATEGORIES, index=9)

    st.divider()

    # ── Medications ───────────────────────────────────────────────────────────
    st.markdown("**Key Medications**")
    insulin_status   = st.selectbox("Insulin",    MED_OPTIONS, index=1)
    metformin_status = st.selectbox("Metformin",  MED_OPTIONS, index=1)
    change           = st.radio("Any diabetes medication changed?", ['No', 'Yes'], horizontal=True)
    diabetes_med     = st.radio("On any diabetes medication?",      ['Yes', 'No'], horizontal=True)

    # Derived flags
    med_change_flag = 1 if change == 'Yes' else 0
    insulin_adjusted = 1 if insulin_status in ['Up', 'Down'] else 0
    comorbidity_tier = 0 if number_diagnoses <= 3 else (1 if number_diagnoses <= 6 else 2)

    st.divider()
    predict_btn = st.button("🔍 Predict Readmission Risk", use_container_width=True, type='primary')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Hospital 30-Day Readmission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-powered risk assessment using XGBoost + SHAP explainability · UCI Diabetes 130-US Hospitals Dataset</div>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"⚠️ Could not load model: `{load_error}`")
    st.info("Make sure `models/best_model.pkl` exists. Run Phase 3 notebook first.")
    st.stop()

# ── Model info banner ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", model_name.split()[0])
with col2:
    st.metric("ROC-AUC", f"{artifact.get('roc_auc', 0):.3f}")
with col3:
    st.metric("Decision Threshold", f"{threshold:.2f}")
with col4:
    st.metric("Training Data", "~70K patients")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    inputs = {
        'age_numeric'              : age_numeric,
        'gender'                   : 1 if gender == 'Male' else 0,
        'race'                     : race,
        'time_in_hospital'         : time_in_hospital,
        'num_lab_procedures'       : num_lab_procedures,
        'num_procedures'           : num_procedures,
        'num_medications'          : num_medications,
        'number_diagnoses'         : number_diagnoses,
        'number_inpatient'         : number_inpatient,
        'number_outpatient'        : number_outpatient,
        'number_emergency'         : number_emergency,
        'diag_1_cat'               : diag_1,
        'diag_2_cat'               : diag_2,
        'diag_3_cat'               : diag_3,
        'insulin'                  : insulin_status,
        'metformin'                : metformin_status,
        'med_change_flag'          : med_change_flag,
        'insulin_adjusted'         : insulin_adjusted,
        'comorbidity_tier'         : comorbidity_tier,
        'diabetesMed'              : 1 if diabetes_med == 'Yes' else 0,
        'admission_type_id'        : 1,
        'discharge_disposition_id' : 1,
        'admission_source_id'      : 7,
    }

    with st.spinner("Running prediction..."):
        X_input = build_feature_vector(inputs, feat_names)

        # Predict
        prob    = model.predict_proba(X_input)[0, 1]
        pred    = int(prob >= threshold)
        risk_label, risk_color, risk_class, risk_emoji = get_risk_level(prob, threshold)

        # SHAP
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            X_transformed = model[:-1].transform(X_input)
            X_for_shap    = pd.DataFrame(X_transformed, columns=feat_names)
        else:
            X_for_shap = X_input

        sv_single = explainer(X_for_shap)
        if len(sv_single.shape) == 3:
            sv_single = sv_single[:, :, 1]

    # ── Risk Score Banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="{risk_class}">
        <h2 style="margin:0; color:{risk_color};">{risk_emoji} {risk_label} RISK</h2>
        <p style="margin:0.3rem 0 0 0; font-size:1.1rem;">
            Predicted 30-day readmission probability: <strong>{prob:.1%}</strong>
            &nbsp;|&nbsp; Threshold: {threshold:.0%}
            &nbsp;|&nbsp; Prediction: {'⚠️ Likely readmitted' if pred else '✅ Unlikely readmitted'}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        st.markdown('<div class="section-header">🔍 SHAP Waterfall — Why this prediction?</div>', unsafe_allow_html=True)
        st.caption("Each bar shows how a feature pushes the prediction above (red) or below (blue) the base rate.")
        waterfall_fig = plot_waterfall(sv_single[0], max_display=12)
        st.pyplot(waterfall_fig, use_container_width=True)
        plt.close()

    with right_col:
        st.markdown('<div class="section-header">📊 Feature Contributions</div>', unsafe_allow_html=True)
        st.caption("Magnitude and direction of each feature's impact on this patient's risk score.")
        bar_fig = plot_shap_bar(sv_single.values[0], feat_names, top_n=12)
        st.pyplot(bar_fig, use_container_width=True)
        plt.close()

    # ── Patient Summary ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.markdown(f'<div class="metric-card"><b>Age</b><br>{age}</div>', unsafe_allow_html=True)
    with summary_cols[1]:
        st.markdown(f'<div class="metric-card"><b>Days in Hospital</b><br>{time_in_hospital}</div>', unsafe_allow_html=True)
    with summary_cols[2]:
        st.markdown(f'<div class="metric-card"><b>Prior Inpatient Visits</b><br>{number_inpatient}</div>', unsafe_allow_html=True)
    with summary_cols[3]:
        st.markdown(f'<div class="metric-card"><b>Diagnoses Coded</b><br>{number_diagnoses}</div>', unsafe_allow_html=True)

    # ── Clinical Insights ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Clinical Insights</div>', unsafe_allow_html=True)
    top_shap   = pd.Series(sv_single.values[0], index=feat_names).abs().nlargest(3)
    directions = pd.Series(sv_single.values[0], index=feat_names)

    for feat in top_shap.index:
        direction = "increases" if directions[feat] > 0 else "decreases"
        arrow     = "🔺" if directions[feat] > 0 else "🔻"
        val       = X_input[feat].values[0] if feat in X_input.columns else "N/A"
        st.markdown(
            f'<div class="insight-box">{arrow} <b>{feat}</b> = {val} &nbsp;→&nbsp; '
            f'{direction} readmission risk (SHAP: {directions[feat]:+.4f})</div>',
            unsafe_allow_html=True
        )

    # ── Risk Gauge ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Risk Gauge</div>', unsafe_allow_html=True)
    fig_gauge, ax = plt.subplots(figsize=(8, 1.8))
    ax.barh([''], [1], color='#F1F5F9', height=0.5)
    ax.barh([''], [prob], color=risk_color, height=0.5, alpha=0.85)
    ax.axvline(threshold, color='#374151', lw=2, ls='--')
    ax.text(threshold + 0.01, 0, f'Threshold\n{threshold:.0%}', va='center', fontsize=9, color='#374151')
    ax.text(prob - 0.02, 0, f'{prob:.1%}', va='center', ha='right', fontsize=11,
            fontweight='bold', color='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted Readmission Probability')
    ax.set_title('Risk Score', fontweight='bold', pad=8)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.info("👈 Fill in patient details in the sidebar and click **Predict Readmission Risk** to get started.")

    st.markdown("### About This Project")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🎯 What it predicts**
        Whether a diabetic patient will be readmitted to hospital within 30 days of discharge —
        the threshold used by CMS for hospital penalty assessments.
        """)
    with col_b:
        st.markdown("""
        **🧠 How it works**
        An XGBoost / LightGBM model trained on 70,000+ real patient records from
        130 US hospitals (1999–2008), with SMOTE to handle class imbalance.
        """)
    with col_c:
        st.markdown("""
        **🔍 Why SHAP?**
        SHAP (SHapley Additive exPlanations) explains *why* each prediction was made —
        making the model trustworthy for clinical and administrative use.
        """)

    st.markdown("### Tech Stack")
    st.code("""
Python  ·  scikit-learn  ·  XGBoost  ·  LightGBM
SHAP  ·  imbalanced-learn (SMOTE)  ·  Streamlit
pandas  ·  matplotlib  ·  seaborn
Dataset: UCI Diabetes 130-US Hospitals
    """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This tool is for educational and portfolio purposes only. "
    "It is not a substitute for clinical judgment or certified medical software."
)
