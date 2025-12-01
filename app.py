import streamlit as st
import pandas as pd
import numpy as np

# Import the logic and views from our modular files
from data_manager import load_artifacts
from risk_view import render_risk_tab
from performance_view import render_performance_tab
from advisor import get_ai_advice
from charts import HAS_PLOTLY

# -------------------------
# 1. PAGE CONFIG & STYLING
# -------------------------

st.set_page_config(
    page_title="Student Dropout Risk AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# 2. DATA LOADING
# -------------------------

pipeline, df, is_demo = load_artifacts()

if "Target" in df.columns:
    feature_cols = [c for c in df.columns if c != "Target"]
else:
    feature_cols = df.columns.tolist()

X = df[feature_cols]
medians = X.median(numeric_only=True)
modes = X.mode().iloc[0]

# -------------------------
# 3. SIDEBAR CONFIG
# -------------------------

st.sidebar.title("🎓 Student Profile")

if is_demo:
    st.sidebar.warning("⚠️ **Demo Mode Active**")

if not HAS_PLOTLY:
    st.sidebar.info("ℹ️ **Note:** `plotly` is not installed. Using basic charts.")

st.sidebar.write("Configure the Academic and Socio-Economic factors below.")

user_input = {}
academic_keywords = ["grade", "sem", "enrolled", "approved", "evaluations", "curricular"]
socio_keywords = ["tuition", "scholarship", "unemployment", "inflation", "gdp", "debtor", "fees"]
academic_widgets = []
socio_widgets = []

for col in feature_cols:
    col_lower = col.lower()
    is_academic = any(k in col_lower for k in academic_keywords)
    is_socio = any(k in col_lower for k in socio_keywords)
    
    if is_academic or is_socio:
        col_data = X[col]
        nice_label = col.replace("Curricular units", "").replace("(", "").replace(")", "").strip().title()
        
        # Determine Widget Type
        widget_args = {}
        if np.issubdtype(col_data.dtype, np.number):
            col_min = float(col_data.min())
            col_max = float(col_data.max())
            default_val = float(medians.get(col, col_min))
            widget_func = st.sidebar.number_input
            widget_args = {"label": nice_label, "min_value": col_min, "max_value": col_max, "value": default_val, "step": 1.0 if col_data.dtype in [np.int64, np.int32] else 0.1, "key": col}
        else:
            unique_vals = sorted(col_data.dropna().unique().tolist())
            widget_func = st.sidebar.selectbox
            widget_args = {"label": nice_label, "options": unique_vals, "index": 0, "key": col}

        if is_academic: academic_widgets.append((col, widget_func, widget_args))
        elif is_socio: socio_widgets.append((col, widget_func, widget_args))
    else:
        # Auto-fill hidden fields
        user_input[col] = medians[col] if col in medians else modes[col]

# Render Sidebar Groups
with st.sidebar.expander("📚 Academic Performance", expanded=True):
    for col, func, args in academic_widgets: user_input[col] = func(**args)

with st.sidebar.expander("💰 Socio-Economic Factors", expanded=True):
    for col, func, args in socio_widgets: user_input[col] = func(**args)

input_df = pd.DataFrame([user_input])

# -------------------------
# 4. MAIN DASHBOARD
# -------------------------

st.title("🎓 Student Dropout Predictor")
st.markdown("**Focus:** Academic & Socio-Economic Analysis")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Profile Summary")
    st.info("Demographic factors (Age, Gender, etc.) are set to the **Class Average**.")
    predict_btn = st.button("🔮 Analyze Risk")
    
    visible_cols = [c for c in input_df.columns if c in [w[0] for w in academic_widgets + socio_widgets]]
    st.caption("Active Inputs:")
    st.dataframe(input_df[visible_cols].T, height=300, use_container_width=True)

with col_right:
    if predict_btn:
        with st.spinner("Analyzing performance patterns..."):
            try:
                # Prediction
                raw_proba = pipeline.predict_proba(input_df)[:, 1][0]
                proba = float(raw_proba)
                raw_pred_class = pipeline.predict(input_df)[0]
                pred_class = int(raw_pred_class)
                
                # Logic for status
                if proba >= 0.7:
                    risk_status, status_color = "High Risk", "red"
                    advice = "🚨 **Critical:** Combined factors indicate a severe trajectory toward dropout."
                elif proba >= 0.4:
                    risk_status, status_color = "Medium Risk", "orange"
                    advice = "⚠️ **Warning:** Risk is elevated. Review tuition status and recent grades."
                else:
                    risk_status, status_color = "Low Risk", "green"
                    advice = "✅ **Stable:** Student is performing within safe ranges."

                # Top Cards
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediction</h3>
                        <h2 style="color: {status_color};">{'Dropout' if pred_class == 1 else 'Graduate'}</h2>
                        <p>{risk_status}</p>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Probability</h3>
                        <h2 style="color: {status_color};">{proba:.1%}</h2>
                        <p>Confidence Level</p>
                    </div>""", unsafe_allow_html=True)

                st.write("") 

                # TABS
                tab1, tab2, tab3 = st.tabs(["📊 Risk Analysis", "📈 Performance Trends", "🤖 AI Counselor"])
                
                # Tab 1: Risk Gauge (Delegated to risk_view.py)
                with tab1:
                    render_risk_tab(proba, advice)

                # Tab 2: Progress Report (Delegated to performance_view.py)
                with tab2:
                    render_performance_tab(input_df)

                # Tab 3: AI Advisor (Delegated to advisor.py)
                with tab3:
                    st.markdown("### 🤖 Personalized AI Counselor")
                    st.info("This feature uses Google's Gemini AI to analyze the specific student profile.")
                    
                    # 1. Try to get key from secrets
                    api_key = None
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                    except (FileNotFoundError, KeyError):
                        api_key = None
                    
                    # 2. If not in secrets, ask user to input it manually
                    if not api_key:
                        api_key = st.text_input("Enter Google Gemini API Key", type="password", help="If you set GEMINI_API_KEY in secrets.toml, this box will be skipped.")
                    else:
                        st.success("✅ API Key loaded securely from secrets.")

                    if st.button("Generate AI Advice"):
                        with st.spinner("Consulting AI Advisor..."):
                            # Call the function from advisor.py
                            ai_response = get_ai_advice(input_df, api_key)
                            st.markdown(ai_response)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.error("Check that your input data types match the model expectations.")
    else:
        st.markdown("""
            <div style="text-align: center; padding: 50px; color: #888;">
                <h3>👈 Configure Profile</h3>
                <p>Set the Academic and Economic factors in the sidebar to begin.</p>
                <img src="https://cdn-icons-png.flaticon.com/512/2995/2995458.png" width="100" style="opacity: 0.5;">
            </div>""", unsafe_allow_html=True)