import streamlit as st
import pandas as pd
import numpy as np
from charts import create_gauge_chart, create_comparison_chart, HAS_PLOTLY
from data_manager import load_artifacts

# -------------------------
# 1. PAGE CONFIG & STYLING
# -------------------------

st.set_page_config(
    page_title="Student Dropout Risk AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# 2. DATA LOADING
# -------------------------

pipeline, df, is_demo = load_artifacts()

# Prepare Feature Columns
if "Target" in df.columns:
    feature_cols = [c for c in df.columns if c != "Target"]
else:
    feature_cols = df.columns.tolist()

X = df[feature_cols]
medians = X.median(numeric_only=True)
means = X.mean(numeric_only=True)

# -------------------------
# 3. SIDEBAR CONFIG
# -------------------------

st.sidebar.title("🛠️ Student Configuration")

if is_demo:
    st.sidebar.warning("⚠️ **Demo Mode Active**\n\nOriginal model/dataset files not found. Using synthetic data.")

if not HAS_PLOTLY:
    st.sidebar.info("ℹ️ **Note:** `plotly` is not installed. Using basic charts. \n\nRun `pip install plotly` for advanced visuals.")

st.sidebar.write("Adjust the student parameters below:")

user_input = {}

# Grouping Logic
groups = {
    "📚 Academic Performance": ["grade", "sem", "enrolled", "approved", "evaluations"],
    "💰 Socio-Economic": ["tuition", "scholarship", "unemployment", "inflation", "gdp", "debtor"],
    "👤 Demographics": ["age", "gender", "marital", "nationality"],
}

# Assign columns to groups
grouped_cols = {k: [] for k in groups}
grouped_cols["⚙️ Other Features"] = []

for col in feature_cols:
    assigned = False
    col_lower = col.lower()
    for group_name, keywords in groups.items():
        if any(k in col_lower for k in keywords):
            grouped_cols[group_name].append(col)
            assigned = True
            break
    if not assigned:
        grouped_cols["⚙️ Other Features"].append(col)

# Render Inputs
for group_name, cols in grouped_cols.items():
    if cols:
        with st.sidebar.expander(group_name, expanded=(group_name == "📚 Academic Performance")):
            for col in cols:
                col_data = X[col]
                if np.issubdtype(col_data.dtype, np.number):
                    col_min = float(col_data.min())
                    col_max = float(col_data.max())
                    default = float(medians.get(col, col_min))
                    
                    nice_label = col.replace("_", " ").title()
                    user_input[col] = st.number_input(
                        label=nice_label,
                        min_value=col_min,
                        max_value=col_max,
                        value=default,
                        step=1.0 if col_data.dtype in [np.int64, np.int32] else 0.1,
                        key=col
                    )
                else:
                    unique_vals = sorted(col_data.dropna().unique().tolist())
                    if not unique_vals: unique_vals = ["Unknown"]
                    user_input[col] = st.selectbox(col, unique_vals, index=0, key=col)

input_df = pd.DataFrame([user_input])

# -------------------------
# 4. MAIN DASHBOARD
# -------------------------

st.title("🎓 Student Dropout Risk Predictor")
st.markdown("Use ML to identify at-risk students and intervene early.")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Action")
    st.info("Adjust values in the sidebar to simulate a student profile, then click Predict.")
    predict_btn = st.button("🔮 Analyze Student Risk")
    st.write("---")
    st.caption("Current Input Snapshot:")
    st.dataframe(input_df.T, height=300, use_container_width=True)

with col_right:
    if predict_btn:
        with st.spinner("Calculating risk profile..."):
            try:
                # Prediction
                proba = pipeline.predict_proba(input_df)[:, 1][0]
                pred_class = pipeline.predict(input_df)[0]
                
                # Logic for status
                if proba >= 0.7:
                    risk_status = "High Risk"
                    status_color = "red"
                    advice = "🚨 **Immediate Intervention Required.** Schedule counseling and review academic standing."
                elif proba >= 0.4:
                    risk_status = "Medium Risk"
                    status_color = "orange"
                    advice = "⚠️ **Monitor Closely.** Student shows signs of struggle. Verify tuition status."
                else:
                    risk_status = "Low Risk"
                    status_color = "green"
                    advice = "✅ **On Track.** Student seems stable, but maintain standard periodic check-ins."

                # Top Cards
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediction</h3>
                        <h2 style="color: {status_color};">{'Dropout' if pred_class == 1 else 'Graduate'}</h2>
                        <p>{risk_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Probability</h3>
                        <h2 style="color: {status_color};">{proba:.1%}</h2>
                        <p>Confidence Level</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("") 

                # Visualizations
                tab1, tab2 = st.tabs(["📊 Risk Analysis", "📉 Comparative View"])
                
                with tab1:
                    if HAS_PLOTLY:
                        st.plotly_chart(create_gauge_chart(proba), use_container_width=True)
                    else:
                        st.metric("Risk Probability", f"{proba:.1%}")
                        st.progress(proba)
                        
                    st.markdown(f"### 💡 AI Recommendation")
                    st.markdown(advice)

                with tab2:
                    chart_or_data = create_comparison_chart(input_df, means)
                    if chart_or_data is not None:
                        if HAS_PLOTLY:
                            st.plotly_chart(chart_or_data, use_container_width=True)
                        else:
                            st.bar_chart(chart_or_data)
                        st.caption("Comparison is based on the dataset average (mean) values.")
                    else:
                        st.write("Not enough numeric data for comparison.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.error("Check that your input data types match the model expectations.")
    
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 50px; color: #888;">
                <h3>👈 Ready to Predict</h3>
                <p>Configure the student profile in the sidebar and click <b>Analyze Student Risk</b>.</p>
                <br>
                <img src="https://cdn-icons-png.flaticon.com/512/2995/2995458.png" width="150" style="opacity: 0.5;">
            </div>
            """, unsafe_allow_html=True
        )