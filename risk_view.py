import streamlit as st
from charts import create_gauge_chart, HAS_PLOTLY

def render_risk_tab(proba, advice):
    """
    Renders the visual content for the Risk Analysis tab.
    """
    if HAS_PLOTLY:
        # Interactive Gauge Chart
        st.plotly_chart(create_gauge_chart(proba), use_container_width=True)
    else:
        # Fallback for when Plotly is missing
        st.metric("Risk Probability", f"{proba:.1%}")
        st.progress(proba)
        
    st.markdown("### 💡 Insights")
    st.markdown(advice)