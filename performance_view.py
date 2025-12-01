import streamlit as st
import pandas as pd

def render_performance_tab(input_df):
    """
    Renders the metrics and trends for the Student Progress Report.
    """
    st.markdown("### Student Progress Report")
    
    # 1. Calculate Grade Trend (Sem 1 vs Sem 2)
    # Use .get() to avoid errors if columns are missing
    grade1 = input_df.get("Curricular units 1st sem (grade)", pd.Series([0])).iloc[0]
    grade2 = input_df.get("Curricular units 2nd sem (grade)", pd.Series([0])).iloc[0]
    grade_delta = grade2 - grade1
    
    # 2. Calculate Pass Rate (Approved / Enrolled)
    enrolled = input_df.get("Curricular units 2nd sem (enrolled)", pd.Series([1])).iloc[0]
    approved = input_df.get("Curricular units 2nd sem (approved)", pd.Series([0])).iloc[0]
    
    # Avoid division by zero
    if enrolled == 0: 
        enrolled = 1 
    success_rate = (approved / enrolled) * 100

    # 3. Financial Check
    fees_paid = input_df.get("Tuition fees up to date", pd.Series([0])).iloc[0]

    # Display Metrics in 3 columns
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(
            label="Grade Trend (2nd Sem)", 
            value=f"{grade2:.1f}", 
            delta=f"{grade_delta:.1f}",
            help="Difference between 1st and 2nd Semester grades"
        )
    
    with m2:
        st.metric(
            label="Course Pass Rate", 
            value=f"{success_rate:.0f}%",
            help="Percentage of enrolled courses that were passed"
        )
    
    with m3:
        status_text = "Paid ✅" if fees_paid == 1 else "Unpaid ❌"
        st.metric(
            label="Tuition Status", 
            value=status_text,
            delta="OK" if fees_paid == 1 else "Critical",
            delta_color="normal" if fees_paid == 1 else "inverse"
        )
    
    st.divider()
    
    # Narrative Explanations (Logic to explain the numbers)
    if grade_delta > 0:
        st.success("📈 **Academic Improvement:** Grades have improved since the first semester. This is a strong retention signal.")
    elif grade_delta < 0:
        st.warning("📉 **Academic Decline:** Grades dropped in the second semester. This downward trend is a risk factor.")
    else:
        st.info("➡️ **Steady Performance:** Grades remained consistent between semesters.")
        
    if success_rate < 50:
        st.error(f"⚠️ **Struggling:** Student passed less than half ({success_rate:.0f}%) of their enrolled classes.")
    
    if fees_paid == 0:
        st.error("💳 **Financial Alert:** Tuition fees are outstanding. This is often the strongest predictor of dropout.")