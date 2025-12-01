import pandas as pd
import numpy as np

# Try to import Plotly, handle error if not installed
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def create_gauge_chart(probability):
    """Creates a gauge chart for risk probability."""
    if not HAS_PLOTLY:
        return None

    if probability < 0.4:
        color = "green"
    elif probability < 0.7:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Dropout Probability", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_comparison_chart(input_df, df_mean):
    """
    Creates a bar chart comparing the student's normalized values 
    vs the average student for key numeric features.
    """
    # Select only numeric columns for comparison
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    
    # Pick top columns with specific interest based on the dataset
    # We prioritize academic/economic columns if present
    priority_cols = ["Curricular units 2nd sem (grade)", "Age at enrollment", "Unemployment rate"]
    cols_to_plot = [c for c in priority_cols if c in numeric_cols]
    
    # If priority columns aren't found, just take the first 3 numeric columns available
    if not cols_to_plot:
        cols_to_plot = numeric_cols[:3]
    
    if not cols_to_plot:
        return None

    # Handle Plotly Chart
    if HAS_PLOTLY:
        student_vals = input_df[cols_to_plot].iloc[0].values
        avg_vals = df_mean[cols_to_plot].values

        # Shorten names for the chart to make it cleaner
        short_names = [c.replace("Curricular units ", "").replace("(grade)", "Grade").replace(" rate", "") for c in cols_to_plot]

        fig = go.Figure(data=[
            go.Bar(
                name='This Student', 
                x=short_names, 
                y=student_vals, 
                marker_color='#3b82f6',
                text=[f"{v:.1f}" for v in student_vals], # Add value labels
                textposition='auto'
            ),
            go.Bar(
                name='Class Average', 
                x=short_names, 
                y=avg_vals, 
                marker_color='#9ca3af',
                text=[f"{v:.1f}" for v in avg_vals], # Add value labels
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Comparison: Student vs. Class Average",
            barmode='group',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    # Fallback: Return data for native Streamlit chart
    else:
        student_vals = input_df[cols_to_plot].iloc[0].values
        avg_vals = df_mean[cols_to_plot].values
        chart_data = pd.DataFrame({
            "This Student": student_vals,
            "Class Average": avg_vals
        }, index=cols_to_plot)
        return chart_data