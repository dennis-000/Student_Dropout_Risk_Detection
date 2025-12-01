import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

@st.cache_resource
def load_artifacts():
    """
    Tries to load real model/data. 
    If not found, generates synthetic data for Demo purposes.
    """
    model_path = os.path.join("models", "best_kaggle_XGBoost.joblib")
    dataset_path = "dataset.csv"
    
    use_demo_mode = False
    pipeline = None
    df = None

    # Check Dataset
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        use_demo_mode = True
        # Create Dummy Data for Demo
        data = {
            'Curricular units 1st sem (grade)': np.random.normal(12, 2, 100),
            'Curricular units 2nd sem (grade)': np.random.normal(11, 2, 100),
            'Tuition fees up to date': np.random.choice([0, 1], 100),
            'Age at enrollment': np.random.randint(18, 40, 100),
            'Unemployment rate': np.random.uniform(5, 15, 100),
            'Inflation rate': np.random.uniform(0, 5, 100),
            'GDP': np.random.uniform(-2, 2, 100),
            'Target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)

    # Check Model
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        use_demo_mode = True
        # Mock class to simulate model behavior
        class MockPipeline:
            def predict_proba(self, X):
                # Fake logic: higher age + lower grades = higher dropout risk
                risk_score = 0.5
                if 'Curricular units 2nd sem (grade)' in X.columns:
                    risk_score -= (X['Curricular units 2nd sem (grade)'].iloc[0] - 10) * 0.05
                if 'Tuition fees up to date' in X.columns:
                     risk_score -= X['Tuition fees up to date'].iloc[0] * 0.2
                
                risk_score = np.clip(risk_score, 0.05, 0.95)
                return np.array([[1-risk_score, risk_score]])
            
            def predict(self, X):
                probs = self.predict_proba(X)
                return [1 if probs[0][1] > 0.5 else 0]

        pipeline = MockPipeline()

    return pipeline, df, use_demo_mode