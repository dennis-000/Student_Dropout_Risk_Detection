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
    
    # this is our model from Kaggle
    model_path = os.path.join("models", "best_kaggle_XGBoost.joblib")
    dataset_path = "dataset.csv"
    
    use_demo_mode = False
    pipeline = None
    df = None

    # 1. Load Dataset
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            st.error(f"Error reading dataset.csv: {e}")
            use_demo_mode = True
    else:
        use_demo_mode = True

    # If dataset missing or failed, generate synthetic data matching the specific schema
    if use_demo_mode and df is None:
        # Create Dummy Data that mimics the structure of your specific dataset
        data = {
            'Curricular units 1st sem (grade)': np.random.normal(12, 2, 100),
            'Curricular units 2nd sem (grade)': np.random.normal(11, 2, 100),
            'Tuition fees up to date': np.random.choice([0, 1], 100),
            'Age at enrollment': np.random.randint(18, 40, 100),
            'Unemployment rate': np.random.uniform(5, 15, 100),
            'Inflation rate': np.random.uniform(0, 5, 100),
            'GDP': np.random.uniform(-2, 2, 100),
            'Scholarship holder': np.random.choice([0, 1], 100),
            'Debtor': np.random.choice([0, 1], 100),
            'Target': np.random.choice(['Dropout', 'Graduate', 'Enrolled'], 100)
        }
        df = pd.DataFrame(data)

    # 2. Load Model
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            use_demo_mode = True
    else:
        use_demo_mode = True

    # If model missing or failed, create a Mock Pipeline
    if use_demo_mode and pipeline is None:
        class MockPipeline:
            def predict_proba(self, X):
                # Fake logic: higher age + lower grades = higher dropout risk
                # We calculate a simple score to make the demo interactive
                risk_score = 0.5
                
                # Check for specific columns to adjust risk logic dynamically
                if 'Curricular units 2nd sem (grade)' in X.columns:
                    # Lower grades increase risk
                    grade = float(X['Curricular units 2nd sem (grade)'].iloc[0])
                    risk_score -= (grade - 10) * 0.05
                
                if 'Tuition fees up to date' in X.columns:
                    # Paying fees reduces risk significantly
                    fees = int(X['Tuition fees up to date'].iloc[0])
                    if fees == 1:
                        risk_score -= 0.3
                    else:
                        risk_score += 0.2

                # Clamp score between 0.05 and 0.95
                risk_score = np.clip(risk_score, 0.05, 0.95)
                
                # return [prob_class_0, prob_class_1]
                return np.array([[1 - risk_score, risk_score]])
            
            def predict(self, X):
                probs = self.predict_proba(X)
                # 1 = Dropout, 0 = Graduate/Enrolled (simplified for binary view)
                return [1 if probs[0][1] > 0.5 else 0]

        pipeline = MockPipeline()

    return pipeline, df, use_demo_mode