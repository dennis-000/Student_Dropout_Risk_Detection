import google.generativeai as genai
import streamlit as st

def get_ai_advice(input_df, api_key):
    """
    Generates personalized academic advice using Google's Gemini API.
    
    Args:
        input_df (pd.DataFrame): The student's data.
        api_key (str): The Google Gemini API key.
        
    Returns:
        str: The advice text or error message.
    """
    if not api_key:
        return "⚠️ Please enter a valid Google API Key to receive AI advice."

    # Configure the API
    try:
        genai.configure(api_key=api_key)
        # Use a standard text model (Gemini 1.5 Flash is fast and effective)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error configuring API: {str(e)}"

    # Construct a readable profile string from the dataframe
    profile_summary = ""
    
    # We focus on the most relevant columns for advice
    relevant_columns = [
        "Curricular units 1st sem (grade)", 
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder",
        "Age at enrollment"
    ]
    
    # Iterate through columns to build the text summary
    for col in input_df.columns:
        # Include relevant columns or columns containing 'grade' or 'sem'
        if col in relevant_columns or "grade" in col.lower() or "sem" in col.lower():
            val = input_df[col].iloc[0]
            profile_summary += f"- {col}: {val}\n"

    # Create the prompt for the AI
    prompt = f"""
    You are an expert academic advisor and student counselor at a university. 
    Analyze the following student data and provide personalized, actionable advice to help them succeed and avoid dropping out.
    
    Student Profile Data:
    {profile_summary}
    
    Instructions:
    1. Identify the top 2 specific risk factors (e.g., declining grades, financial arrears, exam attendance).
    2. Provide 3 specific, encouraging, and actionable steps they can take immediately.
    3. If the student is doing well (High grades, fees paid), commend them and suggest how to maintain it.
    4. Keep the tone supportive, professional, but serious about the risks.
    5. Format the output with clear Markdown headings and bullet points.
    """

    # Call the API
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini AI: {str(e)}\n\nMake sure your API key is correct and valid."