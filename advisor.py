import streamlit as st

def get_ai_advice(input_df, api_key):
    """
    Generates personalized academic advice using Google's Gemini API.
    Auto-switch version: Tries 'Flash' first, falls back to 'Pro'.
    """
    print("--- DEBUG: AI Advisor Called ---") # Look for this in your terminal

    # 1. Check for API Key
    if not api_key:
        return "⚠️ **Error:** API Key is missing."

    # 2. Check for Library
    try:
        import google.generativeai as genai
    except ImportError:
        return "⚠️ **Library Error:** Run `pip install google-generativeai` in your terminal."

    # 3. Configure API
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"❌ **Config Error:** {str(e)}"

    # 4. Build Profile
    profile_summary = ""
    try:
        # Extract meaningful data
        for col in input_df.columns:
            # Only include columns with specific keywords to keep prompt clean
            if any(x in col.lower() for x in ['grade', 'sem', 'approved', 'tuition', 'debtor', 'scholarship', 'age']):
                val = input_df[col].iloc[0]
                profile_summary += f"- {col}: {val}\n"
        print(f"DEBUG: Profile Summary created ({len(profile_summary)} chars)")
    except Exception as e:
        return f"❌ **Data Processing Error:** {str(e)}"

    # 5. Prompt
    prompt = f"""
    Act as a university academic counselor. Analyze this student profile and provide 3 short, distinct, actionable tips to prevent dropout.
    
    Student Data:
    {profile_summary}
    
    Format:
    ### 🧠 AI Counselor Analysis
    * **Risk Assessment:** [One sentence summary]
    * **Action 1:** [Tip]
    * **Action 2:** [Tip]
    * **Action 3:** [Tip]
    """

    # 6. Call API with Fallback Logic
    try:
        # Attempt 1: Try the newest Flash model (Fast & Cheap)
        print("DEBUG: Attempting with gemini-1.5-flash...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("DEBUG: Success with Flash model.")
        return response.text

    except Exception as e_flash:
        print(f"DEBUG: Flash model failed ({str(e_flash)}). Switching to fallback...")
        
        try:
            # Attempt 2: Fallback to Gemini Pro (Standard, widely available)
            print("DEBUG: Attempting with gemini-pro...")
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            print("DEBUG: Success with Pro model.")
            return response.text
            
        except Exception as e_pro:
            print(f"DEBUG: All models failed. Error: {str(e_pro)}")
            # Return a detailed error to the UI
            return f"""
            ❌ **Connection Failed**
            
            Both model attempts failed. This usually means your API key is invalid or your library is very old.
            
            **Error Details:** {str(e_pro)}
            
            **Try this fix:**
            Open your terminal and run: `pip install --upgrade google-generativeai`
            """