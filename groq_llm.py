import requests
import os  # 👈 Add this
from dotenv import load_dotenv  # 👈 Optional, for local dev only
import streamlit as st

# Load .env file (only required locally, not on Streamlit Cloud)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"] # ✅ secure and clean
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def ask_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        json_data = response.json()

        if "choices" in json_data:
            return json_data["choices"][0]["message"]["content"]
        elif "error" in json_data:
            return f"⚠️ API Error: {json_data['error']}"
        else:
            return "⚠️ Unexpected API response format."
    except Exception as e:
        return f"❌ Failed to call Groq API: {str(e)}"
