
import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_excel_text_clean, vectorize_text, retrieve_docs
from groq_llm import ask_groq

st.set_page_config(page_title="📊 AI Competitor Insight Engine", layout="wide")
st.title("📊 Competitor Analysis Assistant (Excel + AI)")

uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    st.success("✅ File uploaded. Parsing...")
    data_text = load_excel_text_clean(uploaded_file)
    vectorstore = vectorize_text(data_text)

    if st.button("🧠 Generate Executive Summary"):
        st.info("Generating summary using Groq...")
        summary = ask_groq(f"Summarize the key competitor insights from this data:\n{data_text[:2000]}")
        st.markdown("### 📌 Executive Summary")
        st.write(summary)

    question = st.text_area("❓ Ask a question about the data")
    if question:
        context = retrieve_docs(vectorstore, question)
        full_prompt = f"Context:\n{context}\n\nAnswer the following:\n{question}"
        answer = ask_groq(full_prompt)
        st.markdown("### 💬 Answer")
        st.write(answer)

    # 🆕 Dashboard chart generation with explanation
    st.markdown("---")
    st.markdown("## 📊 Ask for a Custom Visual Dashboard")
    viz_query = st.text_area("🧠 Ask a dashboard/chart question (e.g., compare Madaen and Tamimi)")

    if viz_query:
        st.info("Analyzing your request and generating dashboard code + explanation...")

        context = retrieve_docs(vectorstore, viz_query)
        prompt = f"""
        You are a smart data assistant. Based on the user's question and this spreadsheet data,
        do the following:
        1. Generate Python chart code (using pandas + matplotlib or seaborn).
        2. Use abbreviation in chart of names as names can be too long(e.g., 'Madaen Real Estate'), abbreviate them in the chart.
        4. Write down all the numeric data you have used to make dashboard into a table in markdown language and dont make any assumption about data, so user know the dashboard is correct.
        3. Write a clear explanation of the chart and do analyses over the dashboard as well and expand any abbreviations.

        User's question:
        {viz_query}

        Spreadsheet context:
        {context}

        Return the response in the following format:
        ```python
        <code>
        ```

        Explanation:
        <text>
        """
        response = ask_groq(prompt)

        # Extract chart code and explanation
        code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)

        if code_match:
            chart_code = code_match.group(1)

            # st.markdown("### 🧪 Generated Chart Code")
            # st.code(chart_code, language="python")

            try:
                plt.close('all')  # clear any previous charts
                exec_globals = {
                    "pd": pd,
                    "plt": plt,
                    "sns": sns,
                    "__builtins__": __builtins__,
                }
                exec(chart_code, exec_globals)

                # Render all generated charts
                for fig_num in plt.get_fignums():
                    st.pyplot(plt.figure(fig_num))

            except Exception as e:
                st.error(f"❌ Error executing chart code: {e}")
        else:
            st.warning("⚠ No valid Python code found in LLM response.")

        if explanation_match:
            st.markdown("### 💡 Chart Insight")
            st.write(explanation_match.group(1))
        else:
            st.warning("⚠ No explanation found in the response.")
