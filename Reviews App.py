# import packages
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from io import BytesIO
from PIL import Image


def reload_model():
    return None


def recommend_companies(df, user_query, top=7):
    """
    :param df: uploaded job
    :param user_query: string
    :return: list of dicts with keys: company, job_title, similarity, explanation
    """

    res = []
    for i in range(min(top, 7)):
        res.append({
            "Company": f"company{i + 1}",
            "Job Title": "Data Analyst Intern",
            "Similarity": round(random.uniform(.55, .95), 3),
            "Explanation": "Matched on:remote, internship,pay"
        })
    return res


def chat_response(user_message, history):
    return f"I hear you: \"{user_message}\".(This is a placeholder response)"


st.set_page_config(page_title="JobMatchAI - Pro", layout="wide", initial_sidebar_state="expanded")

AUTO_THEME_SCRIPT = """
    <script>
(function() {
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const root = document.documentElement;
  if (prefersDark) {
    root.setAttribute('data-theme', 'dark');
  } else {
    root.setAttribute('data-theme', 'light');
  }
})();
</script>
"""

CUSTOM_CSS = r"""
    <style>
:root[data-theme="light"] {
  --bg: #0f172a;
  --card: rgba(255,255,255,0.06);
  --text: #0b1220;
  --accent1: linear-gradient(90deg,#7c3aed, #06b6d4);
}
:root[data-theme="dark"] {
  --bg: #070812;
  --card: rgba(255,255,255,0.04);
  --text: #dbeafe;
  --accent1: linear-gradient(90deg,#06b6d4, #7c3aed);
}

/* Apply glass card effect to streamlit elements */
main .block-container {
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  padding: 1.6rem 2rem;
}
section[data-testid="stSidebar"] .css-1d391kg {
  background: transparent;
}
.css-1d391kg, .css-1d391kg .stButton button {
  border-radius: 14px;
}

/* Title style */
.header {
  display:flex; align-items:center; gap:12px;
}
.logo-circle {
  width:56px;height:56px;border-radius:12px;
  background: var(--accent1);
  display:flex;align-items:center;justify-content:center;color:white;font-weight:700;
  box-shadow: 0 8px 30px rgba(99,102,241,0.15);
}

/* Card style used in columns */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  padding: 16px;
  border-radius: 12px;
}

/* small pills */
.pill {
  display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;background:rgba(255,255,255,0.03);
}

/* bot bubble */
.bot {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 10px 12px;border-radius:12px;margin:6px 0;
}
</style>
"""

st.markdown(AUTO_THEME_SCRIPT, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px'><div class='logo-circle'>JM</div><div><h3 style='margin:0'>JobMatchAI</h3><div style='font-size:12px;color:gray'>NLP · Transformers · Explainability</div></div></div>",
        unsafe_allow_html=True)
    st.markdown("------")
    email = st.text_input("Your email (option)", placeholder="you@example.com")
    st.markdown("**Quick Settings**")
    top = st.slider("Number of Recommendation", 3, 12, 7)
    show_explain = st.checkbox("Show explanation", value=True)
    st.markdown("------")
    st.caption("Built w/ spacy + Sentence-Transformers . prototype")

tabs = st.tabs(["Dashboard", "Upload & Query", "Chatbot", "Insight", "Settings"])

with tabs[0]:
    st.markdown(
        "<div style='display:flex;justify-content:space-between;align-items:center'><div><h1 style='margin:0'>JobMatchAI</h1><div style='color:gray'>AI-powered job recommender — semantic search on reviews</div></div></div>",
        unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])
    with c1:
        st.markdown(
            "<div class='card'><h4 style='margin:0'>Top Match Preview</h4><div style='color:gray;margin-top:6px'>Quick glance at what users search for</div></div>",
            unsafe_allow_html=True)
        st.write("")
    st.dataframe(pd.DataFrame({
        "Company": ["Company A", "Company B", "Company C"],
        "Top Role": ["Data Analyst Intern", "ML Engineer", "Product Analyst"],
        "Avg Rating": [4.4, 4.1, 3.9]
    }))

with c2:
    st.markdown(
        "<div class='card'><h4 style='margin:0'>Companies</h4><div style='font-size:22px;font-weight:700'>120</div></div>",
        unsafe_allow_html=True)
with c3:
    st.markdown(
        "<div class='card'><h4 style='margin:0'>Avg Rating</h4><div style='font-size:22px;font-weight:700'>4.1</div></div>",
        unsafe_allow_html=True)
with c4:
    st.markdown(
        "<div class='card'><h4 style='margin:0'>Queries/day</h4><div style='font-size:22px;font-weight:700'>57</div></div>",
        unsafe_allow_html=True)

with tabs[1]:
    st.header("Upload Google Form (CSV / XLSX) & run a query")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Uploaded Google Form CSV or Excel", type=["csv", "xlsx"])
        st.markdown("or try an example dataset below")

        sample = pd.DataFrame({
            "Job Title": ["Data Intern", "ML Research Intern", "Business Analyst"],
            "Job Rating": [4.2, 4.6, 3.9],
            "Role_Type": ["internship", "internship", "employee"],
            "Pros": ["Remote work, good pay", "Great mentorship, remote", "Flexible hours"],
            "Cons": ["Fast-paced", "High workload", "Low pay"],
            "Company_Name": ["Alpha", "Beta", "Gamma"]
        })

        uploaded_file = BytesIO()
        sample.to_csv(uploaded_file, index=False)
        uploaded_file.seek(0)
        st.success("Example dataset loaded (temporary) - press run query")

    query = st.text_input("Job Preferences (e.g. 'remote internship pays more')", value="")
    run = st.button("Run Query")

    with col2:
        st.markdown("<div class='card'><h4>Upload tips</h4><ul><li>Use Google Form export CSV</li><li>Ensure columns: Pros, Cons, Role_Type, Job Rating</li><li>We auto-clean text</li></ul></div>", unsafe_allow_html=True)

    # precess upload
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, BytesIO) or uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded dataset -{len(df)} rows")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error("Could not read file: " + str(e))
            df = None
    else:
        df = None

    if run:
        if df is None:
            st.warning("Upload a dataset or choose the example dataset first")
        elif query.strip() == "":
            st.warning("Enter a job preference query first.")
        else:
            with st.spinner("Running recommendations"):
                time.sleep(.6)
                res = recommend_companies(df, query, top=top)
            st.markdown("### Top Recommendations")
            for r in res:
                st.markdown(f"**{r['company']}** — *{r['job_title']}*  ·  Match: `{r['similarity']}`")
                if show_explain:
                    st.markdown(f"<div class='bot'>{r['explanation']}</div>", unsafe_allow_html=True)

            sim_df = pd.DataFrame(res)
            fig = px.bar(sim_df, y="company", x="similarity", orientation="h", color="similarity", range_x=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Chat with JobMatchAI")
    if "message" not in st.session_state:
        st.session_state.message = []
    # chat display
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col1:
        message = st.text_input("Ask me about jobs, companies, or your recommendations", key="chat_input")
        if st.button("Send", key="send_btn"):
            if message.strip

