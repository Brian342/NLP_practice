# import packages
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
import pickle
from PIL import Image
from io import BytesIO
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_data():
    with open("Best_model.pkl", 'rb') as f:
        data = pickle.load(f)
        return data


model = load_model()
df_transform = load_data()


def recommend_companies(df_transform, user_query, top=5):
    """
    :param top:
    :param df_transform:
    :param df: uploaded job
    :param user_query: string
    :return: list of dicts with keys: company, job_title, similarity, explanation
    """
    query_emb = model.encode(user_query, convert_to_tensor=True)
    company_emb = model.encode(df_transform["combined_text"].tolist(), convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, company_emb)[0].cpu().numpy()

    df_transform["query_similarity"] = similarities
    top_matches = df_transform.sort_values("query_similarity", ascending=False).head(top)

    print(type(top_matches))
    res = []
    for _, row in top_matches.iterrows():
        res.append({
            "Company": row["Company_Name"],
            "Rating": round(row["Cleaned Rating"], 2),
            "Similarity": round(row["query_similarity"], 3),
            "Pros": row["Pros Clean"][:150],
            "Cons": row["Cons Clean"][:150]
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

    # --- THEME TOGGLE ---
    st.markdown("### 🌓 Theme")
    st.markdown("""
    <style>
    .toggle-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255,255,255,0.04);
      padding: 8px 14px;
      border-radius: 12px;
      margin-top: 6px;
      cursor: pointer;
      font-size: 14px;
    }
    .toggle-switch {
      width: 42px;
      height: 22px;
      background: rgba(255,255,255,0.1);
      border-radius: 999px;
      position: relative;
      transition: all 0.3s ease;
    }
    .toggle-ball {
      width: 18px;
      height: 18px;
      background: white;
      border-radius: 50%;
      position: absolute;
      top: 2px;
      left: 2px;
      transition: all 0.3s ease;
    }
    [data-theme='dark'] .toggle-ball {
      transform: translateX(20px);
      background: linear-gradient(45deg, #06b6d4, #7c3aed);
    }
    </style>
    """, unsafe_allow_html=True)

    # toggle logic
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "auto"

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("☀️ Light"):
            st.session_state.theme_mode = "light"
    with colB:
        if st.button("🌙 Dark"):
            st.session_state.theme_mode = "dark"
    with colC:
        if st.button("⚙️ Auto"):
            st.session_state.theme_mode = "auto"

    if st.session_state.theme_mode == "light":
        st.markdown("<script>document.documentElement.setAttribute('data-theme', 'light');</script>",
                    unsafe_allow_html=True)
    elif st.session_state.theme_mode == "dark":
        st.markdown("<script>document.documentElement.setAttribute('data-theme', 'dark');</script>",
                    unsafe_allow_html=True)
    else:
        st.markdown(AUTO_THEME_SCRIPT, unsafe_allow_html=True)

    st.markdown("------")
    st.caption("Built w/ spacy + Sentence-Transformers . prototype")

tabs = st.tabs(["Dashboard", "Upload & Query", "Chatbot", "Insight", "Settings"])

with tabs[0]:
    st.markdown(
        """
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px'>
            <div>
                <h1 style='margin:0;background:linear-gradient(90deg,#06b6d4,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                    JobMatchAI
                </h1>
                <div style='color:gray;font-size:15px;'>AI-powered job recommender — semantic search on reviews</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Stats cards ---
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])

    with c1:
        st.markdown(
            "<div class='card'><h4 style='margin:0'>Top Match Preview</h4>"
            "<div style='color:gray;margin-top:6px'>Quick glance at what users search for</div></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            "<div class='card' style='text-align:center'><h4 style='margin:0'>Companies</h4>"
            "<div style='font-size:28px;font-weight:700;color:#06b6d4'>120</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            "<div class='card' style='text-align:center'><h4 style='margin:0'>Avg Rating</h4>"
            "<div style='font-size:28px;font-weight:700;color:#10b981'>4.1</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            "<div class='card' style='text-align:center'><h4 style='margin:0'>Queries/day</h4>"
            "<div style='font-size:28px;font-weight:700;color:#f59e0b'>57</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Top Companies Data ---
    st.subheader("Top Companies Overview")

    top_companies = (
        df_transform.groupby("Company_Name")
        .agg({
            "Cleaned Rating": "mean",
            "Employment_Type": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
            "Role_Type": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
            "Duration": "first",
            "Job Title Clean": lambda x: ', '.join(x.head(2)),
            "Pros Clean": lambda x: x.iloc[0][:100] + "..." if isinstance(x.iloc[0], str) else "",
            "Cons Clean": lambda x: x.iloc[0][:100] + "..." if isinstance(x.iloc[0], str) else ""
        })
        .reset_index()
        .sort_values(by="Cleaned Rating", ascending=False)
        .head(10)
    )

    top_companies["Cleaned Rating"] = top_companies["Cleaned Rating"].round(2)

    # Add a better column order
    top_companies = top_companies[
        ["Company_Name", "Cleaned Rating", "Employment_Type", "Role_Type", "Duration", "Job Title Clean", "Pros Clean",
         "Cons Clean"]
    ]

    # Expand table full width
    st.dataframe(
        top_companies.reset_index(drop=True),
        use_container_width=True,
        height=325,
    )

with tabs[1]:
    st.header("Upload Google Form (CSV / XLSX) & run a query")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Uploaded Google Form CSV or Excel", type=["csv", "xlsx"])
        st.markdown("or try an example dataset below")

        sample = df_transform[["Company_Name", "Cleaned Rating", "Role_Type", "Pros Clean",
                               "Cons Clean"]]

        uploaded_file = BytesIO()
        sample.to_csv(uploaded_file, index=False)
        uploaded_file.seek(0)
        st.success("Example dataset loaded (temporary) - press run query")

    query = st.text_input("Job Preferences (e.g. 'remote internship pays more')", value="")
    run = st.button("Run Query")

    with col2:
        st.markdown(
            "<div class='card'><h4>Upload tips</h4><ul><li>Use Google Form export CSV</li><li>Ensure columns: Pros, Cons, Role_Type, Job Rating</li><li>We auto-clean text</li></ul></div>",
            unsafe_allow_html=True)

    # precess upload
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, BytesIO) or uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded dataset of System has-{len(df)} rows")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error("Could not read file: " + str(e))
            df = None
    else:
        df = None

    if run:
        if query.strip() == "":
            st.warning("Enter a Job Preference First")
        else:
            with st.spinner("Finding your Best Company Matches ..."):
                time.sleep(.6)
                res = recommend_companies(df_transform, query, top=top)

            st.markdown("### Top Recommendations")
            for r in res:
                st.markdown(f"""
                        **{r['Company']}**
                        -  Rating: `{r['Rating']}`
                        -  Similarity: `{r['Similarity']}`
                        -  Pros: {r['Pros']}...
                        -  Cons: {r['Cons']}...
                        ---
                        """)

            sim_df = pd.DataFrame(res)
            fig = px.bar(
                sim_df,
                y="Company",
                x="Similarity",
                color="Similarity",
                orientation="h",
                range_x=[0, 1],
                title="Top Recommended Companies"
            )

            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    context_df = (
        df_transform.groupby("Company_Name")
        .agg({
            "Cleaned Rating": "mean",
            "Employment_Type": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
            "Role_Type": lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A",
            "Job Title Clean": lambda x: ', '.join(x.head(2))
        })
        .reset_index()
        .sort_values(by="Cleaned Rating", ascending=False)
        .head(20)
    )
    context_summary = context_df.to_string(index=False)
    st.header("Chat with JobMatchAI")

    if "message" not in st.session_state:
        st.session_state.message = [
            {"role": "assistance", "content": "Hey I'm JobMatchAI. Ask me about companies, roles or job trends"}
        ]
    # display History
    for msg in st.session_state.message:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat display
    chat_col1, chat_col2 = st.columns([3, 1])

    with chat_col1:
        if prompt := st.chat_input("Ask about job recommendations, companies, or internship advice!!"):
            st.session_state.message.append({"role": "user", "content": prompt})

            contextual_prompt = f"""
                    You are JobMatchAI, a helpful career insights assistant.

                    Here’s some recent dataset info about companies, job titles, and ratings:
                    {context_summary}

                    Now answer the following user query in a clear, concise, data-driven way:
                    {prompt}
                    """
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI Response
            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "system", "content": "You are JobMatchAI, a professional data analyst that gives job insights."},
                    {"role": "user", "content": contextual_prompt}
                ]
                )
                reply = response.choices[0].message.content
                st.markdown(reply)

                # store assistance message
            st.session_state.messages.append({"role": "assistant", "content": reply})


    with chat_col2:
        st.markdown(
            """
            <div class='card' style='padding:10px;border-radius:10px;background-color:#f7f9fc;'>
                <b>💡 Chat Tips:</b>
                <ul style='margin:5px 0 0 15px'>
                    <li>Ask for remote internships</li>
                    <li>Get company insights</li>
                    <li>Find top-rated roles</li>
                    <li>Ask for salary or rating analysis</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


with tabs[3]:
    st.header("Insights & Charts")
    st.markdown("Visualize dataset-level insights (from uploaded dataset)")

    if 'df' in locals() and df_transform is not None:
        # Rating distribution
        st.subheader("Rating Distribution")
        fig = px.histogram(df, x="Cleaned Rating", nbins=10)
        st.plotly_chart(fig, use_container_width=True)

        # Role Type counts
        st.subheader("Role Type Counts")
        # role_counts = (
        #     df['Role_Type']
        #     .value_counts()
        #     .reset_index()
        # )
        role = df_transform['Role_Type'].value_counts().reset_index()
        role.columns = ['Role_Type', 'Count']

        total_employee = role["Count"].sum()

        fig2 = px.pie(role, values="Count", names="Role_Type", color_discrete_sequence=px.colors.qualitative.Set3,
                      hole=.4)
        st.plotly_chart(fig2, use_container_width=True)



    else:
        st.info("Upload a dataset to see insights.")

with tabs[4]:
    st.header("Settings & About")
    st.markdown("Model info Configuration")
    st.markdown(
        "- Model: sentence-transformers (replace with your model)\n- Explainability: token overlap + token similarity heatmaps\n- Embeddings are recommended to be cached (FAISS/Annoy) for production")
    if st.button("Clear chat & cache"):
        st.session_state.message = []
        st.experimental_rerun()
