import streamlit as st
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util


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
)


def get_top_company():
    best = context_df.iloc[0]
    return f"The Top-Rated company is **{best["Company_Name"]}** with a rating of {best["Cleaned Rating"]:.2f}/5 rating"


def analyze_role(role):
    matches = context_df[context_df["Role_Type"].str.contains(role, case=False, na=False)]
    if matches.empty:
        return f"No roles found matching '{role}'"
    res = matches.head(5)
    out = "\n".join(
        [f"- **{r.Company_Name}** ({r.Role_Type}) — {r['Cleaned Rating']:.1f}/5" for _, r in res.iterrows()])
    return f" Top companies offering roles like **{role}**:\n{out}"


def get_company_info(name):
    match = context_df[context_df["Company_Name"].str.contains(name, case=False, na=False)]
    if match.empty:
        return f"No Company found with name '{name}'"
    row = match.iloc[0]
    return (
        f"  **{row.Company_Name}**\n"
        f"- Average Rating: {row['Cleaned Rating']:.1f}/5\n"
        f"- Common Role Type: {row['Role_Type']}\n"
        f"- Common Employment Type: {row['Employment_Type']}\n"
        f"- Popular Job Titles: {row['Job Title Clean']}"
    )


def list_commands():
    return """
          **Available Commands**
        - `top_company` → returns the best-rated company
        - `analyze_role [role]` → Find companies hiring for a specific role
        - `company_info [company]` → Get detailed info about a company
        - `help` → Show this list
        """


def handle_commands(command, args=None):
    if command == "top_company":
        return get_top_company()
    elif command == "analyze_role":
        return analyze_role(args or "")
    elif command == "company_info":
        return get_company_info(args or "")
    elif command == "help":
        return list_commands()
    else:
        return "Unknown Command. Try /help for available commands"
