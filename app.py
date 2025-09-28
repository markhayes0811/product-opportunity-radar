
import streamlit as st, pandas as pd
from pathlib import Path
import plotly.express as px, subprocess, sys

DATA_DIR = Path(__file__).resolve().parent / "data"
st.set_page_config(page_title="Product Opportunity Radar", layout="wide")
st.title("📈 Product Opportunity Radar")
st.write("Demo with synthetic data. Replace CSVs in /data to use your own.")

def ensure_pipeline():
    if not (DATA_DIR/"opportunities.csv").exists():
        subprocess.check_call([sys.executable, str(Path(__file__).resolve().parent/"src"/"opportunity_pipeline.py")])
ensure_pipeline()

@st.cache_data
def load():
    return (pd.read_csv(DATA_DIR/"opportunities.csv"),
            pd.read_csv(DATA_DIR/"transactions.csv", parse_dates=["date"]),
            pd.read_csv(DATA_DIR/"search_logs.csv", parse_dates=["ts"]))

opp, tx, searches = load()

st.subheader("Top Opportunities")
st.dataframe(opp.sort_values("opportunity_score", ascending=False), use_container_width=True)

fig = px.bar(opp.sort_values("opportunity_score", ascending=False), x="category", y="opportunity_score", title="Opportunity Score by Category")
st.plotly_chart(fig, use_container_width=True)
