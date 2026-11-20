"""Plot Twins: a Streamlit app that finds films with similar plots."""
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

st.set_page_config(page_title="Plot Twins", layout="wide")


@st.cache_resource
def load():
    films = pd.read_csv("app_data/films.csv")
    emb = np.load("app_data/embeddings.npy")
    tfidf = sparse.load_npz("app_data/tfidf.npz").tocsr()
    return films, emb, tfidf


films, emb, tfidf = load()


def label(i):
    r = films.iloc[i]
    return f"{r['Title']} ({r['Release Year']}, {r['Origin/Ethnicity']})"


st.title("Plot Twins")
st.write("Find films with similar plots across Bollywood, South Indian cinema and Hollywood.")

method = st.radio(
    "How to compare plots",
    ["Word overlap (TF-IDF)", "Meaning (sentence embeddings)"],
    horizontal=True,
)
query = st.text_input("Type part of a film title", "Sethu")
matches = []
if query:
    matches = films.index[films["Title"].str.contains(query, case=False, regex=False)].tolist()[:200]
if not matches:
    st.info("No film found. Try another title.")
    st.stop()

choice = st.selectbox("Pick the film", matches, format_func=label)
origins = sorted(films["Origin/Ethnicity"].unique())
chosen = st.multiselect("Show results from", origins, default=origins)
k = st.slider("Number of results", 5, 25, 10)

if method.startswith("Word"):
    scores = (tfidf @ tfidf[choice].T).toarray().ravel()
else:
    scores = emb @ emb[choice]
scores = scores.astype(float)
scores[choice] = -np.inf
allowed = films["Origin/Ethnicity"].isin(chosen).to_numpy()
scores[~allowed] = -np.inf
top = [j for j in np.argsort(-scores)[:k] if np.isfinite(scores[j])]

result = films.loc[top, ["Title", "Release Year", "Origin/Ethnicity", "Wiki Page"]].copy()
result["Similarity"] = scores[top].round(3)
st.dataframe(
    result,
    hide_index=True,
    column_config={"Wiki Page": st.column_config.LinkColumn("Wikipedia", display_text="open")},
)

with st.expander("How well does this work?"):
    if os.path.exists("results/metrics.csv"):
        st.dataframe(pd.read_csv("results/metrics.csv"), hide_index=True)
        st.write(
            "Test: each known remake was used as the query, and the table shows how often "
            "its original appeared among the top results. The pair list is small and partly "
            "compiled by hand, so treat this as a rough check, not a benchmark."
        )
    else:
        st.write("Evaluation results not found.")

st.caption(
    "Plot text originates from English Wikipedia (CC BY-SA), via a Kaggle compilation. "
    "Portfolio project."
)
