"""Nearest-neighbour search over film plot embeddings."""
import numpy as np
import pandas as pd


def load_data(clean_path="data/clean.csv", emb_path="data/embeddings.npy"):
    """Load the films table and its embeddings; fail loudly if they are out of sync."""
    df = pd.read_csv(clean_path)
    emb = np.load(emb_path)
    assert len(df) == emb.shape[0], "films and embeddings are out of sync"
    return df, emb


def find_row(df, title, year):
    """Return the row number of the film with this exact title and year."""
    hits = df.index[(df["Title"] == title) & (df["Release Year"] == year)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one match for {title} ({year}), found {len(hits)}")
    return hits[0]


def similar(df, emb, title, year, k=10, origins=None):
    """Top-k films whose plots are closest to the given film.

    origins: optional list such as ["American"] to only return films from those origins.
    """
    i = find_row(df, title, year)
    scores = emb @ emb[i]      # cosine similarity to every film (vectors are normalised)
    scores[i] = -1             # never return the film itself
    order = np.argsort(-scores)  # best match first
    if origins is not None:
        order = [j for j in order if df.at[j, "Origin/Ethnicity"] in origins]
    top = list(order[:k])
    out = df.loc[top, ["Title", "Release Year", "Origin/Ethnicity"]].copy()
    out["similarity"] = scores[top].round(3)
    return out.reset_index(drop=True)
