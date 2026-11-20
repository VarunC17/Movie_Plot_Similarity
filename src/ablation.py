"""Does TF-IDF win only because of shared names? Compare variants on the known remake pairs.

Variants: embedding, tfidf, tfidf_no_names (capitalised words removed), hybrid
(equal-weight sum of z-scored embedding and tfidf similarities, untuned).
Run from the project root:  python -m src.ablation
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import rank_of
from src.search import find_row, load_data


def zscore(x):
    return (x - x.mean()) / x.std()


def main():
    df, emb = load_data()
    pairs = pd.read_csv("data/remake_pairs.csv")

    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(df["Plot"])

    # Crude name removal: drop every capitalised word (names, places, sentence starts).
    no_names = df["Plot"].str.replace(r"\b[A-Z][A-Za-z]*\b", " ", regex=True)
    Xn = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(no_names)

    ranks = {"embedding": [], "tfidf": [], "tfidf_no_names": [], "hybrid": []}
    for _, r in pairs.iterrows():
        o = find_row(df, r["original_title"], r["original_year"])
        m = find_row(df, r["remake_title"], r["remake_year"])
        e = emb @ emb[m]
        t = (X @ X[m].T).toarray().ravel()
        tn = (Xn @ Xn[m].T).toarray().ravel()
        h = zscore(e) + zscore(t)
        for name, s in [("embedding", e), ("tfidf", t), ("tfidf_no_names", tn), ("hybrid", h)]:
            ranks[name].append(rank_of(s, m, o))

    rows = []
    for name, rk in ranks.items():
        rk = np.array(rk)
        rows.append({
            "method": name,
            "pairs": len(rk),
            "recall@10": (rk <= 10).mean(),
            "recall@50": (rk <= 50).mean(),
            "MRR": (1 / rk).mean(),
            "median_rank": float(np.median(rk)),
        })
    out = pd.DataFrame(rows).round(3)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/ablation_metrics.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
