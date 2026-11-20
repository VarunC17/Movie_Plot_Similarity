"""Is the 256-token limit why embeddings lose? Give TF-IDF only the first 190 words too.

Run from the project root:  python -m src.truncation_test
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import rank_of
from src.search import find_row, load_data


def main():
    df, emb = load_data()
    pairs = pd.read_csv("data/remake_pairs.csv")

    def fit(texts):
        return TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(texts)

    X_full = fit(df["Plot"])
    first = df["Plot"].str.split().str[:190].str.join(" ")
    X_first = fit(first)

    ranks = {"embedding": [], "tfidf_full_plot": [], "tfidf_first190": []}
    for _, r in pairs.iterrows():
        o = find_row(df, r["original_title"], r["original_year"])
        m = find_row(df, r["remake_title"], r["remake_year"])
        scores = {
            "embedding": emb @ emb[m],
            "tfidf_full_plot": (X_full @ X_full[m].T).toarray().ravel(),
            "tfidf_first190": (X_first @ X_first[m].T).toarray().ravel(),
        }
        for name, s in scores.items():
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
    out.to_csv("results/truncation_test.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
