"""Evaluate plot-embedding search against known remake pairs, with a TF-IDF baseline.

Run from the project root:  python -m src.evaluate
For each (original, remake) pair, the remake is the query and we find the rank
of the original among all other films (1 = most similar).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search import find_row, load_data


def rank_of(scores, query_idx, target_idx):
    """Rank (1 = best) of the target film, ignoring the query film itself."""
    scores = scores.copy()
    scores[query_idx] = -np.inf
    return int((scores > scores[target_idx]).sum()) + 1


def main():
    df, emb = load_data()
    pairs = pd.read_csv("data/remake_pairs.csv")

    # Baseline: TF-IDF word-count vectors with standard settings (not tuned).
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    X = tfidf.fit_transform(df["Plot"])

    rows = []
    for _, r in pairs.iterrows():
        o = find_row(df, r["original_title"], r["original_year"])
        m = find_row(df, r["remake_title"], r["remake_year"])
        emb_scores = emb @ emb[m]
        tfidf_scores = (X @ X[m].T).toarray().ravel()
        rows.append({
            "original": r["original_title"],
            "remake": r["remake_title"],
            "rank_embedding": rank_of(emb_scores, m, o),
            "rank_tfidf": rank_of(tfidf_scores, m, o),
        })
    res = pd.DataFrame(rows)

    metrics = []
    for name in ["embedding", "tfidf"]:
        ranks = res["rank_" + name]
        metrics.append({
            "method": name,
            "pairs": len(ranks),
            "recall@1": (ranks <= 1).mean(),
            "recall@10": (ranks <= 10).mean(),
            "recall@50": (ranks <= 50).mean(),
            "MRR": (1 / ranks).mean(),
            "median_rank": ranks.median(),
        })
    metrics = pd.DataFrame(metrics).round(3)

    os.makedirs("results", exist_ok=True)
    res.to_csv("results/pair_ranks.csv", index=False)
    metrics.to_csv("results/metrics.csv", index=False)
    print(res.to_string())
    print()
    print(metrics.to_string())
    print("films searched:", len(df))

    cols = ["recall@1", "recall@10", "recall@50"]
    x = np.arange(len(cols))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, name in enumerate(["embedding", "tfidf"]):
        vals = metrics.loc[metrics["method"] == name, cols].iloc[0].values
        ax.bar(x + (k - 0.5) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("fraction of remake pairs")
    ax.set_ylim(0, 1)
    ax.set_title("Finding the original from its remake (n=%d pairs)" % len(res))
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/recall_chart.png", dpi=150)
    print("saved results/recall_chart.png")


if __name__ == "__main__":
    main()
