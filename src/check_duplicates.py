"""Find films whose plot text is identical to another film's, and test how they affect the evaluation.

Run from the project root:  python -m src.check_duplicates
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import rank_of
from src.search import find_row, load_data


def main():
    df, emb = load_data()
    norm = df["Plot"].str.strip().str.lower()

    dup_mask = norm.duplicated(keep=False)
    dups = df[dup_mask].copy()
    dups["plot_start"] = dups["Plot"].str[:60]
    dups = dups.sort_values("plot_start")
    dups[["Title", "Release Year", "Origin/Ethnicity", "plot_start"]].to_csv("results/duplicate_plots.csv", index=False)
    print("films sharing an identical plot with another film:", int(dup_mask.sum()))
    print(dups[["Title", "Release Year", "Origin/Ethnicity"]].head(20).to_string(index=False))

    ci = pd.read_csv("results/cross_industry.csv")
    print("\nBollywood films whose best American match has embedding score >= 0.999:", int((ci["emb_score"] >= 0.999).sum()))

    pairs = pd.read_csv("data/remake_pairs.csv")
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(df["Plot"])
    rows = []
    for _, r in pairs.iterrows():
        o = find_row(df, r["original_title"], r["original_year"])
        m = find_row(df, r["remake_title"], r["remake_year"])
        rows.append({
            "original": r["original_title"],
            "remake": r["remake_title"],
            "identical_plot": bool(norm.iloc[o] == norm.iloc[m]),
            "emb_cosine": round(float(emb[o] @ emb[m]), 3),
            "rank_embedding": rank_of(emb @ emb[m], m, o),
            "rank_tfidf": rank_of((X @ X[m].T).toarray().ravel(), m, o),
        })
    res = pd.DataFrame(rows)
    print("\nPairs whose original and remake have identical plot text:", int(res["identical_plot"].sum()))
    print(res[res["identical_plot"]].to_string(index=False))

    def metrics(sub, label):
        out = []
        for name in ["embedding", "tfidf"]:
            rk = sub["rank_" + name].to_numpy()
            out.append({
                "set": label,
                "method": name,
                "pairs": len(rk),
                "recall@10": round(float((rk <= 10).mean()), 3),
                "MRR": round(float((1 / rk).mean()), 3),
                "median_rank": float(np.median(rk)),
            })
        return out

    table = pd.DataFrame(metrics(res, "all pairs") + metrics(res[~res["identical_plot"]], "excluding identical-plot pairs"))
    table.to_csv("results/metrics_excluding_identical_plots.csv", index=False)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
