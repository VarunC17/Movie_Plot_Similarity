"""For every Bollywood film, find its closest American film by plot (both methods).

Run from the project root:  python -m src.cross_industry
Writes results/cross_industry.csv and prints the strongest matches.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search import load_data


def main():
    df, emb = load_data()
    b_idx = np.where((df["Origin/Ethnicity"] == "Bollywood").to_numpy())[0]
    a_idx = np.where((df["Origin/Ethnicity"] == "American").to_numpy())[0]

    # Embedding scan: every Bollywood film against every American film
    S = emb[b_idx] @ emb[a_idx].T
    best_e = S.argmax(axis=1)
    score_e = S.max(axis=1)
    del S

    # TF-IDF scan, in chunks of 200 films to limit memory use
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(df["Plot"]).tocsr()
    Xa_T = X[a_idx].T.tocsc()
    best_t = np.zeros(len(b_idx), dtype=int)
    score_t = np.zeros(len(b_idx))
    for s in range(0, len(b_idx), 200):
        block = (X[b_idx[s:s + 200]] @ Xa_T).toarray()
        best_t[s:s + 200] = block.argmax(axis=1)
        score_t[s:s + 200] = block.max(axis=1)

    out = pd.DataFrame({
        "bollywood_film": df.loc[b_idx, "Title"].to_numpy(),
        "year": df.loc[b_idx, "Release Year"].to_numpy(),
        "emb_match": df.loc[a_idx[best_e], "Title"].to_numpy(),
        "emb_match_year": df.loc[a_idx[best_e], "Release Year"].to_numpy(),
        "emb_score": score_e.round(3),
        "tfidf_match": df.loc[a_idx[best_t], "Title"].to_numpy(),
        "tfidf_match_year": df.loc[a_idx[best_t], "Release Year"].to_numpy(),
        "tfidf_score": score_t.round(3),
    })
    out["methods_agree"] = a_idx[best_e] == a_idx[best_t]
    out.to_csv("results/cross_industry.csv", index=False)

    print("Bollywood films scanned:", len(out))
    print("methods pick the same American film for:", int(out["methods_agree"].sum()))
    print("\nTop 15 where both methods agree (sorted by TF-IDF score):")
    agree = out[out["methods_agree"]].sort_values("tfidf_score", ascending=False).head(15)
    print(agree[["bollywood_film", "year", "tfidf_match", "tfidf_match_year", "tfidf_score", "emb_score"]].to_string(index=False))
    print("\nTop 10 by embedding score:")
    top_e = out.sort_values("emb_score", ascending=False).head(10)
    print(top_e[["bollywood_film", "year", "emb_match", "emb_match_year", "emb_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
