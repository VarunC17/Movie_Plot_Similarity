"""Create the slim files the Streamlit app needs in app_data/ (no plot text is copied)."""
import os

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search import load_data


def main():
    df, emb = load_data()
    os.makedirs("app_data", exist_ok=True)
    df[["Title", "Release Year", "Origin/Ethnicity", "Wiki Page"]].to_csv("app_data/films.csv", index=False)
    np.save("app_data/embeddings.npy", emb.astype(np.float32))
    # Same TF-IDF settings as the evaluation, so the app uses the method we tested.
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(df["Plot"])
    sparse.save_npz("app_data/tfidf.npz", X.astype(np.float32))
    for f in ["films.csv", "embeddings.npy", "tfidf.npz"]:
        print(f, round(os.path.getsize(os.path.join("app_data", f)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
