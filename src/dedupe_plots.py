"""Remove films whose plot text is identical to another film's (dubbed versions, same film listed twice).

Films used in data/remake_pairs.csv are always kept. Filters data/clean.csv and data/embeddings.npy
together so row i still matches embedding i. Backups are saved once as *_before_plot_dedup.*.
Run from the project root after src/embed.py:  python -m src.dedupe_plots
"""
import os
import shutil

import numpy as np
import pandas as pd

from src.search import find_row, load_data


def main():
    df, emb = load_data()
    pairs = pd.read_csv("data/remake_pairs.csv")

    protected = set()
    for _, r in pairs.iterrows():
        protected.add(int(find_row(df, r["original_title"], r["original_year"])))
        protected.add(int(find_row(df, r["remake_title"], r["remake_year"])))

    norm = df["Plot"].str.strip().str.lower().tolist()
    seen = set()
    keep = []
    for i in range(len(df)):
        if i in protected:
            keep.append(i)
            seen.add(norm[i])
    for i in range(len(df)):
        if i in protected or norm[i] in seen:
            continue
        seen.add(norm[i])
        keep.append(i)
    keep = sorted(keep)
    keep_set = set(keep)
    removed = [i for i in range(len(df)) if i not in keep_set]

    print("films before:", len(df), "| removed:", len(removed), "| after:", len(keep))
    print(df.loc[removed, ["Title", "Release Year", "Origin/Ethnicity"]].head(15).to_string(index=False))

    if not os.path.exists("data/clean_before_plot_dedup.csv"):
        shutil.copy2("data/clean.csv", "data/clean_before_plot_dedup.csv")
        shutil.copy2("data/embeddings.npy", "data/embeddings_before_plot_dedup.npy")

    new_df = df.iloc[keep].reset_index(drop=True)
    new_emb = emb[keep]
    new_df.to_csv("data/clean.csv", index=False)
    np.save("data/embeddings.npy", new_emb)

    # Every evaluation pair must still be found exactly once
    for _, r in pairs.iterrows():
        find_row(new_df, r["original_title"], r["original_year"])
        find_row(new_df, r["remake_title"], r["remake_year"])
    print("all", len(pairs), "remake pairs still present; new shape:", new_df.shape, new_emb.shape)


if __name__ == "__main__":
    main()
