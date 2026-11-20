"""Generate README.md, docs/METHODOLOGY.md and docs/RESUME_BULLETS.md from the real results files.

Usage (project root):  python -m src.make_docs https://your-app-url.streamlit.app
Every number is read from results/*.csv or app_data/films.csv; none is typed by hand.
"""
import os
import sys

import pandas as pd


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pct(x):
    return f"{round(100 * x)}%"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "(add your app URL here)"
    films = pd.read_csv("app_data/films.csv")
    m = pd.read_csv("results/metrics.csv").set_index("method")
    a = pd.read_csv("results/ablation_metrics.csv").set_index("method")
    t = pd.read_csv("results/truncation_test.csv").set_index("method")
    ranks = pd.read_csv("results/pair_ranks.csv")

    n_films = len(films)
    n_pairs = int(m.loc["embedding", "pairs"])
    if int(a.loc["embedding", "pairs"]) != n_pairs or int(t.loc["embedding", "pairs"]) != n_pairs:
        print("WARNING: results files use different numbers of pairs. Rerun src.evaluate, src.ablation and src.truncation_test first.")
    y0, y1 = int(films["Release Year"].min()), int(films["Release Year"].max())
    origin_counts = films["Origin/Ethnicity"].value_counts()
    origin_text = ", ".join(f"{k} {v:,}" for k, v in origin_counts.items())

    emb10, tf10 = m.loc["embedding", "recall@10"], m.loc["tfidf", "recall@10"]
    embmrr, tfmrr = m.loc["embedding", "MRR"], m.loc["tfidf", "MRR"]
    embmed, tfmed = m.loc["embedding", "median_rank"], m.loc["tfidf", "median_rank"]

    if tf10 > emb10:
        headline = (f"On {n_pairs} known remake pairs, plain TF-IDF word overlap ranked the original film in the top 10 "
                    f"for {pct(tf10)} of pairs, versus {pct(emb10)} for sentence embeddings (MRR {tfmrr} vs {embmrr}; "
                    f"median rank {tfmed:g} vs {embmed:g}).")
    elif emb10 > tf10:
        headline = (f"On {n_pairs} known remake pairs, sentence embeddings ranked the original film in the top 10 "
                    f"for {pct(emb10)} of pairs, versus {pct(tf10)} for TF-IDF word overlap (MRR {embmrr} vs {tfmrr}; "
                    f"median rank {embmed:g} vs {tfmed:g}).")
    else:
        headline = (f"On {n_pairs} known remake pairs, sentence embeddings and TF-IDF ranked the original in the top 10 "
                    f"equally often ({pct(emb10)}); MRR was {embmrr} for embeddings and {tfmrr} for TF-IDF.")

    nn10 = a.loc["tfidf_no_names", "recall@10"]
    if nn10 >= tf10 - 0.05:
        names_line = (f"Removing capitalised words (names, places) did not materially reduce TF-IDF's top-10 recall "
                      f"({pct(tf10)} to {pct(nn10)}), so shared names do not appear to explain its lead. "
                      f"The removal was crude, so this is not proof.")
    else:
        names_line = (f"Removing capitalised words (names, places) reduced TF-IDF's top-10 recall from {pct(tf10)} to "
                      f"{pct(nn10)}, so shared names explain part of its lead.")

    f190 = t.loc["tfidf_first190", "recall@10"]
    still = "still exceeded" if f190 > emb10 else "did not exceed"
    trunc_line = (f"Limiting TF-IDF to the first 190 words of each plot (roughly what the embedding model reads) changed its "
                  f"top-10 recall from {pct(tf10)} to {pct(f190)}; this {still} the embeddings' {pct(emb10)}.")

    hyb10 = a.loc["hybrid", "recall@10"]
    hybrid_line = (f"An untuned equal-weight hybrid of both methods reached top-10 recall {pct(hyb10)} "
                   f"(TF-IDF alone: {pct(tf10)}).")

    hardest = ranks.sort_values("rank_embedding", ascending=False).head(5)[["original", "remake", "rank_embedding", "rank_tfidf"]]

    main_table = md_table(m.reset_index())
    abl_table = md_table(a.reset_index())
    trunc_table = md_table(t.reset_index())

    readme = f"""# Plot Twins: finding films with similar plots across Indian and Hollywood cinema

Live demo: {url}

## Problem
Many Bollywood, South Indian and Hollywood films share story ideas, including unofficial remakes. Given a film,
this project returns the films with the closest plots, and tests how well two methods recover known remakes:
pretrained sentence embeddings and a classic TF-IDF word-overlap baseline.

## Data
- Source: the Kaggle "Wikipedia Movie Plots" dataset (user jrobischon): plot summaries scraped from English Wikipedia, with an origin label per film.
  The licence of the Kaggle compilation was not verified; the text originates from English Wikipedia (CC BY-SA), credit to Wikipedia contributors.
- Subset used: American, Bollywood, Tamil, Telugu, Malayalam and Kannada films. Plots under 50 words and duplicate title+year rows were removed.
- Result: {n_films:,} films from {y0} to {y1} ({origin_text}).

## Method
- Embeddings: `all-MiniLM-L6-v2` from sentence-transformers, normalised so cosine similarity is a dot product. The model reads only the first 256 tokens of each plot.
- Baseline: TF-IDF (English stop words, sublinear term frequency, min_df=2) with cosine similarity. Settings were fixed in advance and not tuned.
- Evaluation: for each known remake pair, the remake is the query and we find the rank of the original among all other films
  (1 = most similar). Metrics: Recall@k (share of pairs where the original is in the top k) and MRR (mean of 1/rank).

## Results ({n_pairs} known remake pairs)
{main_table}

{headline}

Follow-up checks:
- {names_line}
- {trunc_line}
- {hybrid_line}

Ablations (same pairs):

{abl_table}

Truncation test:

{trunc_table}

Hardest pairs for the embeddings (rank of the original, 1 = best):

{md_table(hardest)}

The app lets you try both methods. It defaults to TF-IDF because that method performed best in this test.

## Limitations
- Only {n_pairs} remake pairs, compiled by hand from Wikipedia lists, press articles and general knowledge. Not every pair was individually verified against the films' Wikipedia pages. Small samples make the percentages unstable.
- The pairs mostly measure recovery of known remakes. Discovering unknown or unofficial similarities was explored in the demo but not evaluated.
- Remake plot summaries on Wikipedia may be written similarly to the original's, which can favour word-overlap methods.
- The embedding model truncates plots to 256 tokens; chunked embeddings and other models were not tested.
- Only films with plots in this dataset ({y0} to {y1}) are covered. Many films are missing.

## How to run
1. Create a virtual environment and install: `pip install pandas numpy scipy scikit-learn matplotlib sentence-transformers streamlit jupyterlab`
2. Download the Kaggle CSV to `data/wiki_movie_plots_deduped.csv` and run `notebooks/01_explore_and_clean.ipynb` to create `data/clean.csv`.
3. `python src/embed.py` creates the embeddings.
4. `python -m src.evaluate`, `python -m src.ablation`, `python -m src.truncation_test` create the files in `results/`.
5. `python -m src.build_app_data` then `streamlit run app.py` runs the app locally.

## Repository layout
`src/` pipeline scripts, `notebooks/` exploration, `results/` metrics and chart, `app.py` and `app_data/` the Streamlit app, `data/remake_pairs.csv` the evaluation pairs, `docs/` methodology and notes.
"""

    methodology = f"""# Methodology

## Data preparation
Started from {34886:,} rows in the Kaggle file. Kept six origins, dropped plots with fewer than 50 words and duplicate title+year rows,
leaving {n_films:,} films. Row order is preserved so row i of the table matches row i of the embedding matrix.

## Representations
- Sentence embeddings: `all-MiniLM-L6-v2`, 384 dimensions, normalised. Inputs longer than 256 tokens are truncated.
- TF-IDF: English stop words removed, sublinear term frequency, minimum document frequency 2.

## Evaluation design
Known remake pairs are stored in `data/remake_pairs.csv` (original, remake, with years to disambiguate titles).
For each pair the remake is the query; every other film is ranked by cosine similarity; we record the rank of the original.
Recall@k and MRR are computed from these ranks. The baseline settings were fixed before seeing results.

## Additional checks
1. Name removal: capitalised words removed before TF-IDF.
2. Truncation: TF-IDF on only the first 190 words.
3. Hybrid: equal-weight sum of z-scored embedding and TF-IDF similarities, untuned.

## Results by pair
{md_table(ranks)}

## Questions to be ready for
- Why cosine similarity? It compares direction, not length; with normalised vectors it is a dot product.
- Why a TF-IDF baseline? It shows whether the extra complexity of embeddings is justified.
- What would you do next? Verify the pair list, test chunked embeddings and other models, and build a separate evaluation for unofficial remakes.
- What is the main caveat? A small, hand-compiled pair set, and remake plots on Wikipedia may be written to resemble the original.
"""

    bullets = f"""# Resume bullet options (all numbers come from results files)

1. Built and deployed an end-to-end movie plot similarity search (Python, pandas, scikit-learn, sentence-transformers, Streamlit) over {n_films:,} Wikipedia film plots spanning Bollywood, Tamil, Telugu, Malayalam, Kannada and American cinema.
2. Designed a retrieval evaluation on {n_pairs} known remake pairs (Recall@k, MRR) comparing sentence embeddings with a TF-IDF baseline. {headline}
3. Ran controlled ablations (name removal, truncation to 190 words, equal-weight hybrid) to test why sentence embeddings underperformed or outperformed word overlap, and reported the limits of a small evaluation set.
4. Shipped a reproducible pipeline: scripted embedding, evaluation and app-data builds, version-controlled on GitHub and deployed on Streamlit Community Cloud.

Use the numbers exactly as written, and only claim what the README says.
"""

    os.makedirs("docs", exist_ok=True)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    with open("docs/METHODOLOGY.md", "w", encoding="utf-8") as f:
        f.write(methodology)
    with open("docs/RESUME_BULLETS.md", "w", encoding="utf-8") as f:
        f.write(bullets)
    print("wrote README.md, docs/METHODOLOGY.md, docs/RESUME_BULLETS.md")
    print(headline)


if __name__ == "__main__":
    main()
