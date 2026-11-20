# Plot Twins: finding films with similar plots across Indian and Hollywood cinema

Live demo: https://movieplotsimilarity-vbkgs7qp9xphmhykgmbysi.streamlit.app/

## Problem
Many Bollywood, South Indian and Hollywood films share story ideas, including unofficial remakes. Given a film,
this project returns the films with the closest plots, and tests how well two methods recover known remakes:
pretrained sentence embeddings and a classic TF-IDF word-overlap baseline.

## Data
- Source: the Kaggle "Wikipedia Movie Plots" dataset (user jrobischon): plot summaries scraped from English Wikipedia, with an origin label per film.
  The licence of the Kaggle compilation was not verified; the text originates from English Wikipedia (CC BY-SA), credit to Wikipedia contributors.
- Subset used: American, Bollywood, Tamil, Telugu, Malayalam and Kannada films. Plots under 50 words, duplicate title+year rows, and films whose plot text is identical to another film's (dubbed versions or the same film listed in consecutive years) were removed.
- Result: 23,621 films from 1901 to 2017 (American 16,258, Bollywood 2,686, Tamil 2,333, Telugu 1,040, Malayalam 945, Kannada 359).

## Method
- Embeddings: `all-MiniLM-L6-v2` from sentence-transformers, normalised so cosine similarity is a dot product. The model reads only the first 256 tokens of each plot.
- Baseline: TF-IDF (English stop words, sublinear term frequency, min_df=2) with cosine similarity. Settings were fixed in advance and not tuned.
- Evaluation: for each known remake pair, the remake is the query and we find the rank of the original among all other films
  (1 = most similar). Metrics: Recall@k (share of pairs where the original is in the top k) and MRR (mean of 1/rank).

## Results (31 known remake pairs)
| method | pairs | recall@1 | recall@10 | recall@50 | MRR | median_rank |
|---|---|---|---|---|---|---|
| embedding | 31 | 0.194 | 0.29 | 0.419 | 0.231 | 84.0 |
| tfidf | 31 | 0.484 | 0.774 | 0.839 | 0.589 | 2.0 |

On 31 known remake pairs, plain TF-IDF word overlap ranked the original film in the top 10 for 77% of pairs, versus 29% for sentence embeddings (MRR 0.589 vs 0.231; median rank 2 vs 84).

Follow-up checks:
- Removing capitalised words (names, places) did not materially reduce TF-IDF's top-10 recall (77% to 77%), so shared names do not appear to explain its lead. The removal was crude, so this is not proof.
- Limiting TF-IDF to the first 190 words of each plot (roughly what the embedding model reads) changed its top-10 recall from 77% to 64%; this still exceeded the embeddings' 29%.
- An untuned equal-weight hybrid of both methods reached top-10 recall 77% (TF-IDF alone: 77%).

Ablations (same pairs):

| method | pairs | recall@10 | recall@50 | MRR | median_rank |
|---|---|---|---|---|---|
| embedding | 31 | 0.29 | 0.419 | 0.231 | 84.0 |
| tfidf | 31 | 0.774 | 0.839 | 0.589 | 2.0 |
| tfidf_no_names | 31 | 0.774 | 0.806 | 0.634 | 1.0 |
| hybrid | 31 | 0.774 | 0.839 | 0.584 | 2.0 |

Truncation test:

| method | pairs | recall@10 | recall@50 | MRR | median_rank |
|---|---|---|---|---|---|
| embedding | 31 | 0.29 | 0.419 | 0.231 | 84.0 |
| tfidf_full_plot | 31 | 0.774 | 0.839 | 0.589 | 2.0 |
| tfidf_first190 | 31 | 0.645 | 0.774 | 0.479 | 3.0 |

Hardest pairs for the embeddings (rank of the original, 1 = best):

| original | remake | rank_embedding | rank_tfidf |
|---|---|---|---|
| Unfaithful | Murder | 10987 | 114 |
| Mrs. Doubtfire | Avvai Shanmughi | 8271 | 2111 |
| Hitch | Partner | 8056 | 1234 |
| Reservoir Dogs | Kaante | 7836 | 104 |
| Gol Maal | Bol Bachchan | 4259 | 1792 |

The app lets you try both methods. It defaults to TF-IDF because that method performed best in this test.

## Limitations
- Only 31 remake pairs, compiled by hand from Wikipedia lists, press articles and general knowledge. Not every pair was individually verified against the films' Wikipedia pages. Small samples make the percentages unstable.
- The pairs mostly measure recovery of known remakes. Discovering unknown or unofficial similarities was explored in the demo but not evaluated.
- Remake plot summaries on Wikipedia may be written similarly to the original's, which can favour word-overlap methods.
- The embedding model truncates plots to 256 tokens; chunked embeddings and other models were not tested.
- Only films with plots in this dataset (1901 to 2017) are covered. Many films are missing.

## How to run
1. Create a virtual environment and install: `pip install pandas numpy scipy scikit-learn matplotlib sentence-transformers streamlit jupyterlab`
2. Download the Kaggle CSV to `data/wiki_movie_plots_deduped.csv` and run `notebooks/01_explore_and_clean.ipynb` to create `data/clean.csv`.
3. `python src/embed.py` creates the embeddings, then `python -m src.dedupe_plots` removes identical-plot films from the table and the embeddings.
4. `python -m src.evaluate`, `python -m src.ablation`, `python -m src.truncation_test` create the files in `results/`.
5. `python -m src.build_app_data` then `streamlit run app.py` runs the app locally.

## Repository layout
`src/` pipeline scripts, `notebooks/` exploration, `results/` metrics and chart, `app.py` and `app_data/` the Streamlit app, `data/remake_pairs.csv` the evaluation pairs, `docs/` methodology and notes.
