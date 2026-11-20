# Resume bullet options (all numbers come from results files)

1. Built and deployed an end-to-end movie plot similarity search (Python, pandas, scikit-learn, sentence-transformers, Streamlit) over 23,621 Wikipedia film plots spanning Bollywood, Tamil, Telugu, Malayalam, Kannada and American cinema.
2. Designed a retrieval evaluation on 31 known remake pairs (Recall@k, MRR) comparing sentence embeddings with a TF-IDF baseline. On 31 known remake pairs, plain TF-IDF word overlap ranked the original film in the top 10 for 77% of pairs, versus 29% for sentence embeddings (MRR 0.589 vs 0.231; median rank 2 vs 84).
3. Ran controlled ablations (name removal, truncation to 190 words, equal-weight hybrid) to test why sentence embeddings underperformed or outperformed word overlap, and reported the limits of a small evaluation set.
4. Shipped a reproducible pipeline: scripted embedding, evaluation and app-data builds, version-controlled on GitHub and deployed on Streamlit Community Cloud.

Use the numbers exactly as written, and only claim what the README says.
