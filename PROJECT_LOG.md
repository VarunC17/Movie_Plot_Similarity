Nov 17: chose Kaggle Wikipedia Movie Plots as the primary dataset; backup is CMU.
Nov 19: sanity check, third pair: Pokkiri->Wanted 0.578. Two of the first five pairs did not match the data (spelling or not in dataset).
Nov 19: pair list now 5 pairs (added Kaththi 2014 -> Khaidi No. 150 2017; titles copied exactly from data).
Nov 20: first evaluation on 13 pairs: embedding recall@10 0.308, MRR 0.250, median rank 74; TF-IDF recall@10 0.923, MRR 0.678, median rank 1. Embeddings underperform the word-count baseline on official remakes; hypothesis: shared names favour TF-IDF. Next: name-masking test.
Nov 20: 31 pairs (13 verified-from-lists, 18 added from memory, unverified). Embedding recall@10 0.290, MRR 0.231, median rank 84; TF-IDF recall@10 0.774, MRR 0.587, median rank 2. Hollywood->Bollywood pairs (Hitch->Partner, Reservoir Dogs->Kaante, Unfaithful->Murder, Mrs. Doubtfire->Avvai Shanmughi) ranked in the thousands for embeddings. Added src/ablation.py (name-removal test + hybrid).
