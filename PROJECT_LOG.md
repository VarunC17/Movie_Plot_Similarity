Nov 17: chose Kaggle Wikipedia Movie Plots as the primary dataset; backup is CMU.
Nov 19: sanity check, third pair: Pokkiri->Wanted 0.578. Two of the first five pairs did not match the data (spelling or not in dataset).
Nov 19: pair list now 5 pairs (added Kaththi 2014 -> Khaidi No. 150 2017; titles copied exactly from data).
Nov 20: first evaluation on 13 pairs: embedding recall@10 0.308, MRR 0.250, median rank 74; TF-IDF recall@10 0.923, MRR 0.678, median rank 1. Embeddings underperform the word-count baseline on official remakes; hypothesis: shared names favour TF-IDF. Next: name-masking test.
