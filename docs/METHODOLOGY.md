# Methodology

## Data preparation
Started from 34,886 rows in the Kaggle file. Kept six origins, dropped plots with fewer than 50 words, duplicate title+year rows and films with identical plot text (dubbed versions, the same film listed twice),
leaving 23,621 films. Row order is preserved so row i of the table matches row i of the embedding matrix.

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
| original | remake | rank_embedding | rank_tfidf |
|---|---|---|---|
| Kaakha Kaakha | Gharshana | 866 | 2 |
| Sethu | Tere Naam | 1368 | 2 |
| Pokkiri | Wanted | 319 | 3 |
| Munna Bhai M.B.B.S. | Shankar Dada MBBS | 71 | 1 |
| Kaththi | Khaidi No. 150 | 6 | 1 |
| Hello Brother | Judwaa | 1 | 1 |
| Missamma | Miss Mary | 1472 | 1 |
| Ghajini | Ghajini | 1 | 1 |
| Drishyam | Papanasam | 1 | 1 |
| Singam | Singham | 45 | 3 |
| Bodyguard | Bodyguard | 233 | 9 |
| Ready | Ready | 623 | 26 |
| Kick | Kick | 26 | 1 |
| Nayakan | Dayavan | 566 | 8 |
| Alaipayuthey | Saathiya | 84 | 20 |
| Dabangg | Gabbar Singh | 1330 | 4 |
| 3 Idiots | Nanban | 1 | 1 |
| Don | Billa | 727 | 1 |
| Don | Don: The Chase Begins Again | 36 | 1 |
| Thani Oruvan | Dhruva | 1 | 1 |
| Premam | Premam | 2206 | 2 |
| Bangalore Days | Bangalore Naatkal | 2 | 1 |
| Mrs. Doubtfire | Avvai Shanmughi | 8271 | 2111 |
| Avvai Shanmughi | Chachi 420 | 40 | 2 |
| Hitch | Partner | 8056 | 1234 |
| Reservoir Dogs | Kaante | 7836 | 104 |
| Unfaithful | Murder | 10987 | 114 |
| Gol Maal | Bol Bachchan | 4259 | 1792 |
| Himmatwala | Himmatwala | 3 | 1 |
| Agneepath | Agneepath | 1 | 1 |
| Apthamitra | Chandramukhi | 71 | 1 |

## Questions to be ready for
- Why cosine similarity? It compares direction, not length; with normalised vectors it is a dot product.
- Why a TF-IDF baseline? It shows whether the extra complexity of embeddings is justified.
- What would you do next? Verify the pair list, test chunked embeddings and other models, and build a separate evaluation for unofficial remakes.
- What is the main caveat? A small, hand-compiled pair set, and remake plots on Wikipedia may be written to resemble the original.
