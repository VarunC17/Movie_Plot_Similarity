import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    df = pd.read_csv("data/clean.csv")
    plots = df["Plot"].tolist()
    if limit:
        plots = plots[:limit]
    print("films to embed:", len(plots))

    model = SentenceTransformer(MODEL_NAME)
    print("max_seq_length:", model.max_seq_length)

    start = time.time()
    embeddings = model.encode(
        plots,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    seconds = round(time.time() - start)
    print("shape:", embeddings.shape, "| seconds:", seconds)

    out_path = "data/embeddings_test.npy" if limit else "data/embeddings.npy"
    np.save(out_path, embeddings)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
