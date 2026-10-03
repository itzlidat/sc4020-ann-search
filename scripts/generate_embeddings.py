import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

N_BASE = 500_000
N_QUERY = 1_000
N_TRAIN = 100_000
SEED = 42
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Outputs go to <repo root>/data/wiki/, whatever the working directory.
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "wiki"

def main():
    ds = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")
    total_needed = N_BASE + N_QUERY + N_TRAIN
    assert len(ds) >= total_needed, "Dataset smaller than requested sample sizes."

    ds = ds.shuffle(seed=SEED)
    base_ds = ds.select(range(0, N_BASE))
    query_ds = ds.select(range(N_BASE, N_BASE + N_QUERY))
    train_ds = ds.select(range(N_BASE + N_QUERY, N_BASE + N_QUERY + N_TRAIN))

    model = SentenceTransformer(MODEL_NAME, device="cuda")

    def embed(dataset, name):
        sentences = dataset["sentence"] if "sentence" in dataset.column_names else dataset[dataset.column_names[0]]
        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings, sentences

    base_emb, base_sent = embed(base_ds, "base set")
    query_emb, query_sent = embed(query_ds, "query set")
    train_emb, train_sent = embed(train_ds, "train set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "wiki_base_embeddings.npy", base_emb)
    np.save(OUT_DIR / "wiki_query_embeddings.npy", query_emb)
    np.save(OUT_DIR / "wiki_train_embeddings.npy", train_emb)

    with open(OUT_DIR / "wiki_sentences.json", "w", encoding="utf-8") as f:
        json.dump({
            "base": list(base_sent),
            "query": list(query_sent),
            "train": list(train_sent),
            "config": {
                "n_base": N_BASE, "n_query": N_QUERY, "n_train": N_TRAIN,
                "seed": SEED, "model": MODEL_NAME,
            },
        }, f)

if __name__ == "__main__":
    main()
