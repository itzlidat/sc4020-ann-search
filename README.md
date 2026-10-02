# SC4020 ANN Search Comparison

NTU SC4020 group coursework project comparing six approximate nearest
neighbour (ANN) search methods against an exact baseline:

- **Linear scan** — exact brute-force search (baseline)
- **LSH** — locality-sensitive hashing via random projections
- **PQ** — product quantization
- **IVF** — inverted file index
- **IVF+PQ** — hybrid inverted file index with product-quantized codes
- **HNSW** — hierarchical navigable small world graphs
- **Annoy** — random projection trees

Methods are evaluated on two datasets:

- **SIFT1M** — 1M 128-d SIFT descriptors, a standard ANN benchmark.
- **Wikipedia sentence embeddings** — sentence embeddings generated with
  `sentence-transformers` over a Wikipedia text corpus.

Each method is compared on recall@k, query latency, and index size using
the shared evaluation harness in `eval/harness.py`.

## Project structure

```
sc4020-ann-search/
├── data/           # datasets (not committed, see below)
│   ├── sift/       # SIFT1M .fvecs/.ivecs files
│   └── wiki/       # Wikipedia embedding .npy files
├── methods/        # one file per ANN method, each with build_index()/search() stubs
├── eval/           # shared evaluation utilities (recall, timing, index size)
├── results/        # output plots and CSVs
├── notebooks/      # exploratory Jupyter notebooks
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Dataset files (SIFT1M vectors and Wikipedia sentence embeddings) are
**not committed** to this repo — they're shared separately among the
group.

- Download link: **TODO — add shared drive/cloud link here**

Place the files in this layout before running any method:

```
data/
├── sift/
│   ├── sift_base.fvecs
│   ├── sift_query.fvecs
│   ├── sift_learn.fvecs
│   └── sift_groundtruth.ivecs
└── wiki/
    ├── wiki_base_embeddings.npy
    ├── wiki_query_embeddings.npy
    ├── wiki_train_embeddings.npy
    └── wiki_sentences.json
```

`eval/run_PQ_IVF.py` also caches Wikipedia ground truth as
`data/wiki/wiki_groundtruth_l2norm.npy` on first run.

## Contributing a method

Each file in `methods/` is independent — implement `build_index()` and
`search()` in your assigned file without touching anyone else's. Use
`eval/harness.py` for ground truth, recall, timing, and index size so
results are comparable across methods.
