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
├── methods/        # one file per ANN method, each with build_index()/search()
├── eval/           # shared evaluation utilities (recall, timing, index size)
├── scripts/        # data generation (Wikipedia embeddings)
├── results/        # results JSONs and plots
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Apple Silicon note:** `annoy` is compiled from source on install, and
with recent Apple clang its default `-ffast-math` build returns wrong
neighbours (recall@10 ≈ 0). Rebuild it without that flag:

```bash
ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-fpermissive,-O3,-std=c++14,-DANNOYLIB_MULTITHREADED_BUILD" \
  pip install --no-cache-dir --force-reinstall --no-deps annoy
```

## Data

Dataset files (SIFT1M vectors and Wikipedia sentence embeddings) are
**not included** in this repo or submission — they're provided
separately.

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

The Wikipedia sentences come from the Hugging Face dataset
`sentence-transformers/wikipedia-en-sentences`, embedded with
`all-MiniLM-L6-v2` (see `config` in `wiki_sentences.json`).

## Reproducing the benchmark

The results in `results/` come from one machine with every method
single-threaded, so query times are comparable. In code: FAISS uses
`faiss.omp_set_num_threads(1)`, HNSW `set_num_threads(1)`, and the Annoy
build `n_jobs=1`. NumPy's BLAS (linear scan, LSH) is limited through the
environment, so run each script with:

```bash
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python methods/linear_scan.py
python methods/lsh.py
python eval/run_PQ_IVF.py
python methods/hnsw.py
python methods/annoy_method.py
python results/annoy_default_search_k.py      # Annoy at its default search_k
python results/annoy_metric_comparison.py     # Annoy metric ablation
python results/plot_combined.py               # combined plots + summary table
```

Supporting analyses used in the report (run with the same exports):

```bash
python results/kmeans_seed_sensitivity.py     # IVF/PQ/IVF+PQ recall across k-means seeds
python results/wiki_query_cases.py            # per-query success/failure cases (Wikipedia)
python results/sift_cosine_gt_overlap.py      # cosine vs euclidean ground truth on SIFT1M
python results/timing_controlled.py           # 5 interleaved re-timing rounds, Wikipedia
python results/timing_controlled.py --sift    # the same for SIFT1M
```

The re-timings build HNSW and Annoy once (single-threaded, about 10-15
minutes) and cache them in the system temp folder; delete
`$TMPDIR/ann_timing_cache` afterwards. Run them plugged in with other
applications closed.

The report is generated from the results JSONs with
`python reports/build_report.py --pdf` (needs python-docx, docx2pdf and
pypdf, plus Microsoft Word for the PDF step).

### Regenerating the Wikipedia embeddings

Only needed if you don't have the shared `data/wiki/` files.
`scripts/generate_embeddings.py` samples 500k base / 1k query / 100k train
sentences (seed 42) from `sentence-transformers/wikipedia-en-sentences`,
embeds them with `all-MiniLM-L6-v2`, and writes the four files into
`data/wiki/`. Its extra dependencies (including PyTorch) live in a separate
file so the main install stays light:

```bash
pip install -r requirements.txt -r requirements-embeddings.txt
python scripts/generate_embeddings.py
```

It needs a CUDA GPU as written (`device="cuda"`): about 3 minutes on an
RTX 2060 Super. On a machine without one, change `device` to `"cpu"`; it
works but takes much longer.

## Contributing a method

Each file in `methods/` is independent — implement `build_index()` and
`search()` in your assigned file without touching anyone else's. Use
`eval/harness.py` for ground truth, recall, timing, and index size so
results are comparable across methods.
