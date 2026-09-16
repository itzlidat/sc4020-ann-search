"""Reader for the texmex .fvecs/.ivecs binary vector format used by SIFT1M.

Format: each vector is stored back-to-back with no file header, as
    [int32 dimension][dimension x (float32 or int32)]
repeated for every vector in the file. .fvecs stores float32 payloads
(base/query/learn sets); .ivecs stores int32 payloads (ground-truth ids).
"""

import os

import numpy as np


def load_fvecs(path: str) -> np.ndarray:
    """Load a .fvecs file of float32 vectors.

    Args:
        path: Path to a .fvecs file (e.g. sift_base.fvecs).

    Returns:
        Array of shape (n_vectors, dim) of dtype float32.
    """
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    vecs = raw.reshape(-1, dim + 1)[:, 1:].copy()
    return vecs.view(np.float32)


def load_ivecs(path: str) -> np.ndarray:
    """Load a .ivecs file of int32 vectors (e.g. ground-truth neighbour ids).

    Args:
        path: Path to a .ivecs file (e.g. sift_groundtruth.ivecs).

    Returns:
        Array of shape (n_vectors, dim) of dtype int32.
    """
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


if __name__ == "__main__":
    sift_dir = os.path.join(os.path.dirname(__file__), "sift")

    base = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    query = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    learn = load_fvecs(os.path.join(sift_dir, "sift_learn.fvecs"))
    gt = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))

    print(f"base:  {base.shape}  (expect (1000000, 128))")
    print(f"query: {query.shape}  (expect (10000, 128))")
    print(f"learn: {learn.shape}  (expect (100000, 128))")
    print(f"gt:    {gt.shape}  (expect (10000, 100))")
