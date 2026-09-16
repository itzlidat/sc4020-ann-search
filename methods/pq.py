"""Product Quantization (PQ).

Splits each vector into subvectors, quantizes each subspace independently
with its own small codebook (via k-means), and represents vectors as
compact codes. Enables fast approximate distance computation and large
memory savings versus storing full-precision vectors.
"""

from typing import List

import numpy as np


class PQ:
    """Product-quantization approximate nearest neighbour index."""

    def __init__(self, num_subvectors: int = 8, num_centroids: int = 256) -> None:
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids

    def build_index(self, vectors: np.ndarray) -> None:
        """Train per-subspace codebooks and encode the base vectors.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: split into subvectors, run k-means per subspace, store codes
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: build distance lookup tables per subspace, sum, top-k
        raise NotImplementedError
