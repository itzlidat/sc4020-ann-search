"""HNSW (Hierarchical Navigable Small World) graph-based ANN.

Builds a multi-layer proximity graph where higher layers have fewer,
longer-range links for fast coarse navigation and lower layers have
denser, short-range links for fine-grained search. Greedy graph traversal
from the top layer down gives high-recall approximate search in
logarithmic-ish time.
"""

from typing import List

import numpy as np


class HNSW:
    """HNSW graph-based approximate nearest neighbour index."""

    def __init__(self, m: int = 16, ef_construction: int = 200, ef_search: int = 50) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build_index(self, vectors: np.ndarray) -> None:
        """Build the multi-layer HNSW graph over the base vectors.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: use hnswlib to build the graph index over `vectors`
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: greedy graph traversal via hnswlib's knn_query
        raise NotImplementedError
