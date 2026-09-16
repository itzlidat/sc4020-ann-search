"""Exact brute-force nearest neighbour search (baseline).

Computes true top-k neighbours by exhaustively comparing a query vector
against every vector in the index. Serves as the ground-truth / accuracy
baseline that every ANN method in this project is measured against.
"""

from typing import List

import numpy as np


class LinearScan:
    """Brute-force exact nearest neighbour search."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        """Store the base vectors for exhaustive search.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: implement (may just be storing `vectors` for exhaustive search)
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of the k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: implement exact distance computation + top-k selection
        raise NotImplementedError
