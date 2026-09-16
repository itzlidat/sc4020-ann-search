"""Locality-Sensitive Hashing (LSH) via random projections.

Hashes vectors into buckets using random hyperplane projections so that
similar vectors are likely to collide in the same bucket, giving
sub-linear approximate nearest neighbour search.
"""

from typing import List

import numpy as np


class LSH:
    """Random-projection LSH approximate nearest neighbour index."""

    def __init__(self, num_tables: int = 4, num_bits: int = 16) -> None:
        self.num_tables = num_tables
        self.num_bits = num_bits

    def build_index(self, vectors: np.ndarray) -> None:
        """Hash all base vectors into LSH buckets.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: generate random hyperplanes and bucket `vectors` per table
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: hash query, gather candidates from matching buckets, rerank
        raise NotImplementedError
