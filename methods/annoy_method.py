"""Annoy (Approximate Nearest Neighbors Oh Yeah) random projection trees.

Builds a forest of binary trees, each recursively splitting the space
with random hyperplanes. At search time, multiple trees are traversed and
their candidate leaves merged, giving approximate nearest neighbours with
a small, disk-friendly index.
"""

from typing import List

import numpy as np


class Annoy:
    """Random-projection-tree approximate nearest neighbour index."""

    def __init__(self, num_trees: int = 10, metric: str = "angular") -> None:
        self.num_trees = num_trees
        self.metric = metric

    def build_index(self, vectors: np.ndarray) -> None:
        """Build the forest of random projection trees over the base vectors.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: implement with the annoy library
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: implement with the annoy library
        raise NotImplementedError
