"""IVF (Inverted File Index).

Partitions the vector space into clusters (via k-means), assigning each
base vector to its nearest cluster centroid. At search time, only a
handful of the most relevant clusters are probed instead of the whole
dataset, trading a small amount of recall for a large speedup.
"""

from typing import List

import numpy as np


class IVF:
    """Inverted-file-index approximate nearest neighbour index."""

    def __init__(self, num_clusters: int = 100, num_probes: int = 8) -> None:
        self.num_clusters = num_clusters
        self.num_probes = num_probes

    def build_index(self, vectors: np.ndarray) -> None:
        """Cluster the base vectors and build per-cluster posting lists.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: run k-means to get centroids, assign vectors to posting lists
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: find nearest `num_probes` centroids, scan their posting lists
        raise NotImplementedError
