"""IVF+PQ hybrid index.

Combines IVF's coarse clustering (to limit the search to a subset of
partitions) with PQ's compact quantized codes (to make scanning each
partition cheap), giving both speed and memory efficiency.
"""

from typing import List

import numpy as np


class IVFPQ:
    """Hybrid IVF (coarse quantizer) + PQ (fine quantizer) ANN index."""

    def __init__(
        self,
        num_clusters: int = 100,
        num_probes: int = 8,
        num_subvectors: int = 8,
        num_centroids: int = 256,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_probes = num_probes
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids

    def build_index(self, vectors: np.ndarray) -> None:
        """Cluster base vectors with IVF, then PQ-encode residuals per cluster.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        # TODO: train IVF centroids, then train/apply PQ codebooks per posting list
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        # TODO: probe nearest clusters, compute PQ approximate distances, top-k
        raise NotImplementedError
