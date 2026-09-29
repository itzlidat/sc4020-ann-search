"""Product Quantization (PQ) approximate nearest-neighbour search."""

import numpy as np
import faiss


class PQ:
    """Product Quantization approximate nearest-neighbour index.

    Supports two distance computations:
    - ADC (asymmetric): the query stays exact, database vectors are codes.
      Used by search().
    - SDC (symmetric): the query is also quantised, and distances come from
      a centroid-to-centroid lookup table. Used by search_symmetric().
    """

    def __init__(self, m: int = 8, nbits: int = 8) -> None:
        self.m = m
        self.nbits = nbits
        self.index = None
        self.dim = None
        self._sdc_ready = False

    def build_index(
        self,
        vectors: np.ndarray,
        train_vectors: np.ndarray | None = None,
    ) -> None:
        """Train PQ codebooks and encode the base vectors."""

        # FAISS expects float32 arrays, shape (n, d).
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Number of dimensions in each vector.
        self.dim = vectors.shape[1]

        # PQ splits each vector into m equal sub-vectors, so d must divide by m.
        if self.dim % self.m != 0:
            raise ValueError(
                f"Vector dimension ({self.dim}) must be divisible by m ({self.m})."
            )

        # Use a separate training set if provided (e.g. sift_learn learns
        # the codebooks, sift_base is what gets compressed).
        # Otherwise, train on the base vectors.
        if train_vectors is None:
            train_vectors = vectors
        else:
            train_vectors = np.ascontiguousarray(
                train_vectors,
                dtype=np.float32,
            )

        # m sub-quantizers, each with 2^nbits centroids over d/m dimensions.
        self.index = faiss.IndexPQ(
            self.dim,
            self.m,
            self.nbits,
        )

        # k-means in each subspace to learn the codebooks.
        self.index.train(train_vectors)

        # Encode every base vector into an m-code (m bytes when nbits=8).
        self.index.add(vectors)

        # A rebuilt index needs its SDC table recomputed.
        self._sdc_ready = False

    def _prepare_query(self, query_vector: np.ndarray) -> np.ndarray:
        """Check the index exists and reshape one query to (1, d) float32."""

        if self.index is None:
            raise RuntimeError(
                "PQ index has not been built. Call build_index() before search()."
            )

        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1),
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension ({query_vector.shape[1]}) "
                f"does not match index dimension ({self.dim})."
            )

        return query_vector

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids using ADC."""

        query_vector = self._prepare_query(query_vector)

        # Make sure we are in asymmetric mode (the FAISS default).
        self.index.search_type = faiss.IndexPQ.ST_PQ

        distances, ids = self.index.search(query_vector, k)

        return ids[0].tolist()

    def search_symmetric(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids using SDC."""

        query_vector = self._prepare_query(query_vector)

        # SDC needs a table of distances between every pair of centroids in
        # each subspace. Compute it once per built index.
        if not self._sdc_ready:
            self.index.pq.compute_sdc_table()
            self._sdc_ready = True

        self.index.search_type = faiss.IndexPQ.ST_SDC
        try:
            distances, ids = self.index.search(query_vector, k)
        finally:
            # Always restore ADC so search() behaves normally afterwards.
            self.index.search_type = faiss.IndexPQ.ST_PQ

        return ids[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS PQ index for index-size measurement."""

        if self.index is None:
            raise RuntimeError(
                "PQ index has not been built. Call build_index() before save()."
            )

        faiss.write_index(self.index, path)
