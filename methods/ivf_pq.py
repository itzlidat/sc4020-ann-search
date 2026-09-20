"""IVF + Product Quantization approximate nearest-neighbour search."""

import numpy as np
import faiss


class IVFPQ:
    """Inverted File Index with Product Quantization."""

    def __init__(
        self,
        nlist: int = 100,
        nprobe: int = 8,
        m: int = 8,
        nbits: int = 8,
    ) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.index = None
        self.dim = None

    def build_index(
        self,
        vectors: np.ndarray,
        train_vectors: np.ndarray | None = None,
    ) -> None:
        """Train IVF centroids and PQ codebooks, then encode base vectors."""

        # FAISS expects float32 arrays
        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        # Extract original vector dimension
        self.dim = vectors.shape[1]

        # PQ must be able to split the vector evenly into m subspaces
        if self.dim % self.m != 0:
            raise ValueError(
                f"Vector dimension ({self.dim}) "
                f"must be divisible by m ({self.m})."
            )

        # Use separate training vectors if provided
        if train_vectors is None:
            train_vectors = vectors
        else:
            train_vectors = np.ascontiguousarray(
                train_vectors,
                dtype=np.float32,
            )

        # Coarse quantizer used by IVF to find the nearest cells
        quantizer = faiss.IndexFlatL2(self.dim)

        # Create IVF + PQ index
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.nlist,
            self.m,
            self.nbits,
            faiss.METRIC_L2,
        )

        # Train:
        # 1. IVF coarse centroids
        # 2. PQ codebooks
        self.index.train(train_vectors)

        # Add base vectors:
        # vectors are assigned to IVF cells and stored using PQ compression
        self.index.add(vectors)

        # Number of IVF cells searched for each query
        self.index.nprobe = self.nprobe

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids."""

        if self.index is None:
            raise RuntimeError(
                "IVF+PQ index has not been built. "
                "Call build_index() before search()."
            )

        # Convert one query from shape (d,) to (1, d)
        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1),
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension ({query_vector.shape[1]}) "
                f"does not match index dimension ({self.dim})."
            )

        distances, ids = self.index.search(
            query_vector,
            k,
        )

        return ids[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS IVF+PQ index."""

        if self.index is None:
            raise RuntimeError(
                "IVF+PQ index has not been built. "
                "Call build_index() before save()."
            )

        faiss.write_index(self.index, path)
