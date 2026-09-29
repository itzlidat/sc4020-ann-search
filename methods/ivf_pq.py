"""IVF + Product Quantization (IVF-PQ) approximate nearest-neighbour search."""

import numpy as np
import faiss


class IVFPQ:
    """IVF-PQ approximate nearest-neighbour index.

    IVF narrows the search to nprobe cells, and PQ stores each vector as a
    short code of the residual (vector minus its cell centroid).
    """

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
        """Train coarse centroids + PQ codebooks, then add the base vectors."""

        # FAISS expects float32 arrays.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Number of dimensions in each vector.
        self.dim = vectors.shape[1]

        # PQ splits each vector into m equal sub-vectors, so d must divide by m.
        if self.dim % self.m != 0:
            raise ValueError(
                f"Vector dimension ({self.dim}) must be divisible by m ({self.m})."
            )

        # Use a separate training set if provided (e.g. sift_learn).
        # Otherwise, train on the base vectors.
        if train_vectors is None:
            train_vectors = vectors
        else:
            train_vectors = np.ascontiguousarray(
                train_vectors,
                dtype=np.float32,
            )

        # Coarse quantizer: exact L2 search over the nlist centroids.
        quantizer = faiss.IndexFlatL2(self.dim)

        # IVF-PQ index.
        # nlist = number of coarse cells
        # m     = number of PQ sub-quantizers (bytes per vector when nbits=8)
        # nbits = bits per sub-quantizer code (2^nbits centroids each)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.nlist,
            self.m,
            self.nbits,
        )

        # Learn the coarse centroids, then the PQ codebooks on the residuals.
        self.index.train(train_vectors)

        # Assign each base vector to a cell and store its PQ code.
        self.index.add(vectors)

        # Number of cells searched for each query.
        self.index.nprobe = self.nprobe

    def set_nprobe(self, nprobe: int) -> None:
        """Change nprobe on a built index (no rebuild needed for sweeps)."""
        self.nprobe = nprobe
        if self.index is not None:
            self.index.nprobe = nprobe

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids."""

        if self.index is None:
            raise RuntimeError(
                "IVF-PQ index has not been built. Call build_index() before search()."
            )

        # Convert one query from shape (d,) to (1, d).
        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1),
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension ({query_vector.shape[1]}) "
                f"does not match index dimension ({self.dim})."
            )

        # Search the nprobe nearest cells using PQ distances.
        distances, ids = self.index.search(query_vector, k)

        return ids[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS IVF-PQ index for index-size measurement."""

        if self.index is None:
            raise RuntimeError(
                "IVF-PQ index has not been built. Call build_index() before save()."
            )

        faiss.write_index(self.index, path)
