"""Inverted File Index (IVF) approximate nearest-neighbour search."""

import numpy as np
import faiss


class IVF:
    """Inverted File Index approximate nearest-neighbour index."""

    def __init__(
        self,
        nlist: int = 100,
        nprobe: int = 8,
    ) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.dim = None

    def build_index(
        self,
        vectors: np.ndarray,
        train_vectors: np.ndarray | None = None,
    ) -> None:
        """Train IVF coarse centroids and add the base vectors."""

        # FAISS expects float32 arrays.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Number of dimensions in each vector.
        self.dim = vectors.shape[1]

        # Use a separate training set if provided.
        # Otherwise, train IVF using the base vectors.
        if train_vectors is None:
            train_vectors = vectors
        else:
            train_vectors = np.ascontiguousarray(
                train_vectors,
                dtype=np.float32,
            )

        # IVF needs a coarse quantizer to represent the cluster centroids.
        # IndexFlatL2 means we use exact L2 distance when comparing
        # vectors/query to the coarse centroids.
        quantizer = faiss.IndexFlatL2(self.dim)

        # Create IVF-Flat index.
        # nlist = number of coarse clusters/cells.
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dim,
            self.nlist,
            faiss.METRIC_L2,
        )

        # Learn the nlist coarse centroids using the training vectors.
        self.index.train(train_vectors)

        # Add the base vectors.
        # FAISS assigns each vector to its nearest coarse centroid/cell.
        self.index.add(vectors)

        # Number of cells searched for each query.
        self.index.nprobe = self.nprobe

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids."""

        if self.index is None:
            raise RuntimeError(
                "IVF index has not been built. Call build_index() before search()."
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

        # Search only the nprobe nearest IVF cells.
        distances, ids = self.index.search(query_vector, k)

        return ids[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS IVF index for index-size measurement."""

        if self.index is None:
            raise RuntimeError(
                "IVF index has not been built. Call build_index() before save()."
            )

        faiss.write_index(self.index, path)
