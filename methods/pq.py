
# PQ Structure 
# │
# ├── __init__()
# │     store parameters
# │
# ├── build_index()
# │     train PQ
# │     compress database vectors
# │     store index
# │
# └── search()
#       receive one query
#       approximate nearest-neighbour search
#       return IDs


"""Product Quantization (PQ) approximate nearest-neighbour search."""

import numpy as np
import faiss


class PQ:
    """Product Quantization approximate nearest-neighbour index."""

    def __init__(self, m: int = 8, nbits: int = 8) -> None:
        self.m = m
        self.nbits = nbits
        self.index = None
        self.dim = None

    def build_index(
        self,
        vectors: np.ndarray,
        train_vectors: np.ndarray | None = None,
    ) -> None:
        """Train PQ codebooks and encode the base vectors."""
    
        # FAISS works with float32 arrays
        vectors = np.ascontiguousarray(vectors, dtype=np.float32) #Converts vectors to float32 (e.g. (10000,128) means 10000 vectors 128 dimensions)
    
        # Extract number of dimensions in each original vector
        self.dim = vectors.shape[1]
    
        # Checks if the dimensions are divisible by m number of subspaces, if does not, then returns error value
        if self.dim % self.m != 0:
            raise ValueError(
                f"Vector dimension ({self.dim}) must be divisible by m ({self.m})."
            )
    
        # In SIFT1M, sift_learn are the train_vectors whcih learns the PQ centroids
        # In SIFT1M, sift_base contains the database vectors that are compressed using the learned PQ centroids.
        if train_vectors is None:
            train_vectors = vectors
        else:
            train_vectors = np.ascontiguousarray(
                train_vectors,
                dtype=np.float32,
            )
    
        # Create the FAISS Product Quantization index.
        # self.dim is the total Dimension D 
        # self.m is the subspaces 
        # self.nbits is the bits resulting in number of centroids (2^self.nbits) 
        self.index = faiss.IndexPQ(
            self.dim,
            self.m,
            self.nbits,
        )
    
        # Learn the PQ codebooks / centroids.
        # FAISS takes each training vector and splits them into m subspaces with D/m dimensions each, forming the subvectors
        # k-means performed in each subspace to generate (2^self.nbits) centroids 
        self.index.train(train_vectors)
    
        # Compress and store all database vectors using the learned codebooks.
        self.index.add(vectors)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[int]:
        """Return approximate k nearest-neighbour ids."""
    
        # Make sure the PQ index has already been built.
        if self.index is None:
            raise RuntimeError(
                "PQ index has not been built. Call build_index() before search()."
            )
    
        # Convert the query to float32 and reshape it from (dimension,) to (1, dimension),
        # because FAISS expects a 2D array of query vectors.
        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1),
            dtype=np.float32,
        )
    
        # # Check that the query has the same number of dimensions
        # # as the database vectors used to build the PQ index.
        # if query_vector.shape[1] != self.dim:
        #     raise ValueError(
        #         f"Query dimension ({query_vector.shape[1]}) "
        #         f"does not match index dimension ({self.dim})."
        #     )
    
        # Search the PQ-compressed database for the approximate top-k neighbours.
        distances, ids = self.index.search(query_vector, k)
    
        # ids has shape (1, k), so return the first row as a Python list.
        return ids[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS index for index-size measurement."""
    
        if self.index is None:
            raise RuntimeError(
                "PQ index has not been built. Call build_index() before save()."
            )
    
        faiss.write_index(self.index, path) # writes the entire FAISS PQ index to a file.
