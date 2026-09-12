import numpy as np

class ExactVectorIndex:
    def __init__(self, num_vectors=50000, dim=128, num_clusters=100, seed=42):
        np.random.seed(seed)
        centers = np.random.randn(num_clusters, dim)
        assignments = np.random.randint(0, num_clusters, num_vectors)
        self.vectors = centers[assignments] + np.random.randn(num_vectors, dim) * 0.1

    def search(self, query_vector, k=5):
        distances = np.sum((self.vectors - query_vector) ** 2, axis=1)
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
