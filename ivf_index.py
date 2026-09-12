import numpy as np

class ApproximateVectorIndex:
    def __init__(self, vectors, n_list=100, max_iter=20, seed=42):
        np.random.seed(seed)
        self.vectors = vectors
        self.n_list = n_list
        num_vectors = vectors.shape[0]
        
        self.centers = vectors[np.random.choice(num_vectors, n_list, replace=False)].copy()
        
        for _ in range(max_iter):
            # Fast vectorized L2 distance using ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            dists = (np.sum(vectors**2, axis=1, keepdims=True) + 
                     np.sum(self.centers**2, axis=1) - 
                     2 * vectors.dot(self.centers.T))
            assignments = np.argmin(dists, axis=1)
            
            new_centers = np.array([
                vectors[assignments == i].mean(axis=0) if np.any(assignments == i) else self.centers[i] 
                for i in range(n_list)
            ])
            
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
            
        self.ivf = {i: np.where(assignments == i)[0] for i in range(n_list)}
            
    def search(self, query_vector, k=5, nprobe=1):
        center_dists = np.sum((self.centers - query_vector) ** 2, axis=1)
        probe_clusters = np.argsort(center_dists)[:nprobe]
        
        candidate_indices = np.concatenate([self.ivf[c] for c in probe_clusters])
        if len(candidate_indices) == 0:
            return np.array([]), np.array([])
            
        candidate_vectors = self.vectors[candidate_indices]
        distances = np.sum((candidate_vectors - query_vector) ** 2, axis=1)
        
        top_k = min(k, len(candidate_indices))
        local_indices = np.argsort(distances)[:top_k]
        
        return candidate_indices[local_indices], distances[local_indices]
