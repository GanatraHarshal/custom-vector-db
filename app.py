import sys
import textwrap
import numpy as np

class ApproximateVectorIndex:
    def __init__(self, vectors, n_list=100, max_iter=20, seed=42):
        np.random.seed(seed)
        self.vectors = vectors.copy()
        self.n_list = n_list
        num_vectors = vectors.shape[0]
        
        self.centers = vectors[np.random.choice(num_vectors, n_list, replace=False)].copy()
        
        for _ in range(max_iter):
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
            
        self.ivf = {i: list(np.where(assignments == i)[0]) for i in range(n_list)}

    def insert(self, vector):
        idx = self.vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vector])
        dists = np.sum((self.centers - vector) ** 2, axis=1)
        cluster_idx = np.argmin(dists)
        self.ivf[cluster_idx].append(idx)

    def search(self, query_vector, k=5, nprobe=1):
        center_dists = np.sum((self.centers - query_vector) ** 2, axis=1)
        probe_clusters = np.argsort(center_dists)[:nprobe]
        
        candidate_indices = np.concatenate([self.ivf[c] for c in probe_clusters]).astype(int)
        if len(candidate_indices) == 0:
            return np.array([]), np.array([])
            
        candidate_vectors = self.vectors[candidate_indices]
        distances = np.sum((candidate_vectors - query_vector) ** 2, axis=1)
        
        top_k = min(k, len(candidate_indices))
        local_indices = np.argsort(distances)[:top_k]
        
        return candidate_indices[local_indices], distances[local_indices]


def generate_embedding(text, dim=128):
    seed = sum(ord(c) for c in text)
    np.random.seed(seed)
    return np.random.randn(dim)


def main():
    print("\n[SYSTEM] Initializing Approximate Vector Index (50,000 vectors)...")
    np.random.seed(42)
    dim = 128
    centers = np.random.randn(100, dim)
    assignments = np.random.randint(0, 100, 50000)
    initial_vectors = centers[assignments] + np.random.randn(50000, dim) * 0.1
    
    index = ApproximateVectorIndex(initial_vectors, n_list=100)
    print("[SYSTEM] Ready. Type a statement to search, 'insert <text>', 'delete', or 'exit'.\n")
    
    while True:
        try:
            cmd = input("vector-db> ").strip()
            if not cmd:
                continue
                
            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            
            if action in ("exit", "quit"):
                break
                
            elif action == "insert":
                if len(parts) < 2:
                    print("  Usage: insert <text>\n")
                    continue
                vec = generate_embedding(parts[1], dim)
                index.insert(vec)
                print(f"  [+] Inserted successfully. Total vectors: {index.vectors.shape[0]}\n")
                
            elif action == "delete":
                reason = (
                    "  [-] Deletion rejected. In an Inverted File (IVF) index, performing an in-place delete is "
                    "computationally prohibitive. Our IVF structure relies on contiguous NumPy arrays for L2 "
                    "distance vectorization and static memory pointers within the adjacency lists. Deleting a "
                    "vector requires an O(N) reallocation of the underlying dense matrix, shifting all subsequent "
                    "indices, which then forces a complete O(N) traversal and rebuild of the IVF adjacency lists "
                    "to remap the shifted pointers. Additionally, frequent deletions degrade cluster centroid "
                    "validity, necessitating a full K-means retraining cycle to prevent precision decay."
                )
                print(textwrap.fill(reason, width=90) + "\n")
                
            else:
                vec = generate_embedding(cmd, dim)
                indices, dists = index.search(vec, k=1)
                if len(indices) > 0:
                    print(f"  [>] Closest Match - ID: {indices[0]} | Distance: {dists[0]:.4f}\n")
                else:
                    print("  [!] No matches found.\n")
                    
        except (KeyboardInterrupt, EOFError):
            print()
            break

if __name__ == "__main__":
    main()
