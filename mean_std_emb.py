import numpy as np
"""
normal
Mean L2 norm: 2145.6382
Std  L2 norm: 36.8221
"""

# Path relative to project root
embeddings_path = "outputs/embeddings_n/embs.npy"
embeddings = np.load(embeddings_path)

#  mean and std
mean_vector = np.mean(embeddings, axis=0)
std_vector = np.std(embeddings, axis=0)

# L2 norms
embedding_norms = np.linalg.norm(embeddings, axis=1)
mean_norm = np.mean(embedding_norms)
std_norm = np.std(embedding_norms)

print(f"Shape of embeddings: {embeddings.shape}")
print(f"Mean vector (first 5 dims): {mean_vector[:]}")
print()
print(f"Std vector  (first 5 dims): {std_vector[:]}")
print()
print(f"\nMean L2 norm: {mean_norm:.4f}")
print(f"Std  L2 norm: {std_norm:.4f}")
