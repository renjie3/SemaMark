import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Load your .npy files
# file1 = 'results/wm_emb.npy'  # Replace with the path to your first .npy file
# file2 = 'results/nowm_emb.npy' # Replace with the path to your second .npy file
file1 = 'results/wm_nering.npy'  # Replace with the path to your first .npy file
file2 = 'results/nowm_nering.npy' # Replace with the path to your second .npy file

# Load vectors from files
vectors1 = np.load(file1)
vectors2 = np.load(file2)

print(vectors1.shape)

# Ensure that both batches have the same number of vectors
assert vectors1.shape[0] == vectors2.shape[0], "The batches have a different number of vectors"

# Calculate cosine similarity for corresponding vectors
similarities = [cosine_similarity(vectors1[i], vectors2[i]) for i in range(vectors1.shape[0])]

# Print or process the cosine similarities
print(np.mean(similarities))
