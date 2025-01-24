import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GloVe vectors
glove_file = "glove.6B.50d.txt"  # Pre-trained file (50 dimensions)
embeddings = {}
with open(glove_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector

# Example: Get vector for a word
#print("Vector for 'dog':", embeddings.get("dog"))

vec_king = embeddings.get('king')
vec_queen = embeddings.get('queen')

simmilarity = cosine_similarity([vec_king],[vec_queen])
print(print("Similarity between 'king' and 'queen':", simmilarity[0][0]))
