import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the raw "allx" features (The features of the training + non-test nodes)
with open('./data/Cora/Cora/raw/ind.cora.allx', 'rb') as f:
    raw_features_matrix = pickle.load(f, encoding='latin1')

# 2. Convert the SciPy sparse matrix to a dense array for counting
raw_dense = raw_features_matrix.toarray()

# 3. Count how many unique words (1s) are in each paper
# Every row is a paper, every column is a word
words_per_paper = raw_dense.sum(axis=1)

print(f"Total Papers inspected: {len(words_per_paper)}")
print(f"Max unique words in a single paper: {words_per_paper.max()}")
print(f"Min unique words in a single paper: {words_per_paper.min()}")
print(f"Average unique words per paper: {words_per_paper.mean():.2f}")

# 4. Visualize the raw word count distribution
plt.figure(figsize=(10, 5))
plt.hist(words_per_paper, bins=30, color='indigo', edgecolor='white')
plt.title("Raw Word Count Distribution (Unprocessed Data)", fontsize=14)
plt.xlabel("Number of Unique Words Present", fontsize=12)
plt.ylabel("Number of Papers", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()