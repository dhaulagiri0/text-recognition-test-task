# turns a BPEmb gensim txt file into a format suitable for tensorflow embedding layers
from bpemb import BPEmb
import numpy as np
import os

DATA_PATH = "dataset/"
TXT_REL = "mul.txt"

dirname = os.path.dirname(DATA_PATH)

# print(embeddings_index)

def create_embed_index(path):
    span = 10
    start = 1000

    embeddings_index = {}
    with open(path) as f:
        for i, line in enumerate(f):
            # ignore first line
            if i < start:
                continue
            if i != 0:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
                print(word)
            if i == start + span:
                break
    
    dim = len(embeddings_index[next(iter(embeddings_index))])

    print(f'Found {len(embeddings_index)} word vectors. Of length {dim}')
    return embeddings_index, dim

def create_maxtric(word_index, embeddings_index, embedding_dim = 300):
    # plus 2 for start and end
    num_tokens = len(word_index) + 2
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

# TODO create a function to convert sentences to BPEmbed indexes and use these to create the embedding weights


embeddings_index, dim = create_embed_index(os.path.join(dirname, TXT_REL))