# turns a BPEmb gensim txt file into a format suitable for tensorflow embedding layers
from bpemb import BPEmb
import numpy as np
import os
import json
from json import JSONEncoder

DATA_PATH = "dataset/"
TXT_REL = "mul.txt"

dirname = os.path.dirname(DATA_PATH)

# print(embeddings_index)

def create_embed_index(path):

    embeddings_index = {}
    with open(path) as f:
        for i, line in enumerate(f):
            # ignore first line
            if i < 1:
                continue
            if i != 0:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
    
    dim = len(embeddings_index[next(iter(embeddings_index))])

    print(f'Found {len(embeddings_index)} word vectors. Of length {dim}')
    return embeddings_index, dim

# # the word_index in this case should be a dictionary of of BPEmb.encode subwords
# def create_maxtric(word_index, embeddings_index, embedding_dim = 300):
#     # plus 2 for start and end
#     num_tokens = len(word_index) + 2
#     hits = 0
#     misses = 0

#     # Prepare embedding matrix
#     embedding_matrix = np.zeros((num_tokens, embedding_dim))
#     for word, i in word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # Words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#             hits += 1
#         else:
#             misses += 1
#     print("Converted %d words (%d misses)" % (hits, misses))
#     return embedding_matrix

# TODO create a function to convert sentences to BPEmbed indexes and use these to create the embedding weights

def get_embedding_pair(sentence, multibpemb):
    # split a sentence into subwords and gives the ids of each subword
    subwords_ids = multibpemb.encode_ids(sentence)
    # gives (1, 300) vector representations of subwords
    embeddings = multibpemb.embed(sentence)
    return subwords_ids, embeddings

def add_to_embed_dict(embed_dict, subwords_ids, embeddings):
    for i, id in enumerate(subwords_ids):
        # only modify if word is not in dictionary
        if id not in embed_dict:
            embed_dict[id] = embeddings[i]

def add_to(dict, keys, values):
    for i, key in enumerate(keys):
        # only modify if word is not in dictionary
        if key not in dict:
            dict[key] = values[i]

def senetences_to_embed(sentences):
    multibpemb = BPEmb(lang="multi", vs=1000000, dim=300)
    # id: embedding
    embed_dict = {}
    # id: word
    word_dict = {}
    for d in sentences:
        en = d["sentence"]
        encoded_en = multibpemb.encode(en)
        mixed = d["mixed"]
        encoded_mixed = multibpemb.encode(mixed)
        chi = d["translated"]
        encoded_chi = multibpemb.encode(chi)
        en_ids, en_embed = get_embedding_pair(en, multibpemb)
        mixed_ids, mixed_embed = get_embedding_pair(mixed, multibpemb)
        chi_ids, chi_embed = get_embedding_pair(chi, multibpemb)
        add_to(embed_dict, en_ids, en_embed)
        add_to(embed_dict, mixed_ids, mixed_embed)
        add_to(embed_dict, chi_ids, chi_embed)
        
        add_to(word_dict, en_ids, encoded_en)
        add_to(word_dict, mixed_ids, encoded_mixed)
        add_to(word_dict, chi_ids, encoded_chi)
        
    return embed_dict, word_dict

def load_json_data(path="dataset/train_trans.json"):
    f = open(path)
    l = json.load(f)
    return l      

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def make_embedding(output_path = "dataset/embedding.json",
                   BPEmbPath = "dataset/", 
                   bpemb_name="multi.wiki.bpe.vs409094.d300.w2v.txt", 
                   dict_path = "dataset/vocab_bpemb.json", ):
    
    np.set_printoptions(suppress=True)
    embedding_index , _ = create_embed_index(f"{BPEmbPath}/{bpemb_name}")
    d = load_json_data(dict_path)
    embedding_matrix = {}

    for word, embedding in embedding_index.items():
        embedding_matrix[int(d[word])] = embedding

    
    with open(output_path, "w") as f:
        print(len(embedding_matrix))
        json.dump(embedding_matrix, f, cls=NumpyArrayEncoder)
        # for key, value in embedding_matrix.items():
        #     coeff = np.array2string(value).replace('\n', '')[1:-1]
        #     f.writelines(f"{key} {coeff}\n")

    
def load_embedding(path):

    embedding = {}
    with open(path) as f:
        for i, line in enumerate(f):
            id, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embedding[id] = coefs
    
    dim = len(embedding[0])
    print(f'Found {len(embedding)} word vectors. Of length {dim}')
    return embedding, dim

class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return np.array([self._decode(v) for v in o])
        else:
            return o

def load_embedding_json(path):

    embedding = {}

    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open(path) as f:
        embedding = json.load(f, cls=Decoder, object_hook=keystoint)

    return embedding, len(embedding[0])

# embeddings_index, dim = create_embed_index(os.path.join(dirname, TXT_REL))

# multibpemb = BPEmb(lang="multi", vs=1000000, dim=300)
# text = 'John F. Kennedy said "Ich bin ein Pfannkuchen". 这是一个中文句子.日本語の文章です。'
# embedding = multibpemb.embed(text)
# print(embedding, embedding.shape)

# text = '这是一个中文句子'
# embedding = multibpemb.embed(text)
# print(embedding, embedding.shape)

# text = '这是一个中文句子 test testtest'
# embedding = multibpemb.embed(text)
# subwords = multibpemb.encode_ids(text)
# print(embedding, subwords, embedding.shape)