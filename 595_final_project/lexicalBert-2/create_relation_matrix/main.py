from tqdm import tqdm
import numpy as np
from nltk.corpus import wordnet as wn
import os
import pickle
import torch


def synset_to_embedding(synset, synset_to_idx, embedding_matrix):
    idx = synset_to_idx[synset.name()]
    return embedding_matrix[idx]


def stack_embedding(embedding):
    return np.stack([embedding] * 4, axis=0)


if __name__ == '__main__':
    synset = wn.synsets("dog")[0]
    synset_to_idx = pickle.load(open("synset_to_idx.p", "rb"))
    embedding_matrix = np.load("/home/zhangge/EWISE/conve/saved_embeddings/embeddings.npz")["embeddings"]
    embedding = synset_to_embedding(synset, synset_to_idx, embedding_matrix)
    print(stack_embedding(embedding).shape)
