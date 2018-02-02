#!/usr/bin/env python

import gensim
import pandas as pd


def gensim_w2v():
    pass

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)

    return averaged

def get_word2vec_embeddings(list_corpus):
    word2vec_path = '/mnt/Data/DL_datasets/word-vectors/GoogleNews-vectors-negative300.bin.gz'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    for i, text in enumerate(list_corpus):
        list_corpus[i] = [tokens for tokens in gensim.utils.tokenize(text)]

    embeddings = []
    for text in list_corpus:
        embeddings.append(get_average_word2vec(text, word2vec))

    return embeddings


def main():
    questions = pd.read_pickle('ready_data.pkl')
    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()
    embeddings = get_word2vec_embeddings(list_corpus)
    print(embeddings[0])



if __name__ == '__main__':
    main()