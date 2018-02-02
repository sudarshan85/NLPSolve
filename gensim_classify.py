#!/usr/bin/env python

import gensim
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from get_metrics import get_metrics
from plot_functions import plot_LSA, plot_confusion_matrix


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
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
    X_train, X_test, y_train, y_test = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)
    plot_LSA(X_train, y_train)

    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train, y_train)
    y_predicted_counts = clf.predict(X_test)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    cm = confusion_matrix(y_test, y_predicted_counts)

    plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False, title='Confusion matrix')

    print(cm)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


if __name__ == '__main__':
    main()
