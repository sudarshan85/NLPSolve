#!/usr/bin/env python

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

def cv(data):
    count_vec = CountVectorizer()
    emb = count_vec.fit_transform(data)

    return emb, count_vec

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

def main():
    questions = pd.read_pickle('ready_data.pkl')
    tweet_lengths = [len(tokens) for tokens in questions['tokens']]
    all_words = [word for tokens in questions['tokens'] for word in tokens]
    VOCAB = sorted(list(set(all_words)))

    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
    X_train_counts, count_vec = cv(X_train)
    X_test_counts = count_vec.transform(X_test)

    fig = plt.figure(figsize=(16, 16))
    plot_LSA(X_train_counts, y_train)
    plt.show()

    # print("{} words total, with a vocabulary size of {}".format(len(all_words), len(VOCAB)))
    # print("Max tweet length is {}".format(max(tweet_lengths)))

    # fig = plt.figure(figsize=(10, 10))
    # plt.xlabel('Tweet lengths')
    # plt.ylabel('Number of tweets')
    # plt.hist(tweet_lengths)
    # plt.show()

if __name__ == '__main__':
    main()