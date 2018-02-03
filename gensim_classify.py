#!/usr/bin/env python

import gensim
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from get_stats import get_metrics, get_statistical_explanation
from plot_functions import plot_LSA, plot_confusion_matrix, plot_important_words

word2vec_path = '/mnt/Data/DL_datasets/word-vectors/gensim-saved/GoogleNews-vectors-negative300.bin'
word2vec = gensim.models.KeyedVectors.load(word2vec_path, mmap='r')


def get_average_word2vec(tokens_list, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [word2vec[word] if word in word2vec else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [word2vec[word] if word in word2vec else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)

    return averaged


def word2vec_pipeline(examples):
    tokenized_list = []
    for example in examples:
        example_tokens = [token for token in gensim.utils.tokenize(example)]
        vectorized_example = get_average_word2vec(example_tokens, generate_missing=False, k=300)
        tokenized_list.append(vectorized_example)

    return clf.predict_proba(tokenized_list)


def plot_important_words_with_lime():
    label_to_text = {0: 'Irrelevant', 1: 'Relevant', 2: 'Unsure'}
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                                            random_state=40)

    sorted_contributions = get_statistical_explanation(clf, X_test_data, 10, word2vec_pipeline, label_to_text)

    # First index is the class (Disaster)
    # Second index is 0 for detractors, 1 for supporters
    # Third is how many words we sample
    top_words = sorted_contributions['Relevant']['supporters'][:10].index.tolist()
    top_scores = sorted_contributions['Relevant']['supporters'][:10].tolist()
    bottom_words = sorted_contributions['Relevant']['detractors'][:10].index.tolist()
    bottom_scores = sorted_contributions['Relevant']['detractors'][:10].tolist()

    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")


if __name__ == '__main__':
    questions = pd.read_pickle('ready_data.pkl')
    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()

    tokenized_corpus = [[tokens for tokens in gensim.utils.tokenize(text)] for text in list_corpus]
    embeddings = [get_average_word2vec(tokens) for tokens in tokenized_corpus]
    X_train, X_test, y_train, y_test = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)
    plot_LSA(X_train, y_train)

    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1,
                             random_state=40)
    clf.fit(X_train, y_train)
    y_predicted_counts = clf.predict(X_test)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    cm = confusion_matrix(y_test, y_predicted_counts)

    plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False, title='Confusion matrix')
    plot_important_words_with_lime(clf, list_corpus, list_labels)

    print(cm)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
