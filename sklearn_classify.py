#!/usr/bin/env python

import gensim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from plot_functions import *


def sklearn_vectorize(train_data, test_data, vectorizer):
    train = vectorizer.fit_transform(train_data)
    test = vectorizer.transform(test_data)

    return train, test, vectorizer


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1


def get_relevant_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }

    top_scores = [a[0] for a in classes[1]['tops']]
    top_words = [a[1] for a in classes[1]['tops']]
    bottom_scores = [a[0] for a in classes[1]['bottom']]
    bottom_words = [a[1] for a in classes[1]['bottom']]

    return top_scores, top_words, bottom_scores, bottom_words

def plot_results(cm, vectorizer, clf, n_features=10):

    top_scores, top_words, bottom_scores, bottom_words = get_relevant_features(vectorizer, clf, n=n_features)

    plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False, title='Confusion matrix')
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

def gensim_w2v():
    pass
    # for i, text in enumerate(list_corpus):
    #     list_corpus[i] = [tokens for tokens in gensim.utils.tokenize(text)]
    # return [tokens for tokens in gensim.utils.tokenize(text)]
    # word2vec_path = '/mnt/Data/DL_datasets/word-vectors/GoogleNews-vectors-negative300.bin.gz'
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

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

def get_word2vec_embeddings(list_corpus, generate_missing=False):
    word2vec_path = '/mnt/Data/DL_datasets/word-vectors/GoogleNews-vectors-negative300.bin.gz'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    for i, text in enumerate(list_corpus):
        list_corpus[i] = [tokens for tokens in gensim.utils.tokenize(text)]

    embeddings = []
    for text in list_corpus:
        embeddings.append(get_average_word2vec(text, word2vec))

    return list(embeddings)

def main():
    questions = pd.read_pickle('ready_data.pkl')
    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()
    # word2vec = None
    #
    #
    #
    # questions['tokens'] = questions['text'].apply(lambda x: [tokens for tokens in gensim.utils.tokenize(x)])
    embeddings = get_word2vec_embeddings(list_corpus)
    # print(questions.head())
    # print(questions.tail())
    # X_train_raw, X_test_raw, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
    # gensim_w2v(list_corpus, list_labels)
    # X_train, X_test, vectorizer = vectorize_data(X_train_raw, X_test_raw, CountVectorizer())
    # X_train, X_test, vectorizer = sklearn_vectorize(X_train_raw, X_test_raw, TfidfVectorizer())
    #
    # plot_LSA(X_train, y_train)
    #
    # clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
    #                          multi_class='multinomial', n_jobs=-1, random_state=40)
    # clf.fit(X_train, y_train)
    # y_predicted_counts = clf.predict(X_test)
    #
    # accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    # cm = confusion_matrix(y_test, y_predicted_counts)
    #
    # plot_results(X_train, y_train, cm, vectorizer, clf, n_features=10)
    # print(cm)
    # print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


if __name__ == '__main__':
    main()
