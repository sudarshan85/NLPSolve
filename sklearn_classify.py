#!/usr/bin/env python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from get_stats import get_metrics
from plot_functions import plot_important_words, plot_LSA, plot_confusion_matrix


def sklearn_vectorize(train_data, test_data, vectorizer):
    train = vectorizer.fit_transform(train_data)
    test = vectorizer.transform(test_data)

    return train, test, vectorizer


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


def classify(X_train, X_test, y_train, y_test, vectorizer):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train, y_train)
    y_predicted_counts = clf.predict(X_test)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    cm = confusion_matrix(y_test, y_predicted_counts)
    top_scores, top_words, bottom_scores, bottom_words = get_relevant_features(vectorizer, clf, n=10)

    plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False, title='Confusion matrix')
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

    print(cm)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


def cv_classify(X_train_raw, X_test_raw, y_train, y_test):
    X_train, X_test, vectorizer = sklearn_vectorize(X_train_raw, X_test_raw, CountVectorizer())
    plot_LSA(X_train, y_train)
    classify(X_train, X_test, y_train, y_test, vectorizer)


def tfidf_classify(X_train_raw, X_test_raw, y_train, y_test):
    X_train, X_test, vectorizer = sklearn_vectorize(X_train_raw, X_test_raw, TfidfVectorizer())
    plot_LSA(X_train, y_train)
    classify(X_train, X_test, y_train, y_test, vectorizer)


if __name__ == '__main__':
    questions = pd.read_pickle('ready_data.pkl')
    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                                random_state=40)

    cv_classify(X_train_raw, X_test_raw, y_train, y_test)
    tfidf_classify(X_train_raw, X_test_raw, y_train, y_test)
