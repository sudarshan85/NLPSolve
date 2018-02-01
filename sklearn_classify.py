#!/usr/bin/env python

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from plot_functions import *

def cv(data):
    count_vec = CountVectorizer()
    emb = count_vec.fit_transform(data)

    return emb, count_vec




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


def get_most_important_features(vectorizer, model, n=5):
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
    return classes

def main():
    questions = pd.read_pickle('ready_data.pkl')

    list_corpus = questions['text'].tolist()
    list_labels = questions['class_label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
    X_train_counts, count_vec = cv(X_train)
    X_test_counts = count_vec.transform(X_test)

    # plot_LSA(X_train_counts, y_train)

    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_test_counts)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    cm = confusion_matrix(y_test, y_predicted_counts)

    plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False, title='Confusion matrix')
    print(cm)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

if __name__ == '__main__':
    main()