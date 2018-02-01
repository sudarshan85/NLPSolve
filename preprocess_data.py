#!/usr/bin/env python
"""
This cleans the tweets and writes the dataframe into a pickle
file for use later
"""

import codecs
from io import StringIO

import pandas as pd
import spacy

nlp = spacy.load('en')


def tokenizer(text):
    return [token.text for token in nlp.tokenizer(text)]


def get_df(fname):
    """
    Function to read in the non-UTF8 encoded file into a StringIO buffer
    which is then read into a dataframe
    :param fname: Filename of the csv
    :return: Dataframe read from the CSV
    """
    with codecs.open(fname, 'r', encoding='utf-8', errors='replace') as inf:
        csv_strio = StringIO(inf.read())

    df = pd.read_csv(csv_strio)
    return df


def standardize_text(df, text_field):
    """
    Function to clean the tweets in the text inside the dataframe
    :param df: dataframe to be cleaned
    :param text_field: the name of the text field
    :return: Cleaned dataframe
    """
    # Remove everything (including) after http
    df[text_field] = df[text_field].str.replace(r"http\S+", "")

    # Don't know why this is here because previous regex should take care of it
    df[text_field] = df[text_field].str.replace(r"http", "")

    # remove twitter handles such as @ablaze
    df[text_field] = df[text_field].str.replace(r"@\S+", "")

    # replace all chars present in the list with a single space (help regex101.com)
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

    # again don't know why this is here, since got rid all chars (including) after @ with @\S+
    df[text_field] = df[text_field].str.replace(r"@", "at")

    # lowercase everything
    df[text_field] = df[text_field].str.lower()
    return df


def main():
    questions = get_df('socialmedia_relevant_cols.csv')
    questions = standardize_text(questions, 'text')
    questions['tokens'] = questions['text'].apply(tokenizer)
    questions.to_pickle('ready_data.pkl')


if __name__ == '__main__':
    main()
