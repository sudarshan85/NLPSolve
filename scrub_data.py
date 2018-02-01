#!/usr/bin/env python

import pandas as pd
import codecs
from io import StringIO

def get_df(fname):
    # Read file into a string buffer with codecs for converting encoding
    with codecs.open(fname, 'r', encoding='utf-8', errors='replace') as inf:
        csv_strio = StringIO(inf.read())

    df = pd.read_csv(csv_strio)
    return df


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def main():
    questions = get_df('socialmedia_relevant_cols.csv')
    questions = standardize_text(questions, 'text')
    questions.to_pickle('scrubbed_data.pkl')

if __name__=='__main__':
    main()
