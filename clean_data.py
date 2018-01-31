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

def main():
    questions = get_df('socialmedia_relevant_cols.csv')
    print(questions.head())
    print(questions.shape)

if __name__=='__main__':
    main()
