#!/usr/bin/env python
"""
Script to extract the relevant columns from the raw CSV using Pandas and storing
it in a pickle file
"""

import pandas as pd

def extract_data(data):
    """
    Extract the relevant columns from the raw dataframe
    :param data: dataframe from which to extract columns
    :return: extracted dataframe
    """
    x = data.loc[:, ['text', 'choose_one', 'choose_one:confidence']]
    x.rename(columns={'choose_one:confidence': 'class_label'}, inplace=True)
    x.loc[x.choose_one == 'Relevant', 'class_label'] = 1
    x.loc[x.choose_one == 'Not Relevant', 'class_label'] = 0

    return x

def main():
    full_data = pd.read_csv('disaster-tweets.csv', encoding='ISO-8859-1')
    questions = extract_data(full_data)
    # extracted_data.to_pickle('cleaned-df.pkl')

if __name__=='__main__':
    main()
