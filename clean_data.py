#!/usr/bin/env python

import pandas as pd

def extract_data(data):
    x = data.loc[:, ['text', 'choose_one', 'choose_one:confidence']]
    x.rename(columns={'choose_one:confidence': 'class_label'}, inplace=True)
    x.loc[x.choose_one == 'Relevant', 'class_label'] = 1
    x.loc[x.choose_one == 'Not Relevant', 'class_label'] = 0

    return x

def main():
    full_data = pd.read_csv('disaster_tweets.csv', encoding='ISO-8859-1')
    extracted_data = extract_data(full_data)
    extracted_data.to_pickle('cleaned_df.pkl')

if __name__=='__main__':
    main()