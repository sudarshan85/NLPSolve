#!/usr/bin/env python

import pandas as pd
import codecs
from io import StringIO

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

def get_csv():
    # Read file into a string with codecs for converting encoding
    csv_str = ""
    with codecs.open('disaster-tweets.csv', 'r', encoding='utf-8', errors='replace') as inf:
        for line in inf:
            csv_str += line

    # convert raw string int string buffer for steaming
    csv_str = StringIO(csv_str)

    return csv_str

def main():
    csv_file = get_csv()
    full_data = pd.read_csv(csv_file)
    qm = extract_data(full_data)
    print(qm.head)
    print(qm.shape)

    # extracted_data.to_pickle('cleaned-df.pkl')

if __name__=='__main__':
    main()
