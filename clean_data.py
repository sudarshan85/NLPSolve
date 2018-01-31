#!/usr/bin/env python

import pandas as pd

def main():
    # tweets = []
    # with open('disaster_tweets.csv', 'r', encoding='ISO-8859-1') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         tweets.append(row)
    #
    # print(len(tweets))

    tweets = pd.read_csv('disaster_tweets.csv', encoding='ISO-8859-1')
    print(tweets.head())


if __name__=='__main__':
    main()