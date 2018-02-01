#!/usr/bin/env python

import pandas as pd

def main():
    questions = pd.read_pickle('ready_data.pkl')
    tweet_lengths = [len(tokens) for tokens in questions['tokens']]
    all_words = [word for tokens in questions['tokens'] for word in tokens]
    VOCAB = sorted(list(set(all_words)))
    print("{} words total, with a vocabulary size of {}".format(len(all_words), len(VOCAB)))
    print("Max tweet length is {}".format(max(tweet_lengths)))
    
if __name__ == '__main__':
    main()