# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:44:08 2019

@author: S534803
"""

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def get_sentiments(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def split_sentiments(sentiments):
    xs = [sent['neg'] for sent in sentiments]
    ys = [sent['neu'] for sent in sentiments]
    zs = [sent['neg'] for sent in sentiments]
    
    return xs,ys,zs