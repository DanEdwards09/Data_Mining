# Part 3: Text mining.
import pandas as pd
import numpy as np
from collections import Counter
import re
import urllib.request
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_csv_3(data_file):
    # Specify latin-1 encoding as instructed
    df = pd.read_csv(data_file, encoding='latin-1')
    return df

def get_sentiments(df):
    # Return list of possible sentiments
    return sorted(df['Sentiment'].unique().tolist())

def second_most_popular_sentiment(df):
    # Count sentiments and return second most common
    sentiment_counts = df['Sentiment'].value_counts()
    return sentiment_counts.index[1]

def date_most_popular_tweets(df):
    # Find date with most extremely positive tweets
    extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
    date_counts = extremely_positive['TweetAt'].value_counts()
    return date_counts.index[0]

def lower_case(df):
    # Convert tweets to lower case
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()
    return df

def remove_non_alphabetic_chars(df):
    # Replace non-alphabetic chars with whitespace
    df['OriginalTweet'] = df['OriginalTweet'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x)
    )
    return df

def remove_multiple_consecutive_whitespaces(df):
    # Replace multiple whitespaces with single whitespace
    df['OriginalTweet'] = df['OriginalTweet'].apply(
        lambda x: ' '.join(x.split())
    )
    return df

def tokenize(df):
    # Convert tweets into lists of words
    df['tokenized'] = df['OriginalTweet'].str.split()
    return df
    
def count_words_with_repetitions(tdf):
    # Count total words including repetitions
    return sum(len(tokens) for tokens in tdf['tokenized'])

def count_words_without_repetitions(tdf):
    # Count unique words across all tweets
    unique_words = set()
    for tokens in tdf['tokenized']:
        unique_words.update(tokens)
    return len(unique_words)

def frequent_words(tdf, k):
    # Get k most frequent words
    all_words = [word for tokens in tdf['tokenized'] for word in tokens]
    return [word for word, count in Counter(all_words).most_common(k)]

def remove_stop_words(tdf):
    # Download stop words
    url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    response = urllib.request.urlopen(url)
    stop_words = set(response.read().decode().splitlines())
    
    # Remove stop words and words with ≤2 characters
    tdf['tokenized'] = tdf['tokenized'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words and len(word) > 2]
    )
    return tdf

def stemming(tdf):
    # Apply Porter Stemming to each word
    stemmer = PorterStemmer()
    tdf['tokenized'] = tdf['tokenized'].apply(
        lambda tokens: [stemmer.stem(word) for word in tokens]
    )
    return tdf

def mnb_predict(df):
    vectorizer = CountVectorizer(ngram_range=(2, 5))
    X = vectorizer.fit_transform(df['OriginalTweet'])

    clf = MultinomialNB()
    clf.fit(X, df['Sentiment'])

    return clf.predict(X)

def mnb_accuracy(y_pred, y_true):
    # Calculate accuracy rounded to 3 decimal places
    return round(np.mean(y_pred == y_true), 3)