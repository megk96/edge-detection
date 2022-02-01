import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
from collections import Counter
import time
import matplotlib.pyplot as plt
import os

DATA_FOLDER = "../data/text_data"
FILE_NAME = "Corona_NLP_train.csv"
OUTPUT_FOLDER = "../outputs"

# Data Inspection
def question_one(df):
    # Unique values are found for the column value Sentiment
    print(pd.unique(df['Sentiment']))
    # value_counts is used to find the number of tweets for each sentiment and it is ordered decreasingly.
    print(df['Sentiment'].value_counts())
    # Tweets are first filtered by sentiment = Extremely Positive and then grouped by the date to find the date with
    # the maximum positive tweets
    date_ext_positive = df.where(df['Sentiment'] == "Extremely Positive").groupby(["TweetAt"]).count()
    print(date_ext_positive)
    print(date_ext_positive.max())
    print(date_ext_positive.idxmax())
    # Preprocessing techniques, string functions such as lower() and replace() are used
    # Replace uses regex to find non alphabetical characters and replace them with a whitespace
    # print(df["OriginalTweet"])
    df["OriginalTweet"] = df["OriginalTweet"].str.lower()
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z]', ' ')
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('\s+', ' ', regex=True)
    print(df["OriginalTweet"])

    return df

# Utility function
def get_stats(series):
    corpus = [i for series_ in series for i in series_]
    # The FreqDist function is used on the entire corpus.
    # It outputs a dictionary with terms and their frequencies.
    freq_dist = FreqDist(corpus)
    print(freq_dist)
    print("Top 10 words")
    print(freq_dist.most_common(10))

# Data Cleaning
def question_two(df):
    # The nltk library provides a list of English stopwords
    stop = stopwords.words('english')
    # The split() function is used to split and proved to be more time efficient that word_tokenize
    df["OriginalTweet"] = df["OriginalTweet"].apply(lambda x: x.split())
    print("Stats before removing any words")
    get_stats(df["OriginalTweet"])
    # Lambda functions are used to remove stop words and words less than 2 characters
    df["OriginalTweet"] = df["OriginalTweet"].apply(lambda x: [item for item in x if item not in stop])
    df["OriginalTweet"] = df["OriginalTweet"].apply(lambda x: [item for item in x if len(item) > 2])
    print("Stats after removing stopwords and words less than 2 characters long")
    get_stats(df["OriginalTweet"])

    return df

# Histogram and Document Frequency
def question_three(df):
    # The word frequencies are first found
    word_frequencies = [Counter(document) for document in df["OriginalTweet"]]
    # Using the word frequencies, the document frequencies are found
    document_frequencies = Counter()
    for word_frequency in word_frequencies:
        document_frequencies.update(word_frequency.keys())
    print("Words most commonly appearing in documents")
    print(document_frequencies.most_common(10))
    # It is stored as a list dividing by document size
    total_documents = df.shape[0]
    document_frequencies = [v / total_documents for v in dict(document_frequencies).values()]
    document_frequencies.sort(reverse=False)
    # This is used to plot a line graph
    plt.plot(document_frequencies)
    plt.xlabel("Index of unique words")
    plt.ylabel("Fraction of document frequency")
    plt.title("Doc Frequency Fraction")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "doc_frequency.jpg"))
    var = len([x for x in document_frequencies if x > 0.2])
    print(f"Number of terms greater than 20% is { var }")
    print(f"Number of terms less than 20% is {len(document_frequencies) - var}")

# Multinomial Naive Bayes
def question_four(df, tokenized=False):
    corpus = []
    if tokenized:
        for tweet in df['OriginalTweet']:
            corpus.append(' '.join(tweet))
    else:
        corpus = df['OriginalTweet'].to_numpy()
    labels = df['Sentiment'].to_numpy()
    # Feeding string into CountVectorizer
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(corpus)
    # This is then pushed to MNB
    clf = MultinomialNB()
    # Data is not traditionally split and error rate is calculated on the training data
    print(clf.fit(data, labels).score(data, labels))


def main():
    start = time.time()
    df = pd.read_csv(os.path.join(DATA_FOLDER, FILE_NAME))
    # print("Accuracy without cleaning data")
    # question_four(df)
    formatted_df = question_one(df)
    cleaned_df = question_two(formatted_df)
    question_three(cleaned_df)
    print("Accuracy after cleaning data")
    question_four(cleaned_df, tokenized=True)
    end = time.time()
    print(f"Runtime of the program is {end - start} seconds")


if __name__ == "__main__":
    main()
