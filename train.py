# Author : Bhaumik Darshan Choksi

from nltk.corpus import stopwords
from nltk.sentiment.util import *
from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Read data and initialize stop-words
news = pd.read_excel('./newscorpus.xlsx')
garbage = set(stopwords.words('english'))


def normalize_text(s, keywords):
    s = s.lower()

    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    s = re.sub('\s+', ' ', s)
    s = s.split()
    s = list(filter((lambda x: x not in garbage), s))
    for word in s:
        keywords.append(word)
    s = " ".join(s)
    return s


keywords = []
news['title'] = [normalize_text(str(s), keywords) for s in news['title']]
keywords = set(keywords)

news['category'] = news['category'].fillna('x')

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['title'])
# TODO: Check if TF-IDF actually helps
# x = TfidfTransformer().fit_transform(x)
encoder = LabelEncoder()
y = encoder.fit_transform(news['category'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
nb = MultinomialNB()
nb.fit(x_train, y_train)
print("Training complete")
print("Accuracy ", nb.score(x_test, y_test))

pickle.dump(vectorizer, open("vectorizer.p", "wb"))
pickle.dump(encoder, open("encoder.p", "wb"))
pickle.dump(keywords, open("keywords.p", "wb"))
pickle.dump(nb, open("classifier.p", "wb"))
print("Model saved")

# Author : Bhaumik Darshan Choksi