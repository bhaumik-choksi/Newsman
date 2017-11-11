# Author : Bhaumik Darshan Choksi

import json
import urllib.request
from nltk.corpus import stopwords
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter
from functools import partial

from LiveNews import LiveNews

API_KEY = "X"

if API_KEY == "X":
    print("FIRST GET YOUR OWN API KEY AT https://newsapi.org/")
    print("Set the API_KEY variable to hold the string containing this API key")
    exit()

garbage = set(stopwords.words('english'))
vectorizer = pickle.load(open("vectorizer.p", "rb"))
encoder = pickle.load(open("encoder.p", "rb"))
keywords = pickle.load(open("keywords.p", "rb"))
classifier = pickle.load(open("classifier.p", "rb"))


def normalize_text(s, keywords):
    s = s.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)
    s = s.split()
    s = list(filter((lambda x: x not in garbage), s))
    for word in s:
        keywords.append(word)
    s = " ".join(s)
    return s


def format_live_news_title(title, keywords):
    return [" ".join(list(word for word in title.split() if word in keywords))]


def find_category(title, vectorizer, encoder, keywords, classifier):
    sample = normalize_text(title, [])
    sample = format_live_news_title(sample, keywords)
    sample = vectorizer.transform(sample)
    output = classifier.predict(sample)
    return encoder.inverse_transform(output)[0]  # Single elements array, so return first elem


ln = LiveNews(API_KEY)
articles = ln.fetch(source="cnbc")
for article in articles:
    print(article["title"], " ", find_category(article["title"], vectorizer, encoder, keywords, classifier))
    print("---")

def get_news_articles(category):
    ln = LiveNews(API_KEY)  # TODO: Remove global variable
    articles = ln.fetch(source="google-news")   # TODO : Add spinner for news sources in the GUI
    for article in articles:
        article["category"] = find_category(article["title"], vectorizer, encoder, keywords, classifier)

    relevant_articles = list(filter((lambda x : category == x["category"]), articles))
    return relevant_articles

# GUI begins

master = tkinter.Tk()
master.title("Bhaumik's newsman")
master.geometry("500x500")
listbox = Listbox(master, width=70, height=20)

business_button = Button(master, text="Business", width=30, height=3, bg="cyan",
                         command=lambda: news_refresh_callback('b', listbox)).grid(row=1, column=1)
technology_button = Button(master, text="Technology", width=30, height=3, bg="green", command=lambda: news_refresh_callback('t', listbox)).grid(row=1, column=2)
entertainment_button = Button(master, text="Entertainment", width=30, height=3, bg="magenta", command=lambda: news_refresh_callback('e', listbox)).grid(row=2, column=1)
medical_button = Button(master, text="Medical", width=30, height=3, bg="yellow", command=lambda: news_refresh_callback('m', listbox)).grid(row=2, column=2)

listbox.grid(row=3, columnspan=3)

listbox.insert(END, "Pick a category!")

def news_refresh_callback(category, listbox):
    listbox.delete(0, END)
    articles = get_news_articles(category)
    if len(articles) == 0:
        listbox.insert(END, "Oops, not articles in this category")
        listbox.insert("Try another source")
        return

    for article in articles:
        listbox.insert(END, article["title"])

master.mainloop()
# GUI ends

# Author : Bhaumik Darshan Choksi