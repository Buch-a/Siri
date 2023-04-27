import pandas as pd
import numpy as np
import nltk

nltk.download('wordnet')
from tqdm.auto import tqdm, trange

data = pd.read_csv('train_full.csv')
data.shape
data = data[:100000]
data.head(5)
# print(data['keyword'].unique(), len(data['keyword'].unique()))
# print(data['location'].unique(), len(data['location'].unique()))
topics = data['location'].unique()
news_in_cat_count = 2000
# print(topics)
df_res = pd.DataFrame()

for topic in tqdm(topics):
    df_topic = data[data['location'] == topic][:news_in_cat_count]
    df_res = pd.concat([df_res, pd.DataFrame(df_topic)], ignore_index=True)
df_res.shape
# print(df_topic)

import string


def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])


def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])


import re


def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')
lem = WordNetLemmatizer()
stops = list(stopwords.words('english'))
stops.extend(['…', '«', '»', '...', '%'])


# print(stops)

def lemmatize_text(text):
    tokens = lem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in stops and token != " "]
    text = " ".join(tokens)
    return text


preproccessing = lambda text: (remove_multiple_spaces(remove_numbers(remove_punctuation(text))))
data['preproccessed'] = pd.Series(list(map(preproccessing, df_res['text'])))
prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in tqdm(df_res['text'])]
len(prep_text)
prep_text[0]
df_res['text_prep'] = prep_text
df_res.head(1)

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stops = list(stopwords.words('english'))
stops.extend(['…', '«', '»', '...', 'etc'])
text = df_res['text_prep'][0]
word_tokenize(text)

from nltk import word_tokenize

stemmed_texts_list = []
for text in tqdm(df_res['text_prep']):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stops]
    text = " ".join(stemmed_tokens)
    stemmed_texts_list.append(text)

df_res['text_stem'] = stemmed_texts_list


def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stops and token != ' ']
    return " ".join(tokens)


sw_texts_list = []
for text in tqdm(df_res['text_prep']):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stops and token != ' ']
    text = " ".join(tokens)
    sw_texts_list.append(text)

df_res['text_sw'] = sw_texts_list
df_res['text_sw'][0]

lemm_texts_list = []
for text in tqdm(df_res['text_sw']):
    # print(text)
    try:
        text_lem = lem.lemmatize(text)
        tokens = [token for token in text_lem if token != ' ' and token not in stops]
        text = " ".join(tokens)
        lemm_texts_list.append(text)
    except Exception as e:
        print(e)

df_res['text_lemm'] = lemm_texts_list


def lemmatize_text(text):
    text_lem = lem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ']
    return " ".join(tokens)


df_res.to_csv('lemm.csv')
df_res = pd.read_csv('lemm.csv', encoding='utf-8')
df_res.head()

df_res['text_lemm'][0]
X = df_res['text_sw']
y = df_res['keyword']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
my_tags = df_res['keyword'].unique()
# print(my_tags)

import warnings

warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logreg = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                   ])

logreg.fit((X_train.astype("U").str.lower()), (y_train.astype("U").str.lower()))
y_pred = logreg.predict(X_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
