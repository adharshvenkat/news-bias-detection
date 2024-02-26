import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_parquet('data/train.parquet').head(10000)


df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

stop_words = set(stopwords.words('english'))
# df['text'] = df['text'].apply(lambda x: x.split())
# df['text'] = df['text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
# df['text'] = df['text'].apply(lambda x: ' '.join(x))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

lemmatizer = WordNetLemmatizer()
# df['text'] = df['text'].apply(lambda x: x.split())
# df['text'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
# df['text'] = df['text'].apply(lambda x: ' '.join(x))
df['text'] = df['text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
df['text'] = df['text'].apply(lambda x: x.lower())

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf'] = list(tfidf_matrix.toarray())

df.to_parquet('data/train_tfidf.parquet')

