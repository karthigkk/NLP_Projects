# Data Source https://www.kaggle.com/c/fake-news/data
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(dir_path + "/fakeNewsClassifier-DL/data/train.csv");

df.head()

df = df.dropna()

X = df.drop('label', axis=1)
y = df['label']

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

# Define vocabulary size
voc_size = 5000

# One hot reporesentation
message = X.copy()
message.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join((review))
    corpus.append(review)
# Convert corpus to one hot representation
onehot_rep = [one_hot(words,voc_size) for words in corpus]

# Pad it to fixed length
sent_lenght = 20
embedded_docs = pad_sequences(onehot_rep, sent_lenght,padding='pre')

# Embedding Representation
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features,input_length=sent_lenght))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state=42)

# Model Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Performance Metrics and Accuracy
y_pred = (model.predict(X_test) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)