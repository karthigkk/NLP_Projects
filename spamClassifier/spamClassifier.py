# Build a model to predict if the sent sms is spam/ham.
# Data obtained from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

# Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import nltk
import ssl

# Import dataset
messages = pd.read_csv("/Users/karthikeyangurusamy/PycharmProjects/nlpProjects/spamClassifier/data/SMSSpamCollection",
                       sep="\t", names=["label", "message"])

# Data cleaning and preprocessing
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords")
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating bag of words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus)
y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# Train test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training model using Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

confusion_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
