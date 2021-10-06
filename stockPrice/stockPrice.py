# Stock sentiment analysis using news headlines
# Data source https://www.kaggle.com/aaron7sun/stocknews
# labels - Class 1 - Stock price increased, Class 0 - Stock price decreased

# Imports
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Import Data and split into train and test
df = pd.read_csv('/Users/karthikeyangurusamy/PycharmProjects/nlpProjects/stockPrice/data/Stock_news_data.csv',
                 encoding='ISO-8859-1')
train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']

# Removing punctuations
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Renaming columns
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index

# Convert all news data to lower case
for index in new_index:
    data[index] = data[index].str.lower()

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

# Implement bag of words
cv = CountVectorizer(ngram_range=(2, 2))
train_dataset = cv.fit_transform(headlines)

# Implement Random Forest classifier
rf = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf.fit(train_dataset, train['Label'])

# Predict for test data set
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = cv.transform(test_transform)
predictions = rf.predict(test_dataset)

# Check model accuracy
matrix = confusion_matrix(test['Label'], predictions)
score = accuracy_score(test['Label'], predictions)
report = classification_report(test['Label'], predictions)
