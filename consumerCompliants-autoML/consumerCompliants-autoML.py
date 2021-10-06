# Test Classification
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re, nltk, h2o, ssl
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML

# Initialize h2o cluster
h2o.init()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

df = pd.read_csv('/Users/karthikeyangurusamy/PycharmProjects/nlpProjects/consumerCompliants-autoML/data/consumer_compliants.csv',
                 sep=',', quotechar='"')
print(df.head())
print(df.columns)
print(df['Product'].value_counts()) # Data set is imbalanced

df['Product'].value_counts().plot(kind='bar')
df['Company'].value_counts()

complaints_df = df[['Consumer complaint narrative','Product', 'Company']].rename(
    columns={'Consumer complaint narrative':'Compliant'})

complaints_df.head(5)
print(complaints_df['Product'].unique())

target = {'Debt collection':0, 'Credit card or prepaid card':1, 'Mortgage':2,
          'Checking or savings account':3, 'Student loan':4, 'Vehicle loan or lease':5}

complaints_df['target'] = complaints_df['Product'].map(target)
print(complaints_df['target'])

X_train, X_test = train_test_split(complaints_df, test_size=0.2, random_state=111)

X_train.shape
X_test.shape

stemmer = nltk.stem.SnowballStemmer('english') # Another stemmer option
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 0
                                                            and len(re.sub('\d+', '', word.strip('Xx/'))) > 3)]
    tokens = map(str.lower, tokens)
    stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stems

vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=.75, max_features=1000, lowercase=False,
                                ngram_range=(1,2))
train_vectors = vectorizer_tf.fit_transform(X_train.Compliant) # TFIDF generates a sparse vector
test_vectors = vectorizer_tf.fit_transform(X_test.Compliant)

train_vectors.A

vectorizer_tf.get_feature_names()

train_df = pd.DataFrame(train_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
train_df = pd.concat([train_df, X_train['target'].reset_index(drop=True)], axis=1)
train_df.head(5)

test_df = pd.DataFrame(test_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
test_df = pd.concat([test_df, X_test['target'].reset_index(drop=True)], axis=1)
test_df.head(5)

h2o_train_df = h2o.H2OFrame(train_df)
h2o_test_df = h2o.H2OFrame(test_df)

# covert target columns to enumeration.  h2o will consider int values in target as regression, in order to consider it
# as classification problem, we should convert it to enumeration
h2o_train_df['target'] = h2o_train_df['target'].asfactor()
h2o_test_df['target'] = h2o_test_df['target'].asfactor()

# Build automl model
# Excluding stacked ensemble algorithms - means stacking one over other alogrithm combination
# setting nfolds = 0, as i dont need cross validation
# setting balance_classes = True, indicating the data set is imbalanced and needs to balance while running algorithms
# setting max_after_balance_size = 0.3 indicating that majority data for each class should be less than 30%.
aml = H2OAutoML(max_models=10, seed=10, exclude_algos=['StackedEnsemble'], verbosity="info", nfolds=0,
                balance_classes=True, max_after_balance_size=0.3)

x = vectorizer_tf.get_feature_names()
y = 'target'

aml.train(x=x, y=y, training_frame=h2o_train_df, validation_frame=h2o_test_df)

aml.leaderboard

pred = aml.leader.predict(h2o_test_df)

aml.leader.model_performance(h2o_test_df)

# Get the parameters of the leader model
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
out = h2o.get_model([mid for mid in model_ids if 'XGBoost' in mid][0])
print(out)
out.convert_H2OXGBoostParams_2_XGBoostParams()

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.5, max_depth=6, objective='multi:softprob',
                            random_state=10, min_child_weight=3.0)
xgb_clf.fit(train_vectors, X_train['target'])
predictions = xgb_clf.predict(test_vectors)
cm = confusion_matrix(X_test['target'], predictions)
print(cm)

print('classification repot : \n', classification_report(X_test['target'], predictions))

# handle imbalanced parameter.  Assign weights for different classes.  More number of records less weight
from sklearn.utils import class_weight
import numpy as np
class_weights = list(class_weight.compute_class_weight('balanced', np.unique(X_train['target']), X_train['target']))
print(class_weights)

weights = np.ones(X_train.shape[0], dtype='float')
for i,val in enumerate(X_train['target']):
    weights[i] = class_weights[val]

xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, objective='multi:softprob',
                            random_state=10)
# Applying the weights while training
xgb_clf.fit(train_vectors, X_train['target'], sample_weight=weights)
predictions = xgb_clf.predict(test_vectors)
cm = confusion_matrix(X_test['target'], predictions)
print(cm)

print('classification repot : \n', classification_report(X_test['target'], predictions))


