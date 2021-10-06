# Sentiment Analysis using Tensorflow

import tensorflow as tf
import tensorflow_datasets as tfds
import os, datetime

print(tf.__version__)

dataset, info = tfds.load('amazon_us_reviews/Mobile_Electronics_v1_00', with_info=True)
train_dataset = dataset['train']

print(len(train_dataset))

BUFFER_SIZE = 30000
BATCH_SIZE = 128

train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

for reviews in train_dataset.take(2):
    print(reviews)

for reviews in train_dataset.take(10):
    review_text = reviews['data']
    print(review_text.get('review_body').numpy())
    print(review_text.get('star_rating').numpy())
    print(tf.where(review_text.get('star_rating')>3,1,0).numpy())
# Tokenize
tokenizer = tfds.deprecated.text.Tokenizer()
vocabulary_set = set()
for _, reviews in train_dataset.enumerate():
    review_text = reviews['data']
    print(review_text['review_body'].numpy())
    reviews_token = tokenizer.tokenize(review_text['review_body'].numpy())
    vocabulary_set.update(reviews_token)
vocab_size = len(vocabulary_set)

encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)

# visualize how to encoded sentence looks like
for reviews in train_dataset.take(5):
    review_text = reviews['data']
    print(review_text.get('review_body').numpy())
    encoded_exmaple = encoder.encode(review_text.get('review_body').numpy())
    print(encoded_exmaple)

# View the encoded value versus the word
for index in encoded_exmaple:
    print('{}  ----> {}'.format(index, encoder.decode([index])))

# Function to encode text and label
def encode(text_tensor, label_tensor):
    encoded_text = encoder.encode(text_tensor.numpy())
    label = tf.where(label_tensor > 3, 1, 0)
    return encoded_text, label

def encode_map_fn(tensor):
    text = tensor['data'].get('review_body')
    label = tensor['data'].get('star_rating')
    encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int32))
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

# Encode the train dataset
ar_encoded_data = train_dataset.map(encode_map_fn)
# View encoded values
for f0, f1  in ar_encoded_data.take(2):
    print(f0)
    print(f1)

# Split train dataset into train and test and pad it to batch_size
TAKE_SIZE = 10000
train_data = ar_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = ar_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)
# Adding 1 to vocab_size because we added the padding.  0 is a new vocab
vocab_size += 1

sample_text, sample_label = next(iter(test_data))
print(sample_text[0], sample_label[0])

# Find target distribution (how many positive and how many negative reviews in test_data)
for f0, f1 in test_data.take(10):
    print(f0, f1)
    print(tf.unique_with_counts(f1)[2].numpy())

# build a LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 128))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dense(1))


logdir = os.path.join('/Users/karthikeyangurusamy/PycharmProjects/nlpProjects/SentimentAnalysis/logs',
                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/Users/karthikeyangurusamy/PycharmProjects/nlpProjects/SentimentAnalysis/model/sentiment_analysis1.hdf5',
                                                  verbose=1, save_best_only=True)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_data, epochs=4, validation_data=test_data, callbacks=[tensorboard_callback, checkpointer])




