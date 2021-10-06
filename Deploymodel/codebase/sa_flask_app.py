import tensorflow as tf
import tensorflow_datasets as tfds
import os
from flask import Flask, jsonify, make_response, request
from healthcheck import HealthCheck
import logging

app = Flask(__name__)
padding_size = 1000
fileDir = os.path.dirname(os.path.realpath('__file__'))
text_encoder = tfds.deprecated.text.TokenTextEncoder.load_from_file(os.path.join(fileDir, '../model/sa_encoder.vocab.tokens'))
model = tf.keras.models.load_model(os.path.join(fileDir, '../model/sentiment_analysis_tf.hdf5'))

logging.basicConfig(filename='flask.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.info('Model and vocabulary loaded......')

healthcheck = HealthCheck(app, "/hcheck")

def howamidoing():
    return True, "I am up and running"

healthcheck.add_check(howamidoing)

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def predict_function(pred_text, pad_size):
  encoded_pred_text = text_encoder.encode(pred_text)
  encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
  encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
  predictions = model.predict(tf.expand_dims(encoded_pred_text, 0))
  return (predictions.tolist())

@app.route('/saclassifier', methods=['POST'])
def predict_sentiment():
    text = request.get_json()['text']
    app.logger.info("Text from request " + text)
    predictions = predict_function(text, padding_size)
    app.logger.info('Prediction converted to string ' + ''.join(map(str, predictions[0])))
    sentiment = 'positive' if float(''.join(map(str, predictions[0]))) > 0 else 'negative'
    return jsonify({'Predictions ':predictions, 'sentiment ':sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# curl command to validate the endpoint
# curl http://localhost:5000/hcheck
# curl -H "Content-Type: application/json" -X POST -d '{"text":"I tell Alexia to play ocean sounds when I go to bed at night and sleep soundly until morning. I wouldn\'t want to be without it. I plan to get another one for my husband\'s Nursing Home room to help him sleep. I especially like Thunder Storms at night. It reminds me that I always slept well when there was a thunderstorm. The sounds are perfect to imagine that I really am curled up with my husband safe inside while the storm goes on outside.  I ask the Alexa Dot for these sounds anytime I want to drift off. If you don\'t like Thunderstorms there are other sounds. One is called Ocean Sounds that is so soothing I can fall asleep to it too."}' http://localhost:5000/saclassifier



