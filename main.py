from flask import Flask, request,jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np 
import zipfile as zp 
app = Flask(__name__)
CORS(app)
with zp.ZipFile('./model_7.zip', 'r') as zip_ref:
    zip_ref.extractall("n_model")
model = tf.keras.models.load_model(r'.\n_model\n_quotes_prediction_model')

interpreter = {
    0: "between 0 and 10 quotes",
    1: "between 10 and 100 quotes",
    2: "between 100 and 500 quotes",
    3: "between 500 and 1000 quotes",
    4: "over 1000 quotes"
}
def model_out(model,obj):
    age = obj["age"]
    gender = obj["gender"]
    job = obj["job"]
    topic = obj["subject"]
    emotion = obj["emotion"]
    nationality = 'united states of america'
    input = {'age': tf.convert_to_tensor([np.array(age)]),
        'nationality': tf.convert_to_tensor([np.array(nationality)]),
        'gender': tf.convert_to_tensor([np.array(gender)]), 
        'occupation': tf.convert_to_tensor([np.array(job)]) ,
        'topic': tf.convert_to_tensor([np.array(topic)]),
        'emotion': tf.convert_to_tensor([np.array(emotion)]),
        }
    pred = model.predict(input)
    res = np.argmax(tf.nn.softmax(pred), axis = 1)
    # print(res)
    return int(res)

@app.route('/predict/', methods = ['GET'])
def predict():
    obj = dict(request.args)
    obj["age"] = int(obj["age"])
    score = model_out(model,obj)
    return jsonify({'score' : score })

