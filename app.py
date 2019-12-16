"""
Created on Sun Dec 15 03:27:57 2019

@author: jadele
"""

from flask import Flask, request, jsonify, render_template
import pickle
from load.py import *
import keras.models
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)

global model, graph
model, graph = init()
tokenizer = pickle.load(open('convert.pkl','rb'))

@app.route('/',methods=[])
def home():
    return render_template('index.html')
        
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.transform(data).toarray()
        prediction = model.predict(vect)
  
    return render_template('result.html', prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)
