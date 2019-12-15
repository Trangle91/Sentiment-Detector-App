#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 03:27:57 2019

@author: jadele
"""

from flask import Flask, request, jsonify, render_template
import pickle
from keras.models import model_from_json

app = Flask(__name__)

global model, graph
model, graph = init()

# loading model
model = model_from_json(open('model.json').read())
model.load_weights('model_weights.h5')
tokenizer = pickle.load(open('convert.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
        
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.transform(data).toarray()
        prediction = model.predict(vect)
    
        if prediction[0]:
            result = "negative"
        else:
            result = "positive"
    return render_template('index.html', prediction_text='The sentence is {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)