"""
Created on Sun Dec 15 03:27:57 2019

@author: jadele
"""

from flask import Flask, request, jsonify, render_template
import pickle
from load.py import *
import keras.models

from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)

#loading model
global model, graph
model, graph = init()
#loading tokenizer
tokenizer = pickle.load(open('convert.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
        
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.transform(data).toarray()
        prediction = model.predict(vect)
  
    return render_template('index.html', Result = prediction)


if __name__ == "__main__":
    app.run(debug=True)
