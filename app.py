"""
Created on Sun Dec 15 03:27:57 2019

@author: jadele
"""

from flask import Flask, request, jsonify, render_template
import pickle
import keras.models
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)

global model, graph
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load woeights into new model
model.load_weights("model_weights.h5")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

tokenizer = pickle.load(open('convert.pkl','rb'))

@app.route('/',methods=[])
def home():
    return render_template('index.html')
        
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.transform(data).toarray()
        prediction = model.predict(vect)
  
    return render_template('result.html', prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)
