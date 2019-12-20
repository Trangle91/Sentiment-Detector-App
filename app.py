import pickle
import keras.models
from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from numpy import loadtxt
from keras.models import load_model
from keras.models import model_from_json

#Loading Flask
app = Flask(__name__)

#Model reconstruction from JSON file
with open('model.json', 'r') as f:
    model = model_from_json(f.read())

#Loading weights into the new model
model.load_weights('model_weights.h5')

#Loading tokenizer
tokenizer = pickle.load(open('convert.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.texts_to_sequences(data)
        vect = pad_sequences(vect, padding='post', maxlen=100)
        prediction = model.predict(vect)
        if prediction > 0.5:
                result = "positive"
        else:
                result = "negative"
        
    return render_template('index.html', Result='The sentiment is {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
