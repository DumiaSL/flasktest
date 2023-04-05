from flask import Flask, request, jsonify
from pydantic import BaseModel
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization
from keras.models import load_model


app = Flask(__name__)

class sentence(BaseModel):
    Sentence:str

model = load_model('model.h5')

#open the vectorizer with the vocab used to train the model
with open('vectorizer_config_new.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    vocab = data['vocab']
    config = data['config']
vectorizer = TextVectorization.from_config(config)
vectorizer.set_vocabulary(vocab)

@app.route('/', methods=['POST'])
def scoring_endpoint():
    item = request.get_json(force=True)
    input_text = item['Sentence']
    x = vectorizer(np.array([input_text]))
    yhat = model.predict(x)

    report={"toxic": str(yhat[0][0]>0.5),
            "severe toxic":str(yhat[0][1]>0.5),
            "obscene":str(yhat[0][2]>0.5),
            "threat":str(yhat[0][3]>0.5),
            "insult":str(yhat[0][4]>0.5),
            "identity hate":str(yhat[0][5]>0.5)
            }

    return jsonify({"prediction": report})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
