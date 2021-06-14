
import random
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import time
import os
import emoji


app = Flask(__name__, static_url_path='')
#global epoch
# os.system('python database.py')
# os.system('python train.py')


CORS(app)


@app.route("/", methods=["GET"])
def home():
    return '<h1>Demo</h1>'


CORS(app)


@app.route("/bot", methods=["POST"])
def response():
    app.logger.info('start')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.logger.info(device)
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
    app.logger.info(intents)
    FILE = "data.pth"
    data = torch.load(FILE)
    app.logger.info(data)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    # return '<h2>sdfjk</h2>'
    query = dict(request.form)['query']

    res = query
    res = tokenize(res)

    X = bag_of_words(res, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        app.logger.info('%d logged in successfully', prob.item())
        app.logger.info(intents['intents'])
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if intent["tag"] == "goodbye":
                    f = open("database.py")
                    f = open("randomDatabase.py")
                    #f = open("train.py")
                    os.system('python train.py')
                    return jsonify({"response": random.choice(intent['responses'])})
                # elif intent["tag"] == "goodbye":
                #      os.system('python train.py')
                #      return jsonify({"response" : train.epoch})
                else:
                    return jsonify({"response": random.choice(intent['responses'])})
                
    # elif prob.item() > 0.50 < 0.70:
    #     app.logger.info('%d logged in successfully', prob.item())
    #     app.logger.info(intents['intents'])
    #     return jsonify({"response": random.choice(['I siee...', 'mmmmmm', 'ops..', 'O_O'])}) 

    else:
        return jsonify({"response": random.choice(['I siee...', 'mmmmmm', 'ops..', '\N{slightly smiling face} \N{slightly smiling face} \N{slightly smiling face}', 'jumm..', 'okeyyy', 'ok', 'tell me more', '\N{thinking face} \N{thinking face}', '\N{face without mouth} ', '\N{lying face} \N{lying face}  jajaj', '\N{relieved face} \N{relieved face}', '\N{OK hand} \N{OK hand} \N{OK hand} \N{OK hand}', '\N{face with open mouth} \N{face with open mouth} \N{face with open mouth}', 'ou \N{flexed biceps} \N{flexed biceps}' , '.. \N{eyes} \N{eyes} ...'  ])})
