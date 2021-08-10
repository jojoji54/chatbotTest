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

#Este archivo permite recibir y enviar datos con la parte de Flutter


CORS(app)

#Get me permitira conseguir ls datos de Flutter
@app.route("/", methods=["GET"])
def home():
    return '<h1>Demo</h1>'


CORS(app)

#POST me eprmitira enviar los datos a Flutter
@app.route("/bot", methods=["POST"])
def response():
    app.logger.info('start')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.logger.info(device)
    
    with open('intents.json', 'r') as json_data: #abro el archivo json que es el archivo con los comandos 
        intents = json.load(json_data)
    app.logger.info(intents)
    
    
    FILE = "data.pth"
    data = torch.load(FILE) #abro el archivo data que es el archivo ya entrenado
    app.logger.info(data)
    input_size = data["input_size"]#recogo el tamaño de los datos de entrada
    hidden_size6 = data["hidden_size6"]
    hidden_size2 = data["hidden_size2"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"] #recogo el tamaño de los datos de salida
    all_words = data['all_words'] #la bolsa de palabras del archivo entrenado
    tags = data['tags'] #las etiquetas tag del archivo entrenado
    model_state = data["model_state"] #El modelo de datos del archivo emtrenad

    model = NeuralNet(input_size, hidden_size6, hidden_size2,hidden_size, output_size).to(device) #Selecciono los datos que voy ha utilizar
    model.load_state_dict(model_state)
    model.eval() #Evaluo los datos del modelo
    # return '<h2>sdfjk</h2>'
    query = dict(request.form)['query'] #Recogo los datos de Flutter que vienen con la etiqueta query

    res = query
    res = tokenize(res) #tokenizo los datos de flutter

    X = bag_of_words(res, all_words) #creo la bolsa de palabras
    X = X.reshape(1, X.shape[0]) #calculo su tamaño
    X = torch.from_numpy(X).to(device) #Solicito utilizar la libreria entrenada

    output = model(X) #selecciono el modelo
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()] #selleciono la etiequeta que es la selleccionada a taves de las predicciones anteriores

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()] #ya seleccionadas las etiquetas miro la que mas probabilidad de que sea tenga

    if prob.item() > 0.75: #Si el comando tiene una probabilidad de que sea la acertada de mas del 70%....
        app.logger.info('%d logged in successfully', prob.item())
        app.logger.info(intents['intents'])
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if intent["tag"] == "goodbye":
                    os.remove("intents.json")
                    os.system('python database.py')
                   # f = open("database.py")
                    #f = open("randomDatabase.py")
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
        #return jsonify({"response": random.choice(['I siee...', 'mmmmmm', 'ops..', 'O_O', 'jumm..', 'okeyyy', 'ok', 'tell me more'])})
        #return jsonify({"response": random.choice(['I siee...', 'mmmmmm', ', 'jumm..', 'okeyyy', 'ok', 'tell me more', '\N{thinking face} \N{thinking face}', '\N{face without mouth} ', '\N{lying face} \N{lying face}  jajaj', '\N{relieved face} \N{relieved face}', '\N{OK hand} \N{OK hand} \N{OK hand} \N{OK hand}', '\N{face with open mouth} \N{face with open mouth} \N{face with open mouth}', 'ou \N{flexed biceps} \N{flexed biceps}' , '.. \N{eyes} \N{eyes} ...'  ])})
        return jsonify({"response": random.choice(['Sorry, I am not sure about the answer...', 'Wait, do you really something about that?', 'mmm I am not sure, may be you have to train me more', 'OPS, I dont have any information about that', 'OHh no ! I dont have the answer of that question :(', 'Ou... that... I dont have any response for that', 'I dont have the answer of that question, Are you sure that you trained me?'])})