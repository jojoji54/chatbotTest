import numpy as np
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# os.system('python database.py')
# os.system('python randomDatabase.py')
#global epoch
os.remove('data.pth')

#Es en este archivo en donde entrenamos nla IA para que sea capaz de reconocer los comandos del archivo de json

#Abro el archivo json que es el archivo que tiene los comandos
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Navego en bucle a traves del fichero json viendo las etiquestas "intents" "patterns"
for intent in intents['intents']:
    tag = intent['tag']
    # Lo añado a la lista tag
    tags.append(tag)
    for pattern in intent['patterns']:
        #Tokenizo cada palabra de la frase
        w = tokenize(pattern)
        # Lo añado a mi bolsa de palabras
        all_words.extend(w)
        # Lo añado a mi pareja xy
        xy.append((w, tag))

# Stem y lo pongo en letras minusculas cada palabra
ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remuevo las entradas duplicadas
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# creo los datos entrenados
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: una bolsa de palabras para cada patron_farse 
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss solo necesita la clase de labels, no otra
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# hiperparámetros
num_epochs = 3000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #soporta los datos del dataset[i] de forma indexa, lo podemos usar para obtener por ejemplo parametros del estilo i-th 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Podemos llamar a la longitud del dataset para calcular el tamaño
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# modelo del entrenamiento
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Pase adelantado
        outputs = model(words)
        # si fuera on-hot, debemos aplicar
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Retroceder y optimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        #epoch = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}';


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
