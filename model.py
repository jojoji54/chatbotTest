import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    #Lo que se define aqui son los numero de nodos que definimos en la red neuronal y el orden de datos de entrada
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size) #ESTO LO HE PUESTO YO COMO TEST
        self.l4 = nn.Linear(hidden_size, hidden_size) #ESTO LO HE PUESTO YO COMO TEST
        self.l5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out) #ESTO LO HE PUESTO YO COMO TEST
        out = self.relu(out) #ESTO LO HE PUESTO YO COMO TEST
        out = self.l4(out) #ESTO LO HE PUESTO YO COMO TEST
        out = self.relu(out) #ESTO LO HE PUESTO YO COMO TEST
        out = self.l5(out)
        # Al final no se activa y no hay softmax
        return out
