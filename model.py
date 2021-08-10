import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size6, hidden_size3, hidden_size2, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size6)
        self.l2 = nn.Linear(hidden_size6, hidden_size3)
        self.l3 = nn.Linear(hidden_size3, hidden_size2) #TESTEO CON 3 CAPAS
        self.l4 = nn.Linear(hidden_size2, hidden_size) #TESTEO CON 3 CAPAS
        self.l5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        # Al final no se activa y no hay softmax
        return out
