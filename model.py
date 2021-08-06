import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size6, hidden_size5, hidden_size4, hidden_size3, hidden_size2, hidden_size1, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size6)
        self.l2 = nn.Linear(hidden_size6, hidden_size5)
        self.l3 = nn.Linear(hidden_size5, hidden_size4)
        self.l4 = nn.Linear(hidden_size4, hidden_size3)
        self.l5 = nn.Linear(hidden_size3, hidden_size2)
        self.l6 = nn.Linear(hidden_size2, hidden_size1)
        self.l7 = nn.Linear(hidden_size1, hidden_size)
        self.l8 = nn.Linear(hidden_size, hidden_size)
        self.l9 = nn.Linear(hidden_size, num_classes)
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
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        out = self.relu(out)
        out = self.l8(out)
        out = self.relu(out)
        out = self.l9(out)
        # Al final no se activa y no hay softmax
        return out
