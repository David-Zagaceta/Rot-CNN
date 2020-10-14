import torch
from torch import nn

class MLP(nn.Module):



    def __init__(self, n_in, n_layers=2, n_hidden=16, activation=nn.ReLU):
        super(MLP, self).__init__()

        if isinstance(n_hidden, int):
            layer_sizes = [n_hidden]*n_layers + [1]

        elif isinstance(n_hidden, list):
            if length(n_hidden) != n_layers:
                raise TypeError("The number of specified hidden layer sizes must be equal to the number of hidden layers")
            layer_sizes = n_hidden + [1]

        else:
            raise TypeError("n_hidden must be either an integer or a list")

        # check if activation is a function
        if callable(activation):
            activations = [activation]*n_layers

        elif isinstance(activation, list):
            if length(activation) != n_layers:
                raise TypeError("The number of activation functions must be equal to the number of hidden layers")
            activations = activation

        else:
            raise TypeError("activation must be a callable or a list")

        modules = []

        for i in range(n_layers):
            if i == 0:
                modules.append(nn.Linear(n_in, layer_sizes[i]))
            else:
                modules.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

            modules.append(activations[i]())

        # output layer
        modules.append(nn.Linear(layer_sizes[n_layers-1], layer_sizes[n_layers]))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x):
        res = self.mlp(x)
        return res
