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

        self.n_layers = n_layers
        self.activations = activations

        layer_string = 'layer'
        for i in range(n_layers):
            if i == 0:
                temp_layer = nn.Linear(n_in, layer_sizes[i])
            else:
                temp_layer = nn.Linear(layer_sizes[i-1], layer_sizes[i])

            setattr(self, layer_string+str(i), temp_layer)

        # output layer
        temp_layer = nn.Linear(layer_sizes[n_layers-1], layer_sizes[n_layers])
        setattr(self, 'out'+layer_string, temp_layer)

    def forward(self, x):
        layer = getattr(self,'layer0')
        res = layer(x)
        activation = self.activations[0]
        m = activation(inplace=False)
        res = m(res)
        for i in range(1,self.n_layers):
            layerkey = 'layer'+str(i)
            layer = getattr(self, layerkey)
            activation = self.activations[i]
            m = activation(inplace=False)
            res = layer(res)
            #res = activation(res)
        layer = getattr(self, 'outlayer')
        res = layer(res)

        return res
