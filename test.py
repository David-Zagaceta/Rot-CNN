import torch
from SphericalHarmonicTransform import SphericalHarmonicTransform
from CGTransform import CgTransform, ConvLinear
from MLP import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# simulated neighbor list for one atom
numNeighbors = 50000
nl = torch.rand(numNeighbors,3)
#nl.requires_grad = True
# CNN params
L = 1
T = 3
Nconv = 2

# construct a network
RotCNN = torch.nn.Sequential(SphericalHarmonicTransform(L),
                            CgTransform(L),
                            ConvLinear(L,T,1,Nconv),
                            CgTransform(L),
                            ConvLinear(L,T,2,Nconv),
                            MLP(n_in=T))

x = RotCNN(nl)
print(x)
#print(torch.autograd.grad(x.sum(), nl, retain_graph=True))
