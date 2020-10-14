import torch
from CGnet import CGnet
import torch.optim as optim
import os
import requests
from ase import Atoms
from NeighborList import AseEnvironmentProvider
from load_json import parse_json

torch.device('cuda')

TrainData = "training.json"
TestData  = "test.json"

trainingset = parse_json(TrainData)

rcut = 4.9

# CNN params
L = 2
T = 3
Nconv = 1

#
nlcalc = AseEnvironmentProvider(rcut)

# just energy for right now
TrainFeatures = []
TrainTargets = []

for i, d in enumerate(trainingset):
    if 'structure' in d:
        structure = d['structure']
        '''
        structure = Atoms(symbols=d['structure'].numbers,
                          positions=d['structure'].cart_coords,
                          cell=d['structure'].lattice._matrix, pbc=True)

        '''
        nl, inds = nlcalc.get_environment(structure)
        energy = d['energy']
        TrainFeatures.append(nl)
        eng = torch.zeros((1,1))
        eng[0,0] = energy
        TrainTargets.append(eng)


cgn = CGnet(L,T,Nconv,rcut)
criterion = torch.nn.MSELoss()
# create your optimizer
optimizer = optim.Adam(cgn.parameters(), lr=0.01)

nepochs = 10

print("Training")
for i in range(nepochs):
    mae = 0
    for (nl, eng) in zip(TrainFeatures, TrainTargets):
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        eng_pred = cgn.forward(nl)
        loss = criterion(eng_pred, eng)
        mae += abs((eng-eng_pred).sum())/nl.shape[0]
        loss.backward()
        optimizer.step()    # Does the update
    print(mae)
