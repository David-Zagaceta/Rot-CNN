import torch
from CGnet import CGnet
import torch.optim as optim
import os
import requests
from ase import Atoms
from NeighborList import AseEnvironmentProvider
from load_json import parse_json
from data import SimpleAtomsDataset
from torch.utils.data import DataLoader
from Trainer import Trainer
import time

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

TrainData = "training.json"
TestData  = "test.json"

trainingset = parse_json(TrainData)
validationset = parse_json(TestData)


rcut = 4.9

# CNN params
L = 2
T = 2
Nconv = 3

'''
nlcalc = AseEnvironmentProvider(rcut)

# with only energy training
trainingdata = SimpleAtomsDataset(trainingset, nlcalc, device=device)
#validationdata = SimpleAtomsDataset(validationset, nlcalc, device=device)

trainingdata = trainingdata[0:50]
'''
cgn = CGnet(L,T,Nconv,rcut,device=device,skip=0)
cgn = cgn.to(device)

cgn.forward(torch.rand((1,10,3),dtype=torch.float32))

'''
criterion = torch.nn.MSELoss()
# create optimizer
optimizer = optim.Adam(cgn.parameters(), lr=3e-4)

nepochs = 200

Trainer = Trainer(cgn, optimizer, criterion, trainingdata, trainingdata, validationinterval=20)
start = time.time()
Trainer.train(nepochs)
print(time.time()-start)
'''
