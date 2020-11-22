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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

TrainData = "training.json"
TestData  = "test.json"

trainingset = parse_json(TrainData)
validationset = parse_json(TestData)


rcut = 4.9

# CNN params
L = 8
T = 12
Nconv = 3


nlcalc = AseEnvironmentProvider(rcut)

# with only energy training
trainingdata = SimpleAtomsDataset(trainingset, nlcalc, device=device)
validationdata = SimpleAtomsDataset(validationset, nlcalc, device=device)

cgn = CGnet(L,T,Nconv,rcut,device=device,skip=0)
cgn = cgn.to(device)


criterion = torch.nn.MSELoss()
# create optimizer
optimizer = optim.Adam(cgn.parameters(), lr=3e-4)

nepochs = 1000

Trainer = Trainer(cgn, optimizer, criterion, trainingdata, trainingdata, validationinterval=20)
