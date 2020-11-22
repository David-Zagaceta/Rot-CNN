import torch
from torch import nn
from SphericalHarmonicTransform import SphericalHarmonicTransform
from CGTransform import CgTransform, ConvLinear
from MLP import MLP

class CGnet(nn.Module):

    def __init__(self, L, T, Nconv, rcut, skip=1, n_layers=5, n_hidden=64, activation=nn.Tanh, device=None):
        super(CGnet, self).__init__()
        modules = []

        if device == None:
            device = 'cpu'

        self.skip = skip
        self.device = device
        if skip == 1:
            self.conv_ids = []

            i = 0
            modules.append(SphericalHarmonicTransform(L,rcut, device))
            for n in range(1,Nconv+1,1):
                modules.append(CgTransform(L,device))
                i += 1
                modules.append(ConvLinear(L,T,n,Nconv))
                i += 1
                self.conv_ids.append(i)

            modules.append(nn.LayerNorm(T*Nconv))
            modules.append(nn.Dropout())
            modules.append(MLP(T*Nconv, n_layers, n_hidden, activation))

            self.cgnet = torch.nn.ModuleList(modules)

        elif skip == 0:
            modules.append(SphericalHarmonicTransform(L,rcut, device))
            for n in range(1,Nconv+1,1):
                modules.append(CgTransform(L,device))
                modules.append(ConvLinear(L,T,n,Nconv))

            modules.append(nn.LayerNorm(T))
            modules.append(nn.Dropout())
            modules.append(MLP(T, n_layers, n_hidden, activation))

            self.cgnet = torch.nn.Sequential(*modules)


    def forward(self, nl):
        if self.skip == 1:
            nAtoms = nl.shape[0]
            energy = torch.zeros((1,1),device=self.device)
            for i in range(nAtoms):
                x = nl[i]
                # all modules up to first conv linear
                for j in range(0, self.conv_ids[0]+1, 1):
                    x = self.cgnet[j](x)
                out = x[0]
                # do the rest of the convolutions
                for j in range(self.conv_ids[0]+1, self.conv_ids[-1]+1, 1):
                    x = self.cgnet[j](x)
                    if j in self.conv_ids:
                        out = torch.cat((out, x[0]), dim=0)
                # finalize the network
                for j in range(self.conv_ids[-1]+1, len(self.cgnet), 1):
                    out = self.cgnet[j](out)
                energy += out
            return energy

        elif self.skip == 0:
            nAtoms = nl.shape[0]
            energy = torch.zeros((1,1),device=self.device)
            for i in range(nAtoms):
                energy += self.cgnet(nl[i])
            return energy
