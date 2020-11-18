import torch
from torch.utils.data import Dataset


class SimpleAtomsDataset(Dataset):
    def __init__(self, atomslist, nlcalc, device):
        self.atomslist = atomslist
        self.nlcalc = nlcalc

        self.datalist = []
        for d in atomslist:
            structure = d['structure']
            nl, inds = self.nlcalc.get_environment(structure)
            nl = nl.to(device)
            energy = d['energy']
            eng = torch.zeros((1,1), device=device)
            eng[0,0] = energy
            self.datalist.append((nl, eng))

        #self.datalist = [self.datalist[0]]


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        '''
        structure = self.atomslist[idx]['structure']
        nl, inds = self.nlcalc.get_environment(structure)
        energy = self.atomslist[idx]['energy']
        eng = torch.zeros((1,1))
        eng[0,0] = energy
        '''

        nl = self.datalist[idx][0]
        eng = self.datalist[idx][1]
        return nl, eng
