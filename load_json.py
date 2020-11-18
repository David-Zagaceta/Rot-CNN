import os
import gc
import time
import shelve
import numpy as np
from ase import Atoms
from copy import deepcopy
from functools import partial
from torch.utils.data import Dataset
from monty.serialization import loadfn
from multiprocessing import Pool, cpu_count
from collections.abc import MutableSequence

def parse_json(path, N=None, Random=False):
    """ Extract structures/energy/forces/stress information from json file. """
    if os.path.isfile(path):
        structure_dict = loadfn(path)
    elif os.path.isdir(path):
        import glob
        cwd = os.getcwd()
        os.chdir(path)
        files = glob.glob('*.json')
        os.chdir(cwd)
        structure_dict = []
        for file in files:
            fp = os.path.join(path, file)
            structure_dict += loadfn(fp)

    if N is None:
        N = len(structure_dict)
    elif Random and N < len(structure_dict):
        structure_dict = sample(structure_dict, N)

    data = []
    for i, d in enumerate(structure_dict):
        if 'structure' in d:
            structure = Atoms(symbols=d['structure'].atomic_numbers,
                              positions=d['structure'].cart_coords,
                              cell=d['structure'].lattice._matrix, pbc=True)
            v = structure.get_volume()
            if 'data' in d:
                key = 'data'
            else:
                key = 'outputs'

            if 'energy_per_atom' in d[key]:
                energy = d[key]['energy_per_atom']*len(structure)
            else:
                energy = d[key]['energy']
            force = d[key]['forces']
            try:
                if d['tags'][0] == 'Strain':
                    group = 'Elastic'
                else:
                    group = 'NoElastic'
            except:
                if d['group'] == 'Elastic':
                    group = 'Elastic'
                else:
                    group = 'NoElastic'
            #group = d['group']
            # vasp default output: XX YY ZZ XY YZ ZX
            # pyxtal_ff/lammps use: XX YY ZZ XY XZ YZ
            # Here we assume the sequence is lammps
            if d['group'] == 'Elastic' and 'Mo 3x3x3 cell' in d['description']: #this is very likely a wrong dataset
                stress = None
                group = 'NoElastic'
            elif 'virial_stress' in d[key]: #kB to GPa
                s = [-1*s/10 for s in d[key]['virial_stress']]

                if d['group'] == 'Ni3Mo': #to fix the issue
                    stress = [s[0], s[1], s[2], s[3], s[4], s[5]]
                else:
                    stress = [s[0], s[1], s[2], s[3], s[5], s[4]]

            elif 'stress' in d[key]: #kB to GPa
                s = [-1*s/10 for s in d[key]['stress']]
                if d['group'] == 'Ni3Mo': #to fix the issue
                    stress = [s[0], s[1], s[2], s[3], s[4], s[5]]
                else:
                    stress = [s[0], s[1], s[2], s[3], s[5], s[4]]
            else:
                stress = None

            data.append({'structure': structure,
                         'energy': energy, 'force': force,
                         'stress': stress, 'group': group})

        else:   # For PyXtal
            structure = Atoms(symbols=d['elements'], scaled_positions=d['coords'],
                              cell=d['lattice'], pbc=True)
            data.append({'structure': structure,
                         'energy': d['energy'], 'force': d['force'],
                         'stress': None, 'group': 'random'})

        if i == (N-1):
            break


    return data
