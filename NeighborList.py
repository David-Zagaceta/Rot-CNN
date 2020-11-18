import math
import numpy as np
import torch
from ase.neighborlist import NeighborList

class AseEnvironmentProvider:
    """
    Environment provider making use of ASE neighbor lists. Supports cutoffs
    and PBCs.
    """

    def __init__(self, cutoff):
        self.rcut = cutoff

    def get_environment(self, atoms):
        # cutoffs for each atom
        cutoffs = [self.rcut/2]*len(atoms)
        # instantiate neighborlist calculator
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        # provide atoms object to neighborlist calculator
        nl.update(atoms)
        # instantiate memory for neighbor separation vectors, periodic indices, and atomic numbers
        center_atoms = np.zeros((len(atoms), 3), dtype=np.float64)
        neighbors = []
        neighbor_indices = []
        atomic_numbers = []

        max_len = 0
        for i in range(len(atoms)):
            # get center atom position
            center_atom = atoms.positions[i]
            center_atoms[i] = center_atom
            # get indices and cell offsets of each neighbor
            indices, offsets = nl.get_neighbors(i)
            # add an empty list to neighbors and atomic numbers for population
            neighbors.append([])
            atomic_numbers.append([])
            # the indices are already numpy arrays so just append as is
            neighbor_indices.append(indices)
            for j, offset in zip(indices, offsets):
                # compute separation vector
                pos = atoms.positions[j] + np.dot(offset, atoms.get_cell()) - center_atom
                neighbors[i].append(pos)
                atomic_numbers[i].append(atoms[j].number)

            if len(neighbors[i]) > max_len:
                max_len = len(neighbors[i])

        # declare arrays to store the separation vectors, neighbor indices
        # atomic numbers of each neighbor, and the atomic numbers of each
        # site
        neighborlist = np.zeros((len(atoms), max_len, 3), dtype=np.float64)
        neighbor_inds = np.zeros((len(atoms), max_len), dtype=np.int64)
        atm_nums = np.zeros((len(atoms), max_len), dtype=np.int64)
        site_atomic_numbers = np.array(list(atoms.numbers), dtype=np.int64)

        # populate the arrays with list elements
        for i in range(len(atoms)):
            neighborlist[i, :len(neighbors[i]), :] = neighbors[i]
            neighbor_inds[i, :len(neighbors[i])] = neighbor_indices[i]
            atm_nums[i, :len(neighbors[i])] = atomic_numbers[i]

        neighborlist = torch.from_numpy(neighborlist)
        neighbor_inds = torch.from_numpy(neighbor_inds)
        return neighborlist, neighbor_inds
