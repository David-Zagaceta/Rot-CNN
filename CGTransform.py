import torch
from torch import nn
from ClebschGordan import ClebschGordanMatrices
from typing import List

def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    #print(a.shape[-2:], b.shape[-2:])
    temp: List[int] = (torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:])).tolist()
    siz1 = torch.Size(temp)
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

class CgTransform(nn.Module):

    """Clebsch Gordan coefficient calculator for
    CG-Transforms based on the Fourier analysis of
    the sphere

    Args:
        L (int): spherical harmonic truncation index
    """

    def __init__(self, L, device):
        super(CgTransform, self).__init__()

        if L < 0:
            raise ValueError("L must be greater than or equal to zero")

        self.device = device

        self.L = L
        CGmatrices = ClebschGordanMatrices(L).calculate()
        for i, item in enumerate(CGmatrices):
            inds, ls, mat = item
            CGmatrices[i][0] = inds.to(device)
            CGmatrices[i][1] = ls.to(device)
            CGmatrices[i][2] = mat.to(device)
        self.matrices = CGmatrices

    def forward(self, clms):
        L = self.L
        CG_mats = self.matrices
        # compute all kronecker products
        lsize = (L+1)**2
        T = clms.shape[1]
        Gs = kron(clms,clms).view(lsize, lsize, T**2)

        # perform the Transform
        temp = []
        for l in range(L+1):
            temp.append([])


        for inds, ls, cg in CG_mats:
            l1, l2 = inds
            l1start = l1**2
            l1end = (l1+1)**2
            l2start = l2**2
            l2end = (l2+1)**2
            l1l2dim = (2*l1+1)*(2*l2+1)
            #index the kron product
            g = Gs[l1start:l1end, l2start:l2end, :].reshape(l1l2dim, T**2)
            prod = cg@g
            for l in ls:
                lstart = l**2
                lend = (l+1)**2
                temp[l].append(prod[lstart:lend,:])

        Hls = []
        for l in range(L+1):
            Hls.append(torch.cat(temp[l], dim=1))
        return Hls

class ConvLinear(nn.Module):

    """
    Linear portion of the Fourier space 3D CNN

    args:
        L (int): Truncation of the spherical harmonic Transform
        T (int): Type (or size) of each output after the Linear
                    transformation
        n (int): specifies the position of this layer
                in the stack of convolution layers
        Nconv (int): specifies the total number of convolution
                    layers in the network

    """

    def __init__(self, L, T, n, Nconv):
        super(ConvLinear, self).__init__()

        if L < 0:
            raise ValueError("L must be greater than or equal to zero")

        if T <= 0:
            raise ValueError("T must be greater than zero")

        if n <= 0:
            raise ValueError("n must be greater than or equal to zero")

        if Nconv <= 0:
            raise ValueError("Nconv must be greater than zero")

        self.L = L
        self.T = T
        self.n = n
        self.Nconv = Nconv


        # get counts

        Counts = torch.zeros(L+1, dtype=torch.int)

        for l2 in range(L+1):
            for l1 in range(l2+1):
                lmin = l2-l1
                if L >= l1+l2:
                    lmax = l1+l2
                else:
                    lmax = L
                for l in range(lmin,lmax+1):
                    Counts[l] += 1

        Tls = Counts*T**2

        weights = []

        for l in range(L+1):
            if n == Nconv:
                if l == 0:
                    weight = torch.nn.Linear(int(Tls[l]), T, bias=False)
                    weights.append(weight)
            elif n == 1:
                weight = torch.nn.Linear(int(Counts[l]), T, bias=False)
                weights.append(weight)
            else:
                weight = torch.nn.Linear(int(Tls[l]), T, bias=False)
                weights.append(weight)

        self.linear = nn.ModuleList(weights)

    def forward(self, Hls: List[torch.Tensor]):

        if self.n == self.Nconv:
            layer = self.linear[0]
            return layer(Hls[0])

        else:
            #layer = self.linear[0]
            #res = layer(Hls[0])
            results = []
            for l, Layer in enumerate(self.linear):
                results.append(Layer(Hls[l]))

            res = torch.cat(results,dim=0)

            return res
