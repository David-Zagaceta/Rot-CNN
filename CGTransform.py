import torch
from torch import nn
from ClebschGordan import ClebschGordanMatrices

def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
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

    def __init__(self, L):
        super(CgTransform, self).__init__()

        if L < 0:
            raise ValueError("L must be greater than or equal to zero")

        self.L = L
        self.matrices = ClebschGordanMatrices(L).calculate()

    def forward(self, clms):
        L = self.L
        CG_mats = self.matrices

        Gs = []

        # compute all kronecker products
        i = 0
        for l2 in range(L+1):
            j = 0
            for l1 in range(l2+1):
                f1 = clms[j:j+2*l1+1]
                f2 = clms[i:i+2*l2+1]
                k = kron(f1,f2)
                Gs.append(k)
                j += 2*l1+1
            i += 2*l2+1

        Hls = []

        # perform the Transform

        CGCount = 0
        Gcount = 0
        CountArray = torch.zeros(L+1,dtype=torch.int)
        for l2 in range(L+1):
            for l1 in range(l2+1):
                lmin = l2-l1
                if L >= l1+l2:
                    lmax = l1+l2
                else:
                    lmax = L
                for l in range(lmin,lmax+1):
                    if CountArray[l] <= 0:
                        temp = torch.matmul(CG_mats[CGCount],Gs[Gcount])
                        Hls.append(temp)
                        CountArray[l] += 1
                    else:
                        temp = torch.matmul(CG_mats[CGCount],Gs[Gcount])
                        Hls[l] = torch.cat((Hls[l],temp),dim=1)
                        CountArray[l] += 1
                    CGCount += 1
                Gcount += 1
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
            raise ValueError("n must be greater than zero")

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

        for l in range(L+1):
            if n == 1:
                weight = torch.nn.Linear(Counts[l], T, bias=False)
                setattr(self, 'W'+str(l), weight)
            elif n == Nconv:
                if l == 0:
                    weight = torch.nn.Linear(Tls[l], T, bias=False)
                    setattr(self, 'W'+str(l), weight)
            else:
                weight = torch.nn.Linear(Tls[l], T, bias=False)
                setattr(self, 'W'+str(l), weight)



        # initialize weights scheme

    def forward(self, Hls):

        if self.n == self.Nconv:
            layer = getattr(self,'W0')
            return layer(Hls[0])

        else:
            layer = getattr(self,'W0')
            res = layer(Hls[0])
            for l in range(1,self.L+1):
                key = 'W'+str(l)
                layer = getattr(self,key)
                res = torch.cat((res,layer(Hls[l])),dim=0)

            return res
