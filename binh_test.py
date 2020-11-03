import torch
from CGnet import CGnet
import time
from CGTransform import kron

# network parameters
L = 2
T = 5


# simulated input
clms = torch.rand(((L+1)**2, T), dtype=torch.float32)
CG_mats = []

Count = 0
for l2 in range(L+1):
    for l1 in range(l2+1):
        lmin = l2-l1
        if L >= l1+l2:
            lmax = l1+l2
        else:
            lmax = L

        for l in range(lmin,lmax+1,1):
            dims = (2*l+1, (2*l1+1)*(2*l2+1))
            CG_mats.append(torch.rand(dims, dtype=torch.float32))

'''
CG Transform portion
'''
Gs = []

# compute all kronecker products
i = 0
for l2 in range(L+1):
    j = 0
    for l1 in range(l2+1):
        # size of f1 (2*l1+1) X T
        f1 = clms[j:j+2*l1+1]
        # size of f2 (2*l2+1) X T
        f2 = clms[i:i+2*l2+1]
        # size of k (2*l1+1)(2*l2+1) X T
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
                # temp has the size (2l+1) X T^2
                temp = torch.matmul(CG_mats[CGCount],Gs[Gcount])
                Hls.append(temp)
                CountArray[l] += 1
            else:
                temp = torch.matmul(CG_mats[CGCount],Gs[Gcount])
                # refer to count code for sizing of Hls
                Hls[l] = torch.cat((Hls[l],temp),dim=1)
                CountArray[l] += 1
            CGCount += 1
        Gcount += 1
