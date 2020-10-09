import torch
from torch import nn
from math import factorial, sqrt

class SphericalHarmonicTransform(nn.Module):

    """Spherical harmonic transform calculator from
        cartesian coordinates for the atomic neighbor
        density function

    Args:
        l (int): spherical harmonic index
        m (int): spherical harmonic index
    """

    def __init__(self, L):
        super(SphericalHarmonicTransform, self).__init__()

        if L < 0:
            raise ValueError("l must be greater than or equal to zero")

        self.L = L

    def forward(self, pos):
        """Computes a real spherical harmonic Transform given
            cartesian vectors"""

        # pos [nneighbors,3]
        pi = torch.acos(torch.zeros(1)).item() * 2
        L = self.L

        clms = torch.zeros((L+1)**2,1)

        # construct covariant spherical coordinates
        norms = torch.norm(pos,dim=1)
        ids = norms > 0
        xpl1 = -0.5*(pos[ids,0] + 1.0j*pos[ids,1])
        xm1 = 0.5*(pos[ids,0] - 1.0j*pos[ids,1])
        x0 = pos[ids,2]

        i = 0
        for l in range(L+1):
            for M in range(-l,1):

                m = abs(M)
                # compute the solid harmonic
                SolidHarmonics = torch.zeros_like(xpl1)
                for p in range(l + 1):
                    q = p - m
                    s = l - p - q
                    if q >= 0 and s >= 0:
                        SolidHarmonics += torch.mul(torch.mul(torch.pow(xpl1, p),
                                 torch.pow(xm1,q)),
                                 torch.pow(x0,s)) / \
                                 (factorial(p) * factorial(q) *
                                  factorial(s))

                SolidHarmonics *= sqrt(factorial(l+m)*factorial(l-m))
                # add atomic numbers here soon
                Ylms = SolidHarmonics * sqrt((2*l+1)/4/pi) *\
                        torch.pow(norms[ids],2)

                clm = torch.sum(Ylms, dim=0)

                if m == 0:
                    clms[i] = clm.real
                else:
                    fac = sqrt(2) * (-1)**m

                    clms[i] = fac*clm.imag
                    clms[i+2*m] = fac*clm.real

                i += 1
            i += l

        return clms
