import torch
from torch import nn
from math import factorial, sqrt

def powcmplx(zreal, zimag, n):
    if n == 2:
        realpart = zreal**2 - zimag**2
        imagpart = 2*zreal*zimag
        return torch.stack((realpart, imagpart), dim=1)
    elif n > 2:
        temp = powcmplx(zreal, zimag, n-1)
        tempreal = temp[:,0]
        tempimag = temp[:,1]
        realpart = zreal*tempreal - zimag*tempimag
        imagpart = zreal*tempimag + zimag*tempreal
        return torch.stack((realpart, imagpart), dim=1)
    elif n == 1:
        return torch.stack((zreal, zimag), dim=1)
    elif n == 0:
        # change this for tensor type
        realpart = torch.ones_like(zreal)
        imagpart = torch.zeros_like(zreal)
        return torch.stack((realpart, imagpart), dim=1)
    else:
        raise IndexError("powcmplx does not support negative exponents")

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

        #pos = pos.real
        # construct covariant spherical coordinates
        norms = torch.norm(pos,dim=1)
        ids = norms > 0
        xpl1real = -0.5*pos[ids,0]
        xpl1imag = -0.5*pos[ids,1]
        xm1real = 0.5*pos[ids,0]
        xm1imag = -0.5*pos[ids,1]
        x0 = pos[ids,2]

        i = 0
        for l in range(L+1):
            for M in range(-l,1):

                m = abs(M)
                # compute the solid harmonic
                realpart = torch.zeros_like(xpl1real)
                imagpart = torch.zeros_like(xpl1imag)
                SolidHarmonics = torch.stack((realpart, imagpart), dim=1)
                for p in range(l + 1):
                    q = p - m
                    s = l - p - q
                    if q >= 0 and s >= 0:
                        z1 = powcmplx(xpl1real, xpl1imag, p)
                        z2 = powcmplx(xm1real, xm1imag, q)
                        zreal = z1[:,0]**2 - z2[:,1]**2
                        zimag = z1[:,0]*z2[:,1] + z1[:,1]*z2[:,0]
                        x0temp = x0**s
                        z = torch.stack((zreal*x0,zimag*x0), dim=1)
                        SolidHarmonics += z / \
                                 (factorial(p) * factorial(q) * factorial(s))

                SolidHarmonics *= sqrt(factorial(l+m)*factorial(l-m))
                # add atomic numbers here soon
                SolidHarmonics[:,0] *= norms[ids]**l
                SolidHarmonics[:,1] *= norms[ids]**l
                Ylms = SolidHarmonics * sqrt((2*l+1)/4/pi)
                clm = torch.sum(Ylms, dim=0)
                if m == 0:
                    clms[i] = clm[0]
                else:
                    fac = sqrt(2) * (-1)**m

                    clms[i] = fac*clm[1]
                    clms[i+2*m] = fac*clm[0]

                i += 1
            i += l

        return clms
