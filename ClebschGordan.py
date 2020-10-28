from sympy.physics.quantum.cg import CG
import torch

class ClebschGordanMatrices:

    """Clebsch Gordan coefficient calculator for
    CG-Transforms based on the Fourier analysis of
    the sphere

    Args:
        L (int): spherical harmonic truncation index
    """

    def __init__(self, L):
        if L < 0:
            raise ValueError("L must be greater than or equal to zero")

        self.L = L

    def calculate(self):
        L = self.L
        CG_matrices = []
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
                    CG_matrices.append(torch.zeros(dims, dtype=torch.float32))

                    for i, m in enumerate(range(-l,l+1)):
                        j = 0
                        for m1 in range(-l1,l1+1):
                            for m2 in range(-l2,l2+1):
                                cg = float(CG(l1,m1,l2,m2,l,m).doit())
                                CG_matrices[Count][i,j] = cg
                                j += 1
                    Count += 1

        return CG_matrices
