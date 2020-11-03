import numpy as np

L = 2
T = 5

Counts = np.zeros(L+1)

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
    print(l)
    # Size of Hls[l]
    print(Tls[l])

'''
# conv params for 1 layer
convparams = np.sum(Tls*T)

nlayers = 2
totalconvparams = convparams*nlayers

# weights
fullyconnected_params = T*16 + 16*16 + 16
# bias
fullyconnected_params += 16+16+16

print(totalconvparams+fullyconnected_params)
'''
