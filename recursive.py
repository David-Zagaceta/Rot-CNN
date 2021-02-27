import numpy as np

def init_rootpqarray(twol):
    ldim = twol+1
    rootpqarray = np.zeros((ldim, ldim))
    for p in range(1, ldim, 1):
        for q in range(1, ldim, 1):
            rootpqarray[p,q] = np.sqrt(p/q)
    return rootpqarray

def compute_uarray_recursive(x, y, z, psi, r, twol):
    '''Compute the Wigner-D matrix of order twol given an axis (x,y,z)
    and rotation angle 2*psi.
    This function constructs a unit quaternion representating a rotation
    of 2*psi through an axis defined by x,y,z; then populates an array of
    Wigner-D matrices of order twol for this rotation.  The Wigner-D matrices
    are calculated using the recursion relations in LAMMPS.
    Parameters
    ----------
    x: float
    the x coordinate corresponding to the axis of rotation.
    y: float
    the y coordinate corresponding to the axis of rotation.
    z: float
    the z coordinate corresponding to the axis of rotation
    psi: float
    one half of the rotation angle
    r: float
    magnitude of the vector (x,y,z)
    twol: integer
    order of hyperspherical expansion
    ulist: 1-D complex array
    array to populate with D-matrix elements, mathematically
    this is a 3-D matrix, although we broadcast this to a 1-D
    matrix
    idxu_block: 1-D int array
    used to index ulist
    rootpqarray:  2-D float array
    used for recursion relation
    Returns
    -------
    None
    '''
    ldim = twol + 1

    idxu_block = np.zeros(ldim, dtype=np.int64)

    idxu_count = 0
    for l in range(0, ldim, 1):
        idxu_block[l] = idxu_count
        for mb in range(0, l + 1, 1):
            for ma in range(0, l + 1, 1):
                idxu_count += 1

    ulist = np.zeros(idxu_count, dtype=np.complex128)

    rootpqarray = init_rootpqarray(twol)


    # construct Cayley-Klein parameters from unit quaternion
    # LAMMPS quaternion
    z0 = r / np.tan(psi)
    r0inv = 1.0 / np.sqrt(r * r + z0 * z0)
    a = r0inv * (z0 - z * 1j)
    b = r0inv * (y - x * 1j)

    ulist[0] = 1.0 + 0.0j

    for l in range(1, ldim, 1):
        llu = idxu_block[l]
        llup = idxu_block[l - 1]

        # fill in left side of matrix layer

        mb = 0
        while 2 * mb <= l:
            ulist[llu] = 0
            for ma in range(0, l, 1):
                rootpq = rootpqarray[l - ma, l - mb]
                ulist[llu] += rootpq * a.conjugate() * ulist[llup]

                rootpq = rootpqarray[ma + 1, l - mb]
                ulist[llu + 1] += -rootpq * b.conjugate() * ulist[llup]
                llu += 1
                llup += 1
            llu += 1
            mb += 1

        # copy left side to right side using inversion symmetry
        llu = idxu_block[l]
        llup = llu + (l + 1) * (l + 1) - 1
        mbpar = 1
        mb = 0
        while 2 * mb <= l:
            mapar = mbpar
            for ma in range(0, l + 1, 1):
                if mapar == 1:
                    ulist[llup] = ulist[llu].conjugate()
                else:
                    ulist[llup] = -ulist[llu].conjugate()

                mapar = -mapar
                llu += 1
                llup -= 1
            mbpar = -mbpar
            mb += 1
    return ulist


def compute_duarray_recursive(x, y, z, psi, r, rscale0, twol):
    ldim = twol + 1

    idxu_block = np.zeros(ldim, dtype=np.int64)

    idxu_count = 0
    for l in range(0, ldim, 1):
        idxu_block[l] = idxu_count
        for mb in range(0, l + 1, 1):
            for ma in range(0, l + 1, 1):
                idxu_count += 1

    dulist = np.zeros((idxu_count,3), dtype=np.complex128)

    ulist = compute_uarray_recursive(x,y,z,psi,r,twol)

    rootpqarray = init_rootpqarray(twol)

    z0 = r/ np.tan(psi)
    rsq = r*r
    r0inv = 1.0 / np.sqrt(rsq + z0*z0)
    a = r0inv * (z0 - 1j*z)
    b = r0inv * (y - x*1j)

    dz0dr = z0 / r - r*rscale0 * (rsq + z0*z0) /rsq

    dr0invdr = -r0inv**3 * (r + z0 * dz0dr)

    ux = x/r
    uy = y/r
    uz = z/r

    dr0inv = np.zeros(3, np.complex128)

    dz0 = np.zeros(3, np.complex128)

    dr0inv[0] = dr0invdr * ux
    dr0inv[1] = dr0invdr * uy
    dr0inv[2] = dr0invdr * uz

    dz0[0] = dz0dr * ux
    dz0[1] = dz0dr * uy
    dz0[2] = dz0dr * uz

    da = np.zeros(3, np.complex128)
    db = np.zeros(3, np.complex128)

    for k in range(3):
        da[k] = dz0[k] * r0inv + z0*dr0inv[k] - 1j*z*dr0inv[k]
        db[k] = (y - 1j*x)*dr0inv[k]

    da[2] += -1j*r0inv

    db[0] += -1j*r0inv
    db[1] += r0inv

    for l in range(1, ldim, 1):
        llu = idxu_block[l]
        llup = idxu_block[l-1]
        mb = 0
        while 2 * mb <= l:
            dulist[llu,0] = 0
            dulist[llu,1] = 0
            dulist[llu,2] = 0
            for ma in range(0,l,1):
                rootpq = rootpqarray[l - ma, l - mb]
                for k in range(3):
                    dulist[llu, k] += rootpq*(da[k].conjugate() * ulist[llup] +
                                              a.conjugate() * dulist[llup,k])

                rootpq = rootpqarray[ma + 1, l - mb]
                for k in range(3):
                    dulist[llu+1,k] = -rootpq * (db[k].conjugate() * ulist[llup] +
                                                 b.conjugate() * dulist[llup, k])

                llu += 1
                llup += 1
            llu += 1
            mb += 1

        llu = idxu_block[l]
        llup = llu + (l + 1) * (l + 1) - 1
        mbpar = 1
        mb = 0
        while 2*mb <= l:
            mapar = mbpar
            for ma in range(0, l + 1, 1):
                if mapar == 1:
                    for k in range(3):
                        dulist[llup,k] = dulist[llu,k].conjugate()
                else:
                    for k in range(3):
                        dulist[llup,k] = -dulist[llu,k].conjugate()

                mapar = -mapar
                llu += 1
                llup -= 1
            mbpar = -mbpar
            mb += 1
    return dulist

def main():
    x = 0
    y = 0
    z = 2.0
    r = np.sqrt(x**2 + y**2 + z**2)
    rcut = 3.0
    rscale0 = 0.99363*np.pi/3.0
    psi = rscale0*r
    twol=2
    print(compute_duarray_recursive(x, y, z, psi, r, rscale0, twol))
main()
