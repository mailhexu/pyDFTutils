"""
Plotting the magnon band structure for
2D square lattice. 
The equation can be found in PhysRevLett.86.5377 (2001) (end of the second page)
The Hamiltonian is defined in Eq. (1). 
"""

from TB2J.plot import plot_magnon_band
from ase.dft.kpoints import bandpath
import numpy as np
from ase.cell import Cell
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, sqrt
from TB2J.spinham.plot import group_band_path


def v(x):
    return cos(2*pi*x)


def get_dispersion(J1, J2, J3, Jc, h, k, l):
    """
    J1~J3: 1st/2nd/3rd NN J
    Jc: four-spin interaction parameter
    """
    AQ = J1-Jc/2.0-(J2-Jc/4)*(1.0-v(h)*v(k))-J3*(1-(v(2*h)+v(2*k))/2.0)
    BQ = (J1-Jc/2)*(v(h)+v(k))/2.0
    wQ = 2*sqrt(AQ**2-BQ**2)
    return wQ


def getk(cell=np.eye(3), knames=None, kvectors=None, supercell_matrix=None, npoints=50):
    if knames is None and kvectors is None:
        # fully automatic k-path
        bp = Cell(cell).bandpath(npoints=npoints)
        spk = bp.special_points
        xlist, kptlist, Xs, knames = group_band_path(bp)
    elif knames is not None and kvectors is None:
        # user specified kpath by name
        bp = Cell(cell).bandpath(knames, npoints=npoints)
        spk = bp.special_points
        kpts = bp.kpts
        xlist, kptlist, Xs, knames = group_band_path(bp)
    else:
        # user spcified kpath and kvector.
        kpts, x, Xs = bandpath(kvectors, cell, npoints)
        spk = dict(zip(knames, kvectors))
        xlist = [x]
        kptlist = [kpts]

    if supercell_matrix is not None:
        kptlist = [np.dot(k, supercell_matrix) for k in kptlist]
    print("High symmetry k-points:")
    for name, k in spk.items():
        if name == 'G':
            name = 'Gamma'
        print(f"{name}: {k}")

    return xlist, kptlist, Xs, knames


def plot_dispersion(J1, J2, J3, Jc):
    """
    J1~J3: 1st/2nd/3rd NN J
    Jc: four-spin interaction parameter
    """
    ax = plt.subplot()
    # Here change the k-path
    xlist, kptlist, Xs, knames = getk(
        knames=['(0.5,0.5)', '(0,0)',
                '(0.5,0)', '(0.5,0.25)'],
        kvectors=[[.5, .5, 0], [0, 0, 0],
                  [.5, 0, 0], [.5, .25, 0]])

    Es = []
    for k in kptlist[0]:
        Es.append(get_dispersion(J1, J2, J3, Jc, k[0], k[1], k[2]))
    ax.plot(xlist[0], Es)
    ax.set_xticks(Xs)
    ax.set_xticklabels(knames)
    for x in Xs:
        ax.axvline(x, color="gray")
    ax.set_ylim(min(Es), max(Es)+30)
    ax.set_xlim(min(xlist[0]), max(xlist[0]))
    return ax


ax = plot_dispersion(J1=72.5, J2=-7.4, J3=0, Jc=0)
plt.savefig("magnon_sq.pdf")
plt.savefig("magnon_sq.png")
plt.show()
