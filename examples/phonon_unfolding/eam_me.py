#!/usr/bin/env python
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from ase.units import Bohr
from frozenphonon import calculate_phonon
import numpy as np
from ase.dft.kpoints import *
from ase.build import bulk
import matplotlib.pyplot as plt

from pyDFTutils.unfolding.phonon_unfolder import phonon_unfolder
#from minimulti.unfolding.phonon_unfolder import phonon_unfolder
from pyDFTutils.phonon.plotphon import plot_band_weight
#from ase_utils import vesta_view


def kpath():
    #atoms = FaceCenteredCubic(size=(1,1,1), symbol="Cu", pbc=True)
    atoms = bulk('Cu', 'fcc', a=3.61)
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    GXW = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(GXW, atoms.cell, 700)
    names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    return kpts, x, X, names


def get_phonon_prim(ax=None):
    atoms = bulk('Cu', 'fcc', a=3.61)
    # vesta_view(atoms)
    calc = EMT()
    atoms.set_calculator(calc)
    phonon = calculate_phonon(
        atoms, calc, ndim=np.eye(3) * 3, primitive_matrix=np.diag([1] * 3))
    kpts, x, X, names = kpath()
    phonon.set_qpoints_phonon(kpts, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    # print freqs
    if ax is None:
        fig, ax = plt.subplot()
    for i in range(3):
        ax.plot(x, freqs[:, i]*33.356, color='black',
                linewidth=0.5, linestyle='-')
    #ax.set_xticks(X, names)
    ax.set_ylabel('Frequency (cm$^{-1}$)')
    ax.set_xlim([X[0], X[-1]])
    ax.set_ylim([0, 350])
    # plt.show()


def phonon_unfold():
    atoms = FaceCenteredCubic(size=(1, 1, 1), symbol="Cu", pbc=True)
    symbols = atoms.get_chemical_symbols()
    #symbols[-1] = 'Ag'
    atoms.set_chemical_symbols(symbols)
    calc = EMT()
    atoms.set_calculator(calc)
    phonon = calculate_phonon(
        atoms, calc, ndim=np.eye(3) * 2, primitive_matrix=np.eye(3)/1.0)

    kpts, x, X, names = kpath()
    kpts = [
        np.dot(k,
               np.linalg.inv(
                   (np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0)))
        for k in kpts
    ]
    phonon.set_qpoints_phonon(kpts, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()

    sc = atoms
    sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    spos = sc.get_scaled_positions()

    uf = phonon_unfolder(atoms, sc_mat, eigvecs, kpts)
    weights = uf.get_weights()

    ax = None
    ax = plot_band_weight([list(x)]*freqs.shape[1], freqs.T*33.356,
                          weights[:, :].T*0.98+0.01, xticks=[names, X], axis=ax)

    # print freqs
    # for i in range(freqs.shape[1]):
    #    plt.plot(x, freqs[:, i], color='blue', alpha=0.2, linewidth=2.5)

    plt.xticks(X, names)
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.ylim([0, 350])
    plt.title('with defect (1/4) (method: reciprocal)')

    get_phonon_prim(ax)
    plt.savefig('defrec4.png')

    plt.show()


def test():
    # get_phonon_prim()
    phonon_unfold()


test()
