#!/usr/bin/env python3
import pickle
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from phonopy import load
from ase.atoms import Atoms
from ase.io import write
from ase.cell import Cell
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from itertools import chain
from pyDFTutils.ase_utils.kpoints import kpath, get_path_special_points


def plot_phonon_deprecated(kpath,path='./', yaml_fname='phonopy_params.yaml', color='blue', unit="cm^-1"):
    phonon=load(phonopy_yaml=os.path.join(path,yaml_fname))
    cell=phonon._primitive.cell
    kpts,xs,xs_special,names=kpath(cell,npoints=300, path=kpath)
    phonon.symmetrize_force_constants()
    phonon.set_qpoints_phonon(kpts, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    for i in range(freqs.shape[1]):
        plt.plot(xs, freqs[:, i]*33.356,color=color,linewidth=1.3,alpha=0.8)
    plt.xlim(xs[0],xs[-1])
    for x in xs_special:
        plt.axvline(x,color='gray',alpha=0.7)
    plt.xticks(xs_special, names)
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.show()

def plot_phonon(path='./',color='blue', unit="cm^-1", knames=None, kvectors=None, npoints=100, figname="phonon.pdf", show=True):
    #phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
    phonon=load(phonopy_yaml=os.path.join(path,'phonopy_params.yaml'))
    cell=Cell(phonon._primitive.cell)
    knames, kpoints = get_path_special_points(cell, knames)
    knames = list(chain.from_iterable(knames))
    phonon.symmetrize_force_constants()
    qpoints, connections = get_band_qpoints_and_path_connections(kpoints, npoints=100)
    phonon.run_band_structure(
        paths=qpoints,
        with_eigenvectors=True,
        with_group_velocities=False,
        #is_band_connection=False,
        path_connections=connections,
        labels=knames,
        is_legacy_plot=False,)
    band=phonon.get_band_structure()
    phonon.write_yaml_band_structure(filename="band.yaml")
    phonon.plot_band_structure()
    if figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()


if __name__=='__main__':
    plot_phonon()
