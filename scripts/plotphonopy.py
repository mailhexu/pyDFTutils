#!/usr/bin/env python3
import pickle
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from pyDFTutils.ase_utils.kpoints import kpath
from phonopy import load
from ase.atoms import Atoms
from ase.io import write
from spglib import spglib
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

def plot_phonon(path='./',color='blue', unit="cm^-1"):
    #phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
    phonon=load(phonopy_yaml=os.path.join(path,'phonopy_params.yaml'))
    cell=phonon._primitive.cell
    #kpts,xs,xs_special,names=kpath(cell,npoints=10, path='GPZQGFZL')
    kpts,xs,xs_special,names=kpath(cell,npoints=400, path='GZ')
    #print(kpts)
    #phonon.symmetrize_force_constants()
    #phonon.symmetrize_force_constants_by_space_group()
    phonon.set_qpoints_phonon(kpts, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    path=np.array([[(0,0,0), (0,0,.5)]])
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=10)
    phonon.run_band_structure(
        paths=qpoints,
        with_eigenvectors=True,
        with_group_velocities=False,
        #is_band_connection=False,
        path_connections=connections,
        labels="GZ",
        is_legacy_plot=False,)
    band=phonon.get_band_structure()
    phonon.write_yaml_band_structure(filename="band.yaml")
    #band.write_yaml()
    #print(freqs)
    #print(eigvecs.shape)
    #print(eigvecs[9,-1,:])
    #weight=np.zeros_like(freqs)
    #for ki in range(freqs.shape[0]):
    #    for i in range(freqs.shape[1]):
    #        weight[ki,i]=np.linalg.norm(eigvecs[ki,i,-3:])
    for i in range(freqs.shape[1]):
        #plt.plot(xs, freqs[:, i]*33.356,color=color,linewidth=1.3,alpha=0.8)
        plt.plot(xs, freqs[:, i],color=color,linewidth=1.3,alpha=0.8)
        #plt.plot(xs, freqs[:, i]*33.356/8.065,color=color,linewidth=1.3,alpha=0.8)
        #plt.scatter(x, freqs[:, i]*33.356,s=weight[:,i]*10,color='blue',alpha=0.3)
    plt.xlim(xs[0],xs[-1])
    for x in xs_special:
        plt.axvline(x,color='gray',alpha=0.7)
    plt.xticks(xs_special, names)
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.show()

if __name__=='__main__':
    plot_phonon()
