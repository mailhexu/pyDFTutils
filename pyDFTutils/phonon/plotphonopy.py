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

def plot_phonon(kpath,path='./', yaml_fname='phonopy_params.yaml', color='blue', unit="cm^-1"):
    phonon=load(phonopy_yaml=os.path.join(path,))
    cell=phonon._primitive.cell
    kpts,xs,xs_special,names=kpath(cell,npoints=300, path=kpath)
    phonon.symmetrize_force_constants()
    #phonon.symmetrize_force_constants_by_space_group()
    phonon.set_qpoints_phonon(kpts, is_eigenvectors=True)
    #phonon.symmetrize_force_constants()
    freqs, eigvecs = phonon.get_qpoints_phonon()
    #print(freqs)
    #print(eigvecs.shape)
    #print(eigvecs[9,-1,:])
    #weight=np.zeros_like(freqs)
    #for ki in range(freqs.shape[0]):
    #    for i in range(freqs.shape[1]):
    #        weight[ki,i]=np.linalg.norm(eigvecs[ki,i,-3:])
    for i in range(freqs.shape[1]):
        plt.plot(xs, freqs[:, i]*33.356,color=color,linewidth=1.3,alpha=0.8)
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
