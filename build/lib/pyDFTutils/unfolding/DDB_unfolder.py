#!/usr/bin/env python
import abipy.abilab as abilab
import numpy as np
from ase.build import bulk
from ase.dft.kpoints import get_special_points, bandpath
from pyDFTutils.unfolding.phonon_unfolder import phonon_unfolder
from pyDFTutils.phonon.plotphon import plot_band_weight
import matplotlib.pyplot as plt
import sys

atomic_masses = np.array([
    1., 1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999,
    18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762,
    32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63,
    74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224,
    92.90637, 95.95, 97.90721, 101.07, 102.9055, 106.42, 107.8682, 112.414,
    114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.90545196, 137.327,
    138.90547, 140.116, 140.90766, 144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49,
    180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592,
    204.38, 207.2, 208.9804, 208.98243, 209.98715, 222.01758, 223.01974,
    226.02541, 227.02775, 232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083, 257.09511, 258.09843,
    259.101, 262.11, 267.122, 268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19, 289.194, 293.204, 293.208,
    294.214
])


def kpath():
    #DDB = abilab.abiopen('out_DDB')
    #struct = DDB.structure
    #atoms = DDB.structure.to_ase_atoms()
    atoms = bulk('Cu','fcc')
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    GXW = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(GXW, atoms.cell, 700)
    names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    return kpts, x, X, names, GXW

def displacement_cart_to_evec(displ_cart, masses, scaled_positions, qpoint=None, add_phase=True):
    """
    displ_cart: cartisien displacement. (atom1_x, atom1_y, atom1_z, atom2_x, ...)
    masses: masses of atoms.
    scaled_postions: scaled postions of atoms.
    qpoint: if phase needs to be added, qpoint must be given.
    add_phase: whether to add phase to the eigenvectors.
    """
    if add_phase and qpoint is None:
        raise ValueError('qpoint must be given if adding phase is needed')
    m = np.sqrt(np.kron(masses,[1,1,1]))
    evec=displ_cart *m
    if add_phase:
        phase = [np.exp(-2j*np.pi*np.dot(pos,qpoint)) for pos in scaled_positions]
        phase = np.kron(phase,[1,1,1])
        evec*=phase
        evec /= np.linalg.norm(evec)
    return evec




def DDB_unfolder(DDB_fname, kpath_bounds,sc_mat,knames=None, kx=None,dipdip=1):
    DDB = abilab.abiopen(DDB_fname)
    struct = DDB.structure
    atoms = DDB.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]

    #print numbers
    #print cell
    #print scaled_positions

    #print kpath_bounds

    phbst, phdos = DDB.anaget_phbst_and_phdos_files(
        nqsmall=2,
        asr=1,
        chneut=1,
        dipdip=dipdip,
        verbose=1,
        ndivsm=40,
        lo_to_splitting=True,
        qptbounds=kpath_bounds,
        )
    #phbst.plot_phbands()
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    m = np.sqrt(np.kron(masses,[1,1,1]))
    #positions=np.kron(scaled_positions,[1,1,1])
    
    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            phmode = phbst.get_phmode(qpt, ibranch)
            evals[iqpt, ibranch] = phmode.freq
            #evec=phmode.displ_cart *m
            #phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            #phase = np.kron(phase,[1,1,1])
            #evec*=phase
            #evec /= np.linalg.norm(evec)
            evec=displacement_cart_to_evec(phmode.displ_cart, masses, scaled_positions, qpoint=qpt, add_phase=True)
            evecs[iqpt,:,ibranch] = evec
            
    uf = phonon_unfolder(atoms,sc_mat,evecs,qpoints,phase=False)
    weights = uf.get_weights()
    x=np.arange(nqpts)
    freqs=evals
    xpts=[]
    for ix, xx in enumerate(x):
        for q in kpath_bounds:
            if np.sum((np.array(qpoints[ix])-np.array(q))**2)<0.00001 and ix not in xpts:
                xpts.append(ix)
    if knames is None:
        knames=[str(k) for k in kpath_bounds]

    #names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*33.356,weights[:,:].T*0.98+0.01,xticks=[names,X],axis=ax)
    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*8065.6,weights[:,:].T*0.98+0.01,xticks=[knames,xpts],style='alpha')
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*8065.6,weights[:,:].T*0.98+0.000001,xticks=[knames, kx],style='alpha' )

    #plt.show()
    return ax

def nc_unfolder(fname, sc_mat, kx=None, knames=None ,ghost_atoms=None, plot_width=False, weight_multiplied_by=None):
    ncfile=abilab.abiopen(fname)
    struct = ncfile.structure
    atoms = ncfile.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[i] for i in numbers]

    #print numbers
    #print cell
    #print scaled_positions



    #print kpath_bounds

    phbst = ncfile.phbands
    #phbst.plot_phbands()
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    m = np.sqrt(np.kron(masses,[1,1,1]))
    #positions=np.kron(scaled_positions,[1,1,1])
    freqs=phbst.phfreqs
    displ_carts=phbst.phdispl_cart
    
    for iqpt, qpt in enumerate(qpoints):
        print(iqpt, qpt)
        for ibranch in range(nbranch):
            #phmode = ncfile.get_phmode(qpt, ibranch)
            #print(2)
            evals[iqpt, ibranch] = freqs[iqpt, ibranch]
            #evec=phmode.displ_cart *m
            #phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            #phase = np.kron(phase,[1,1,1])
            #evec*=phase
            #evec /= np.linalg.norm(evec)
            evec=displacement_cart_to_evec(displ_carts[iqpt, ibranch,: ], masses, scaled_positions, qpoint=qpt, add_phase=True)
            evecs[iqpt,:,ibranch] = evec
            
    uf = phonon_unfolder(atoms,sc_mat,evecs,qpoints,phase=False, ghost_atoms=ghost_atoms)
    weights = uf.get_weights()
    if plot_width:
        weights=(weights*(1.0-weights))**(0.5)
    if weight_multiplied_by is not None:
        weights=weights*weight_multiplied_by
    x=np.arange(nqpts)
    freqs=evals
    #names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*33.356,weights[:,:].T*0.98+0.01,xticks=[names,X],axis=ax)
    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*8065.6,weights[:,:].T*0.98+0.000001,xticks=[knames, kx],style='alpha' )
    #plt.show()
    return ax

def main():
    #sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    #sc_mat=np.array([[0,1,1],[1,0,1],[1,1,0]])
    sc_mat=np.eye(3)
    #points = kpath()[-1]
    points=np.array([(0,0,0),(0,.5,0),(0.5,0.5,0),[.5,.5,.5],[0,0,0]])
    DDB_unfolder(DDB_fname='out_DDB', kpath_bounds = [np.dot(k, sc_mat) for k in points],sc_mat=sc_mat)


