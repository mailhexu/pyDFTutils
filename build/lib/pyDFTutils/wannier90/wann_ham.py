#!/usr/bin/env python

import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.units import Bohr
from pyDFTutils.ase_utils.ase_utils import symbol_number
from pyDFTutils.ase_utils.kpoints import ir_kpts
from .wannier import read_basis
from pyDFTutils.wannier90 import pythtb_forj as pythtb
from pyDFTutils.vasp.vasp_utils import read_efermi
#import pythtb
from pyDFTutils.plot.wannier_band_plot import plot_band_weight


class anatb():
    """
    analyze the tight binding model.
    """

    def __init__(self, tbmodel, kpts=None, kweights=None):
        self.tbmodel = tbmodel
        self.kpts = kpts
        self.kweights = kweights
        if kpts is not None and kweights is None:
            nkpts = len(kpts)
            self.kweights = np.ones(nkpts, dtype='float') / nkpts
        self.norbs = self.tbmodel.get_num_orbitals()

    def calc_cohp_k(self, ham_k, evec_kj):
        """
        calculate COHP for a wavefunction at a point.
        Parameters:
        --------------
        ham_k: The hamiltonian at k. (ndim*ndim), where ndim is the number of wannier functions.
        evec_kj: the jth eigenvector at k point

        Return:
        --------------
        a matrix huv, u&v are the indices of wannier functions.
        """
        cohp_kj = np.outer(np.conj(evec_kj), evec_kj) * ham_k
        #cohp_kj = np.outer(evec_kj, evec_kj) * ham_k
        return np.real(cohp_kj)

    def calc_cohp_allk(self, kpts=None, iblock=None, jblock=None):
        """
        calculate all COHPs.
        """
        nkpts = len(self.kpts)
        if iblock is None:
            iblock = range(self.norbs)
        if jblock is None:
            jblock = range(self.norbs)
        self.evals = np.zeros(( self.norbs,nkpts))
        self.cohp = np.zeros((nkpts, self.norbs, len(iblock), len(jblock)))
        ## [ikpt,iband, iorb, iorb]
        for ik, k in enumerate(self.kpts):
            ham_k = self.tbmodel._gen_ham(k)
            evals_k, evecs_k = self.tbmodel._sol_ham(ham_k, eig_vectors=True)
            self.evals[:,ik] = evals_k
            ## for each kpt,band there is a cohp matrix.
            for iband in range(self.norbs):
                ## note that evec[i,:] is the ith eigenvector
                evec = evecs_k[iband, :]
                self.cohp[ik, iband] = self.calc_cohp_k(ham_k, evec)
        return self.cohp

    def get_cohp_pair(self, i, j):
        return self.cohp[:, :, i, j]

    def get_cohp_block_pair(self, iblock, jblock):
        iblock = np.array(iblock, dtype=int)
        jblock = np.array(jblock, dtype=int)
        iiblock=np.array(list(set(iblock)&set(jblock)),dtype=int) # for removing diagonal terms.
        I, J = np.meshgrid(iblock, jblock)
        # print(self.cohp.shape)
        #return np.sum(self.cohp[:, :, I, J], axis=(2,3))
        return np.einsum('ijkl->ij', self.cohp[:,:,I,J]) - np.einsum('ijk->ij',
                self.cohp[:,:,iiblock,iiblock])

    def get_cohp_all_pair(self):
        return self.get_cohp_block_pair(range(self.norbs), range(self.norbs))

    def get_cohp_density(self, kpts=None, kweights=None, emin=-20, emax=20):
        """
        cohp(E)= sum_k cohp(k) (\delta(Ek-E))
        """
        if kpts is None:
            kpts = self.kpts
        if kweights is None:
            kweights = self.kweights

    def get_COHP_energy(self):
        """
        COHP as function of energy.
        """
        # raise NotImplementedError('COHP density has not been implemented yet.')
        pass

    def plot_COHP_fatband(self,kpts=None,k_x=None,iblock=None,jblock=None,show=False,efermi=None, axis=None,**kwargs):
        self.calc_cohp_allk(kpts=kpts)
        if iblock is None:
            wks = self.get_cohp_all_pair()
        else:
            wks = self.get_cohp_block_pair(iblock,jblock)
        wks = np.moveaxis(wks, 0, -1)
        kslist = [k_x] * self.norbs
        ekslist = self.evals
        axis = plot_band_weight(
            kslist,
            ekslist,
            wkslist=wks,
            efermi=efermi,
            yrange=None,
            style='color',
            color='blue',
            width=10,
            axis=axis,
            **kwargs)
        axis.set_ylabel('Energy (eV)')
        if show:
            plt.show()
        return axis


class wann_ham():
    def __init__(self,
                 path,
                 atoms=None,
                 min_hopping_norm=1e-3,
                 max_distance=None,
                 nelect=0,
                 efermi=0):
        self.path = path
        self.tb_up = pythtb.w90(path, 'wannier90.up')
        self.tb_dn = pythtb.w90(path, 'wannier90.dn')
        self.tbmodel_up = self.tb_up.model(
            min_hopping_norm=min_hopping_norm, max_distance=max_distance)
        self.tbmodel_dn = self.tb_dn.model(
            min_hopping_norm=min_hopping_norm, max_distance=max_distance)
        self.efermi = efermi
        self.nelect = nelect
        self.atoms = atoms
        self.basis = None
        self.kpts = None
        self.kweights = None

        basis_fname = os.path.join(self.path, 'basis.txt')
        if os.path.exists(basis_fname):
            self.basis = read_basis(basis_fname)
        self.nwann = len(self.basis.keys())
        self.block_dict = basis_to_block(self.basis)

    def set_kpts(self, mpgrid=[6, 6, 6], ir=False):
        """
        set the kpoint mesh.
        """
        assert (
            self.atoms is not None
        ), 'should set atomic structure first (use self.set_atoms(atoms)).'
        self.kpts, self.kweights = ir_kpts(
            self.atoms, mpgrid, is_shift=[0, 0, 0], ir=ir)

    def set_basis(self, basis):
        self.basis = basis

    def set_atoms(self, atoms):
        """
        set the atomic structure.
        """
        self.atoms = atoms

    def set_efermi(self, efermi):
        self.efermi = efermi

    def write_system(self):
        """
        write system.am
        """
        # 1. hash. removed from exchange code
        text = '&hash\n' + '  0\n\n'

        # 2: cell.
        cell_text = '&cell\n'
        cell_text += '1.0\n'
        cell = self.atoms.get_cell()
        for l in cell:
            cell_text += '  ' + '  '.join(
                ['%.5f' % (x / Bohr) for x in l]) + '\n'
        text += cell_text + '\n'

        # 3. positions.
        atoms_text = '&atoms\n'
        atoms_text += str(len(self.atoms)) + '\n'
        sdict = symbol_number(self.atoms)
        for s in sdict:
            atoms_text += s + '  '
            pos = self.atoms.get_positions()[sdict[s]] / Bohr
            atoms_text += '  '.join(['%.5f' % x for x in pos]) + '\n'
        text += atoms_text + '\n'

        #4. nelect
        text += '&nelec\n  %s\n\n' % (self.nelect)

        #5. efermi
        text += '&efermi\n  %s\n\n' % (self.efermi)

        #6. basis
        basis_text = '&basis\n'
        ## number of wannier basis, number of atoms with wannier functions.
        basis_text += str(self.nwann) + '  ' + str(
            len(self.block_dict.keys())) + '\n'

        block_dict = self.block_dict
        block_start = 1
        for key in block_dict:
            atom_sym, l_sym = key
            atom_num = sdict[atom_sym] + 1
            orbitals = block_dict[key]
            block_dim = len(orbitals)
            basis_text += '%s  %s %s %s  %s    ' % (
                atom_sym, atom_num, l_sym, block_dim, block_start) + ' '.join(
                    [str(orb) for orb in orbitals]) + '\n'
            block_start += block_dim

        text += basis_text
        print(text)

        with open('system.am', 'w') as myfile:
            myfile.write(text)

    def write_hamil(self):
        """
        write 'hamilt.am'
        """
        text = ''
        ## 1. hash
        text += '&hash\n  0\n\n'
        ## 2. spin
        text += '&nspin\n2\n\n'

        ## 3. number of kpts.
        text += '&nkp\n  %s\n\n' % (len(self.kpts))

        ## 4. number of wannier basis.
        text += '&dim\n  %s\n\n' % (self.nwann)

        ## 5. kpoints
        text += '&kpoints\n'
        for w, k in zip(self.kweights, self.kpts):
            text += ' %s  ' % (w) + '  '.join(map(str, k)) + '\n'

        text += '\n'

        ## 6. hamiltonian.
        text += '&hamiltonian\n'
        print(text)

        for model in [self.tbmodel_up, self.tbmodel_dn]:
            for k in self.kpts:
                ham = model._gen_ham(k)
                for i in range(self.nwann):
                    for j in range(i, self.nwann):
                        hij = ham[i, j]
                        text += '%s %s\n' % (hij.real, hij.imag)
        with open('hamilt.am', 'w') as myfile:
            myfile.write(text)

    def write_input(self):
        """
        write input file to exchanges.def.in
        """
        with open('exchange.def.in', 'w+') as myfile:
            myfile.write("""&exchanges
emin = -15
emax = 0.1
height = 0.1
nz1 = 250
nz2 = 450
nz3 = 250
mode = 'distance'
distance = 1.5
/
""")

    def run_exchange(self):
        os.system('exchanges.x < exchange.def.in | tee exchanges.def.out')


def basis_to_block(bdict):

    orbs = [
        's', 'y', 'z', 'x', 'xy', 'yz', '3z^2-1', 'xz', 'x^2-y^2',
        'y(3x^2-y^2)', 'xyz', 'y(5z^2-1)', 'z(5z^2-3)', 'x(5z^2-1)',
        'z(x^2-y^2)', 'x(3y^2-x^2)'
    ]

    orbs_name = [
        's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2',
        'fy(3x^2-y^2)', 'fxyz', 'fy(5z^2-1)', 'fz(5z^2-3)', 'fx(5z^2-1)',
        'fz(x^2-y^2)', 'fx(3y^2-x^2)'
    ]

    orbs_dict = OrderedDict()
    for i, o in enumerate(orbs_name):
        orbs_dict[o] = i + 1
    blocks_dict = OrderedDict()
    for symbol in bdict:
        atoms_sym, orb_sym, _, _ = symbol.split('|')
        l_sym = orb_sym[0]
        key = (atoms_sym, l_sym)
        if key in blocks_dict:
            blocks_dict[key].append(orbs_dict[orb_sym])
        else:
            blocks_dict[key] = [orbs_dict[orb_sym]]
    return blocks_dict


#print(basis_to_block(read_basis('basis.txt')))


def exchange():
    atoms = read('./POSCAR')
    efermi = read_efermi('SCF/OUTCAR')
    exchTB = exchange('./', nelect=164, efermi=efermi)
    exchTB.set_atoms(atoms)
    exchTB.set_kpts(mpgrid=[5, 5, 5])
    exchTB.write_system()
    exchTB.write_input()
    exchTB.write_hamil()
    exchTB.run()
