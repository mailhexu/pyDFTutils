from collections import OrderedDict
from ase.io import read
from ase.units import Bohr
from pyDFTutils.ase_utils import symbol_number
from pyDFTutils.ase_utils.kpoints import ir_kpts
from pyDFTutils.wannier90.wannier import read_basis
import pythtb
import os
import numpy as np
from green import green_J



class exchange():
    def __init__(self,
                 path,
                 atoms=None,
                 min_hopping_norm=1e-4,
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

    def to_green_J(self):
        """
        generate green_J object
        """
        last_atom=None
        iatom=-1
        atom_orb_labels=None
        adict=OrderedDict()
        for symbol in self.basis:
            atom_sym, orb_sym, _, _ = symbol.split('|')
            if atom_sym in adict:
                adict[atom_sym].append(orb_sym)
            else:
                adict[atom_sym]=[orb_sym]
        atom_index=list(range(len(adict.keys())))
        orb_labels=list(adict.values())
        print(atom_index)
        print(orb_labels)


        Hk=np.zeros((2, len(self.kpts), self.nwann, self.nwann,), dtype='complex128')
        for ispin, model in enumerate([self.tbmodel_up, self.tbmodel_dn]):
            for ik, k in enumerate(self.kpts):
                model._gen_ham(k)
                Hk[ispin, ik, :,:]=model._gen_ham(k)
        g=green_J(
                Hk,  # Hk[ispin, ikpt, iorb, jorb ]
                positions=self.atoms.get_scaled_positions(),  # [iatom, xyz]
                kpts=self.kpts,  # [ikpt, 3]
                kweights=self.kweights,  # [ikpt]
                atom_index=atom_index,  # e.g. [0,1,2]
                orb_labels=orb_labels,  # e.g. [('px','py', 'pz'), ('dxy','dyz',...)]
                efermi=self.efermi,  # efermi
)
        return g



    def write_system(self):
        """
        write system.am
        """
        # 1. hash. removed from exchange code
        text = '&hash\n' + '  0\n\n'

        # 2: cell.
        cell_text = '&cell\n'
        cell = self.atoms.get_cell()
        a=cell[0,0]
        cell_text += '%s\n'%(a/Bohr)
        for l in cell:
            cell_text += '  ' + '  '.join(
                ['%.5f' % (x/a ) for x in l]) + '\n'
        text += cell_text + '\n'

        # 3. positions.
        atoms_text = '&atoms\n'
        atoms_text += str(len(self.atoms)) + '\n'
        sdict = symbol_number(self.atoms)
        for s in sdict:
            atoms_text += s + '  '
            pos = self.atoms.get_scaled_positions()[sdict[s]] / Bohr
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
            atom_num = sdict[atom_sym]+1
            orbitals = block_dict[key]
            block_dim = len(orbitals)
            basis_text += '%s  %s %s %s  %s    ' % (
                atom_sym, atom_num, l_sym, block_dim, block_start) + ' '.join(
                    [str(orb) for orb in orbitals]) + '\n'
            block_start += block_dim

        text += basis_text
        print(text)

        with open('system.am','w') as myfile:
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
        for w,k in zip(self.kweights, self.kpts):
            text+= ' %s  '%(w) +'  '.join(map(str,k)) +'\n'

        text+='\n'


        ## 6. hamiltonian.
        text += '&hamiltonian\n'
        print(text)

        for model in [self.tbmodel_up, self.tbmodel_dn]:
            for k in self.kpts:
                ham=model._gen_ham(k)
                for i in range(self.nwann):
                    for j in range(i,self.nwann): # only upper triangle written.
                        hij=ham[i,j]
                        text+= '%s %s\n'%(hij.real, hij.imag)
        with open('hamilt.am','w') as myfile:
            myfile.write(text)




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

def test():
    atoms = read('./POSCAR')
    exchTB = exchange('./',nelect=164,efermi=-0.7074)
    exchTB.set_atoms(atoms)
    exchTB.set_kpts(mpgrid=[6,6,4])
    exchTB.write_system()
    exchTB.write_hamil()
    exit()
    g=exchTB.to_green_J()
    g.set(emin=-15,
          emax=0.1,
          height=0.5,
          nz1=30,
          nz2=100,
          nz3=30)
    g.prepare()
    g.get_occupations()
    print("J:")
    J1=g.get_J(15, 12, (0,0,0))
    J2=g.get_J(15, 13, (0,0,0))
    J3=g.get_J(15, 14, (0,0,0))
    J4=g.get_J(15, 15, (0,0,1))
    print(J1, J2, J3, J4)

    J1=g.get_J(14, 12, (0,0,0))
    J2=g.get_J(14, 13, (0,0,0))
    J3=g.get_J(14, 14, (0,0,1))
    J4=g.get_J(14, 15, (0,0,0))
    print(J1, J2, J3, J4)

    J5=g.get_J(15, 1, (0,0,0))
    print(J5)

test()
