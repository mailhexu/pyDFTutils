#!/usr/bin/env python
from pyDFTutils.perovskite.lattice_factory import PerovskiteCubic
from pyDFTutils.ase_utils import my_write_vasp,normalize, vesta_view, set_element_mag
from ase.io.vasp import read_vasp
try:
    from ase.atoms import string2symbols
except:
    from ase.symbols import string2symbols
import numpy as np
from ase.build import make_supercell

def gen222(name=None,
           A='Sr',
           B='Mn',
           O='O',
           latticeconstant=3.9,
           mag_order='FM',
           mag_atom=None,
           m=5,
           sort=True):
    if name is not None:
        symbols=string2symbols(name)
        A, B, O, _, _ = symbols

    atoms = PerovskiteCubic([A, B, O], latticeconstant=latticeconstant)
    atoms = atoms.repeat([2, 2, 2])
    if sort:
        my_write_vasp('UCPOSCAR', atoms, vasp5=True, sort=True)
        atoms = read_vasp('UCPOSCAR')

    spin_dn = {
        'FM': [],
        'A': [0, 1, 4, 5],
        'C': [0, 2, 5, 7],
        'G': [0, 3, 5, 6]
    }
    if mag_order != 'PM':
        mag = np.ones(8)
        mag[np.array(spin_dn[mag_order], int)] = -1.0
        if mag_atom is None:
            atoms = set_element_mag(atoms, B, mag * m)
        else:
            atoms = set_element_mag(atoms, mag_atom, mag * m)
    return atoms


def gen_primitive(name=None,A=None,B=None,O=None, latticeconstant=3.9, mag_order='FM',mag_atom=None, m=5):
    """
    generate primitive cell with magnetic order.

    Parameters:
    ---------------
    name: string
        ABO3, eg. BiFeO3, CsPbF3
    """
    if name is not None:
        symbols=string2symbols(name)
        A, B, O, _, _ = symbols
    atoms = PerovskiteCubic([A, B, O], latticeconstant=latticeconstant)
    direction_dict = {
        'A': ([1, 0, 0], [0, 1, 0], [0, 0, 2]),
        'C': ([1, -1, 0], [1, 1, 0], [0, 0, 1]),
        'G': ([0, 1, 1], [1, 0, 1], [1, 1, 0]),
        'FM': np.eye(3),
    }
    size_dict = {'A': (1, 1, 2), 'C': (1, 1, 1), 'G': (1, 1, 1)}
    A, B, O = atoms.get_chemical_symbols()[0:3]
    if mag_order == 'PM':
        atoms = atoms
    elif mag_order == 'FM':
        atoms = atoms
        if mag_atom is None:
            atoms = set_element_mag(atoms, B, [m])
        else:
            atoms = set_element_mag(atoms, mag_atom, [m])
    else:
        atoms.translate([0.045] * 3)
        atoms = normalize(atoms)
        atoms = make_supercell(atoms, direction_dict[mag_order])
        atoms.translate([-0.045] * 3)
        if mag_atom is None:
            atoms = set_element_mag(atoms, B, [m])
        else:
            atoms = set_element_mag(atoms, mag_atom, [m])
    return atoms


if __name__ == '__main__':
    atoms = gen_primitive(name='LaMnO3',mag_order='G')
    vesta_view(atoms)
