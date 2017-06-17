#!/usr/bin/env python

from ase.io.xyz import read_xyz
from ase.io import read
from collections import OrderedDict
from ase import Atoms
from ase_utils import symbol_number
import numpy as np
import numpy.linalg as nl
import itertools


def read_win(filename='wannier90.up.win', poscar=None):
    # read cell
    cell = []
    with open(filename) as infile:
        start = False
        for line in infile:
            if line.strip().startswith('begin unit_cell_cart'):
                start = True
                continue
            if line.strip().startswith('end unit_cell_cart'):
                start = False
                break
            if start:
                vec = map(float, line.strip().split())
                cell.append(vec)

    # read positons
    chemical_symbols = []
    positions = []
    with open(filename) as infile:
        start = False
        for line in infile:
            if line.strip().startswith('begin atoms_cart'):
                start = True
                continue
            if line.strip().startswith('end atoms_cart'):
                start = False
                break
            if start:
                words = line.strip().split()
                sym = words[0]
                pos = map(float, words[1:])
                chemical_symbols.append(sym)
                positions.append(pos)

    if poscar is not None:
        atoms = read(poscar)
    else:
        atoms = Atoms(symbols=chemical_symbols, positions=positions, cell=cell)

    # read projection
    projections = OrderedDict()
    with open(filename) as infile:
        start = False
        for line in infile:
            if line.strip().startswith('begin projections'):
                start = True
                continue
            if line.strip().startswith('end projections'):
                start = False
                break
            if start:
                elem, orbs_text = line.strip().split(':')
                orbs = [x.strip() for x in orbs_text.split(',')]
                projections[elem] = orbs
    return atoms, projections


def projections_to_basis(atoms, projections):
    """
    projection_dict, e.g. {'Mn':['dxy','dxz']} --> basis, e.g. [('Mn1','dxy'),('Mn1',dxz),('Mn2','dxy'),('Mn2',dxz)]
    """
    #_,projections=read_win()
    basis = []
    symbols = atoms.get_chemical_symbols()
    for sym in projections:
        nsym = symbols.count(sym)
        for i in range(1, nsym + 1):
            for orb in projections[sym]:
                basis.append(('%s%s' % (sym, i), orb))
    return basis


def projection_dict_by_site_to_basis(atoms, projections):
    """
    e.g. {('Mn1','Mn2'):['dxy','dxz']} --> basis, e.g. [('Mn1','dxy'),('Mn1',dxz),('Mn2','dxy'),('Mn2',dxz)]
    """
    basis = []
    for key, val in projections.items():
        basis += list(itertools.product(key, val))
    return basis


def check_center(wannier_centers,
                 basis,
                 atoms,
                 max_distance=0.5,
                 orig_cell=True):
    """
    check whether the wannier centers are near the atoms of the basis.
    """
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    sdict = symbol_number(atoms)
    all_good = True
    for wann_pos, bas in zip(wannier_centers, basis):
        atom_pos = positions[sdict[bas[0]]]
        if orig_cell:
            scaled_dis = np.dot((atom_pos - wann_pos), nl.inv(cell))
            ndis = []
            for ix in scaled_dis:
                if ix > 0.5:
                    ix = ix - 1
                ndis.append(ix)
            scaled_dis = np.array(ndis)
            distance = nl.norm(np.dot(scaled_dis, cell))

        else:
            distance = nl.norm(atom_pos - wann_pos)

        if distance > max_distance:
            print("Bad,orb:", bas, wann_pos, atom_pos)
            all_good = False
        # assert distance<0.1
    return all_good


def read_centers(filename=None):
    # read
    atoms = list(read_xyz(filename))[0]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # split X (wannier centers) and atoms.
    # map( (filter( (lambda x: x[0]=='X'), zip(symbols,positions))))
    # wannier_centers= [xpos[1] for xpos in zip(symbols,positions) if xpos[0] ]
    wannier_centers = [
        xpos[1] for xpos in zip(symbols, positions) if xpos[0] == 'X'
    ]
    asymbols = [xpos[0] for xpos in zip(symbols, positions) if xpos[0] != 'X']
    aposes = [xpos[1] for xpos in zip(symbols, positions) if xpos[0] != 'X']
    cell = atoms.get_cell()
    atoms = Atoms(symbols=asymbols, positions=aposes, cell=cell)

    return wannier_centers, atoms



if __name__ == '__main__':
    atoms, projections = read_win()
    basis = projections_to_basis(atoms, projections)
    wann_centers, atoms = read_centers()
    check_center(wann_centers, basis, atoms)
