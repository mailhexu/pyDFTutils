#!/usr/bin/env python
"""
utils for vasp only.
"""
import numpy as np
import os.path
from ase.io import read
from ase.io.vasp import read_vasp
from pyDFTutils.ase_utils.symbol import symbol_number, symnum_to_sym
from ase.atoms import Atoms
import re


def read_sort(sort_file='ase-sort.dat'):
    """
    get the second column of the file "ase-sort.dat"
    """
    if not os.path.exists(sort_file):
        raise IOError("file %s not exisits" % sort_file)
    a = np.loadtxt(sort_file, dtype=int)
    if len(a.shape) == 1:
        return np.array([1])
    return a[:, 1]


def get_unsorted(arr, sort_list=None, sort_file='./ase-sort.dat'):
    """
    arr: the array to be unsorted
    sort_list: column 2 of the file 'ase-sort.dat'.
    sort_file: ase-sort.dat or something like that.
    """
    arr = np.array(arr)
    if sort_list is None:
        sort_list = read_sort(sort_file)
    return arr[sort_list]


def read_poscar_and_unsort(filename='CONTCAR'):
    """
    read poscar and unsort. return the symbols,positions and cell in unsorted order.
    """
    atoms = read_vasp(filename)
    symbols = get_unsorted(atoms.get_chemical_symbols())
    positions = get_unsorted(atoms.get_positions())
    cell = atoms.get_cell()
    return symbols, positions, cell


def read_unsorted_poscar(filename='CONTCAR'):
    s, p, c = read_poscar_and_unsort(filename=filename)
    atoms = Atoms(symbols=s, positions=p, cell=c)
    return atoms


def get_electrons(filename='POTCAR'):
    """
    get dict {symbol: valence} from POTCAR
    """
    nelect = dict()
    lines = open(filename).readlines()
    for n, line in enumerate(lines):
        if line.find('TITEL') != -1:
            symbol = line.split('=')[1].split()[1].split('_')[0].strip()
            valence = float(
                lines[n + 4].split(';')[1].split('=')[1].split()[0].strip())
            nelect[symbol] = valence
    return nelect


def get_symdict(filename='POSCAR', atoms=None):
    """
    get a symbol_number: index dict.
    """
    if filename is None and atoms is not None:
        syms = atoms.get_chemical_symbols()
    elif filename is not None and atoms is None:
        syms = read(filename).get_chemical_symbols()

    symdict = symbol_number(syms)
    return symdict


def read_charges(fname='OUTCAR'):
    """
    Get the lines containing the charge of each atom from the OUTCAR
    """
    with open(fname) as myfile:
        start = False
        text = ''
        for line in myfile:
            if line.strip() == 'total charge':
                start = True
                text = ''
            if start:
                text += line
            if line.startswith('tot'):
                start = False
    return text


def read_efermi(filename='OUTCAR'):
    """
    read the fermi energy from OUTCAR.
    """
    text = open(filename, 'r').read()
    m = re.search(r'fermi\s*:\s*([-+]?\d*\.\d*)', text)
    if m:
        t = m.group(1)
    else:
        raise ValueError('fermi energy not found')
    try:
        t = float(t)
    except Exception as E:
        raise ValueError("t:%s can not be converted to float." % t)
    return t


def read_nband(filename='OUTCAR'):
    """
    read the fermi energy from OUTCAR.
    """
    text = open(filename, 'r').read()
    # number of bands    NBANDS=     96
    m = re.search(r'NBANDS\s*=\s*(\d*)', text)
    if m:
        t = m.group(1)
    else:
        raise ValueError('NBANDS not found')
    try:
        t = int(t)
    except Exception as E:
        raise ValueError("t:%s can not be converted to int." % t)
    return t


def read_magx(fname='OUTCAR'):
    """
    Get the lines containing the magnetic moments of each atom from the OUTCAR
    """
    with open(fname) as myfile:
        start = False
        text = ''
        for line in myfile:
            if line.startswith(' magnetization (x)'):
                start = True
                text = ''
            if start:
                text += line
            if line.startswith('tot'):
                start = False
    return text


def check_converge(filename='log'):
    """
    check the log file to see if the calculation is converged.
    If converged, return the number of steps of iteration, else return False.
    """
    nstep = 0
    with open(filename) as myfile:
        for line in myfile:
            l = line
            try:
                nstep = int(line.strip().split()[0])
            except Exception:
                pass
        if l.strip().startswith('writing'):
            return nstep
        else:
            return False


if __name__ == '__main__':
    print(symnum_to_sym('1'))
