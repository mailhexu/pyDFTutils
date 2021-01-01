#!/usr/bin/env python
import os
import numpy as np
from glob import glob
import json
from ase.data import chemical_symbols
from ase.atoms import Atoms
from collections import OrderedDict
import collections
import abipy.abilab as abilab
from abipy.abilab import Structure
from abipy.flowtk.abiobjects import LdauParams


def set_Hubbard_U(abi_inp, ldau_dict={}, ldau_type=1, unit='eV'):
    """
    set DFT+U parameters.
    """
    structure = abi_inp.structure
    symbols = structure.symbol_set
    luj_params = LdauParams(ldau_type, structure)
    for symbol in ldau_dict:
        if symbol in symbols:
            luj_params.luj_for_symbol(
                symbol,
                l=ldau_dict[symbol]['L'],
                u=ldau_dict[symbol]['U'],
                j=ldau_dict[symbol]['J'],
                unit=unit)
    u_vars = luj_params.to_abivars()
    abi_inp.set_vars(u_vars)
    return abi_inp


def set_spinat(abi_inp, magmoms):
    """
    set initial spin configuration
    """
    sdict = None
    if len(magmoms) == 0:
        return abi_inp
    elif isinstance(magmoms[0], collections.Iterable):
        if len(magmoms[0]) == 3:
            sdict = {'spinat': magmoms}
    else:
        sdict = {'spinat': [[0, 0, m] for m in magmoms]}
    if sdict is None:
        raise ValueError('magmoms should be 1D array or 3d array')
    abi_inp.set_vars(sdict)
    return abi_inp


pp_path = os.path.expanduser('~/.local/pp/abinit')


def find_pp(symbol, xc, family, label='',pp_path=pp_path):
    """
    find pseudo potential.
    """
    if family.lower() == 'jth':
        xcdict = {'LDA': 'LDA_PW', 'PBE': 'GGA_PBE', 'PBEsol':'GGA-PBESOL'}
        name = os.path.join('JTH-%s*' % xc,
                            '%s.%s-JTH%s.xml' % (symbol, xcdict[xc], label))
    elif family.lower() == 'gbrv':
        xcdict = {'LDA': 'lda', 'PBE': 'pbe', 'PBEsol':'pbe'}
        axc=xcdict[xc]
        name = os.path.join(pp_path, 'GBRV-%s/%s_%s%s*' %
                            (axc.upper(), symbol.lower(), axc.lower(), label))
    elif family.lower() == 'dojo':
        xcdict = {'LDA': 'PW', 'PBE': 'PBE', 'PBEsol': 'PBEsol'}
        if label == '':
            label = 'standard'
        pp_dir = os.path.join(pp_path, 'ONCVPSP-%s-PD*' % (xcdict[xc]))
        fname = glob(os.path.join(pp_dir, '%s.djson'%label))
        if len(fname)>0:
            pp_dict=json.load(open(fname[0]))['pseudos_metadata']
        name=os.path.join(pp_dir,symbol,pp_dict[symbol]['basename'])
        #if len(fname) > 0:
        #    print(fname)
        #    ppdict = {}
        #    with open(fname[0]) as myfile:
        #        lines = myfile.readlines()
        #    for line in lines:
        #        elem = line.strip().split('/')[0]
        #        name = os.path.join(pp_dir, line.strip())
        #        ppdict[elem] = name
        #    for i in range(57, 72):
        #        elem = chemical_symbols[i]
        #        ppdict[elem] = os.path.join(pp_dir,
        #                                    '%s/*f-in-core*.psp8' % elem)
        #name = ppdict[symbol]
        #print(name)

    pp = os.path.join(pp_path, name)
    names = glob(pp)

    if len(names) > 0 and os.path.isfile(names[0]):
        return names[0]
    else:
        raise ValueError("Can't find pseudopotential %s at %s." %
                         (name, pp_path))


def to_abi_structure(obj,magmoms=False):
    ms=None
    if isinstance(obj, str) and os.path.isfile(obj):
        structure = Structure.as_structure(obj)
    elif isinstance(obj, Structure):
        structure = obj
    elif isinstance(obj, Atoms):
        #cell=obj.get_cell()
        #acell0=np.linalg.norm(cell[0])
        #acell1=np.linalg.norm(cell[1])
        #acell2=np.linalg.norm(cell[2])
        #cell0=cell[0]/acell0
        #cell1=cell[1]/acell1
        #cell2=cell[2]/acell2
        #acell=[acell0, acell1, acell2]
        #rprim=[cell0, cell1, cell2]
        #xred=obj.get_scaled_positions()
        #znucl=list(set(obj.get_atomic_numbers()))
        #ntypat=len(znucl)
        #typat=[]
        #for z in obj.get_atomic_numbers():
        #    for i,n in enumerate(znucl):
        #        if z==n:
        #            typat.append(i)
        #structure = Structure.from_abivars(acell=acell, rprim=rprim, typat=typat, xred=xred, ntypat=ntypat, znucl=znucl)
        structure = Structure.from_ase_atoms(obj)
        if magmoms:
            ms=obj.get_initial_magnetic_moments()
    else:
        raise ValueError(
            'obj should be one of these:  abipy structure file name| abipy structure| ase atoms.'
        )
    if magmoms:
        return structure,ms
    else:
        return structure


def find_all_pp(obj, xc, family, label_dict={},pp_path=pp_path):
    symbols = []
    if isinstance(obj, collections.Iterable) and (
            not isinstance(obj, str)) and isinstance(obj[0], str):
        symbols = obj
    elif isinstance(obj, str) and os.path.isfile(obj):
        structure = Structure.as_structure(obj)
        symbols = set(structure.symbol_set)
    elif isinstance(obj, Structure):
        symbols = obj.symbol_set
    elif isinstance(obj, Atoms):
        for elem in obj.get_chemical_symbols():
            if elem not in symbols:
                symbols.append(elem)
    else:
        raise ValueError(
            'obj should be one of these: list of chemical symbols| abipy structure file name| abipy structure| ase atoms.'
        )
    ppdict = OrderedDict()
    for elem in symbols:
        if elem not in ppdict:
            if elem in label_dict:
                label = label_dict
            else:
                label = ''
            ppdict[elem] = find_pp(elem, xc, family, label,pp_path=pp_path)
    return list(ppdict.values())


def test_pp_finder():
    print(find_pp('La', 'LDA', 'jth'))
    #print(find_pp('Lu', 'LDA', 'jth', label='_fincore'))
    print(find_pp('Sn', 'LDA', 'jth', label='_sp'))
    print(find_pp('Ga', 'LDA', 'gbrv'))
    print(find_pp('Lu', 'LDA', 'gbrv', label='_fincore'))
    print(find_pp('Ca', 'LDA', 'dojo', label=''))
    #print(find_pp('Lu', 'PBEsol', 'dojo', label=''))
    #print(find_pp('Lu', 'PBEsol', 'dojo', label=''))

    print(find_all_pp(['Ba', 'Ti', 'O'], 'LDA', 'gbrv'))
    print(find_all_pp(['Sr', 'Mn', 'O'], 'LDA', 'dojo'))


#test_pp_finder()
