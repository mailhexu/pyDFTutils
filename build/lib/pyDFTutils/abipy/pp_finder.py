#!/usr/bin/env python
import os
from glob import glob
from ase.data import chemical_symbols
from collections import OrderedDict
import collections
import abipy.abilab as abilab


def set_Hubbard_U(abi_inp, ldau_dict={}, ldau_type=1,unit='eV'):
    """
    set DFT+U parameters.
    """
    structure = abi_inp.structure
    symbols = structure.symbol_set
    luj_params = abilab.LdauParams(ldau_type, structure)
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


def find_pp(symbol, xc, family, label=''):
    """
    find pseudo potential.
    """
    if family.lower() == 'jth':
        xcdict = {'LDA': 'LDA_PW', 'PBE': 'GGA_PBE'}
        name = os.path.join('JTH-%s*' % xc,
                            '%s.%s-JTH%s.xml' % (symbol, xcdict[xc], label))
    elif family.lower() == 'gbrv':
        xcdict = {'LDA': 'lda', 'PBE': 'pbe'}
        name = os.path.join(pp_path, 'GBRV-%s/%s_%s%s*' %
                            (xc, symbol.lower(), xc.lower(), label))
    elif family.lower() == 'dojo':
        xcdict = {'LDA': 'PW', 'PBE': 'PBE', 'PBEsol': 'PBEsol'}
        if label == '':
            label = 'standard'
        pp_dir = os.path.join(pp_path, 'ONCVPSP-%s-PD*' % (xcdict[xc]))
        fname = glob(os.path.join(pp_dir, label))
        if len(fname) > 0:

            print(fname)
            ppdict = {}
            with open(fname[0]) as myfile:
                lines = myfile.readlines()
            for line in lines:
                elem = line.strip().split('/')[0]
                name = os.path.join(pp_dir, line.strip())
                ppdict[elem] = name
            for i in range(57, 72):
                elem = chemical_symbols[i]
                ppdict[elem] = os.path.join(pp_dir,
                                            '%s/*f-in-core*.psp8' % elem)
        name = ppdict[symbol]
        print(name)

    pp = os.path.join(pp_path, name)
    names = glob(pp)

    if len(names) > 0 and os.path.isfile(names[0]):
        return names[0]
    else:
        raise ValueError("Can't find pseudopotential %s at %s." %
                         (name, pp_path))


def find_all_pp(symbols, xc, family, label_dict={}):
    ppdict = OrderedDict()
    for elem in symbols:
        if elem not in ppdict:
            if elem in label_dict:
                label = label_dict
            else:
                label = ''
            ppdict[elem] = find_pp(elem, xc, family, label)
    return list(ppdict.values())


def test_pp_finder():
    print(find_pp('La', 'LDA', 'jth'))
    print(find_pp('Lu', 'LDA', 'jth', label='_fincore'))
    print(find_pp('Sn', 'LDA', 'jth', label='_sp'))
    print(find_pp('Ga', 'LDA', 'gbrv'))
    print(find_pp('Lu', 'LDA', 'gbrv', label='_fincore'))
    #print(find_pp('Ca', 'LDA', 'dojo', label=''))
    #print(find_pp('Lu', 'PBEsol', 'dojo', label=''))
    #print(find_pp('Lu', 'PBEsol', 'dojo', label=''))

    print(find_all_pp(['Ba', 'Ti', 'O'], 'LDA', 'gbrv'))


#test_pp_finder()
