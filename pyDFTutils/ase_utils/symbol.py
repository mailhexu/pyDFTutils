import re
from collections import OrderedDict
from ase.io import read

def symbol_number(symbols):
    """
    symbols can be also atoms. Thus the chemical symbols will be used.
    Fe Fe Fe O -> {Fe1:0 Fe2:1 Fe3:2 O1:3}
    """
    try:
        symbs=symbols.copy().get_chemical_symbols()
    except Exception:
        symbs=symbols
    symdict={}
    result=OrderedDict()
    for i,sym in enumerate(symbs):
        if sym in symdict:
            symdict[sym]=symdict[sym]+1
        else:
            symdict[sym]=1
        result[sym+str(symdict[sym])]=i
    return result


def symnum_to_sym(symbol_number):
    """
    symnum-> sym. eg: Fe1-> Fe
    """
    try:
        a=re.search('[A-Za-z]+',symbol_number).group()
        return a
    except AttributeError:
        raise AttributeError('%s is not a good symbol_number'%symbol_number)

def get_symdict(filename='POSCAR',atoms=None):
    """
    get a symbol_number: index dict.
    """
    if atoms is not None:
        syms=atoms.get_chemical_symbols()
    elif filename is not None and atoms is None:
        syms=read(filename).get_chemical_symbols()

    symdict=symbol_number(syms)
    return symdict

