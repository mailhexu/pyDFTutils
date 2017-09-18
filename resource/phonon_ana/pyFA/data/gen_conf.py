#!/usr/bin/env python
from ase.atoms import Atoms
from pyFA.abinit import abinit_calculator
def gen_atoms(fname):
    with open(fname) as myfile:
        lines=myfile.readlines()

    symbols= ''.join([l.strip() for l in lines])
    atoms=Atoms(symbols=symbols)
    return atoms

calc=abinit_calculator(xc='PBEsol')
atoms=gen_atoms('./PBEsol.txt')
calc.scf_calculation(atoms)
