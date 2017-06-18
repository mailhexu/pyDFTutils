#!/usr/bin/env  python
from ase.io.vasp import read_vasp
from mypylib.myDFT.ase_utils import cell_to_cellpar
import pyspglib.spglib as spglib
import sys

def view_cellpars(filename='POSCAR'):
    atoms=read_vasp(filename)
    print('CELLPARS: %s'%cell_to_cellpar(atoms.get_cell()))
    print('Volume: %s'%atoms.get_volume())
def view_spacegroup(filename='POSCAR',symprec=1e-3):
    atoms=read_vasp(filename)
    print("SPACEGROUP: %s"%spglib.get_spacegroup(atoms,symprec=symprec))

def viewall(filename='POSCAR',symprec=1e-3):
    view_cellpars(filename=filename)
    view_spacegroup(filename=filename,symprec=symprec)

if __name__=='__main__':
    if len(sys.argv)==1:
        viewall(filename='POSCAR')
    elif len(sys.argv)==2:
        viewall(filename=sys.argv[1])
    elif len(sys.argv)==3:
        viewall(filename=sys.argv[1],symprec=float(sys.argv[2]))
    else:
        print("Error")
