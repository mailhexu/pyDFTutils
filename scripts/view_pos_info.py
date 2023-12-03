#!/usr/bin/env  python
from ase.io import read
from ase.geometry.cell import cell_to_cellpar
import spglib.spglib as spglib
import sys

def view_cellpars(filename='POSCAR'):
    atoms=read(filename)
    print('CELLPARS: %s'%cell_to_cellpar(atoms.get_cell()))
    print('Volume: %s'%atoms.get_volume())

def view_spacegroup(filename='POSCAR',symprec=1e-4):
    atoms=read(filename)
    print("SPACEGROUP: %s"%spglib.get_spacegroup(atoms,symprec=symprec))
    #print(spglib.get_symmetry_dataset(atoms, symprec=symprec))

def viewall(filename='POSCAR',symprec=1e-4, angle_tolerance=-1.0):
    view_cellpars(filename=filename)
    view_spacegroup(filename=filename,symprec=symprec, angle_tolerance=angle_tolerance)

if __name__=='__main__':
    if len(sys.argv)==1:
        view_spacegroup(filename='POSCAR')
    elif len(sys.argv)==2:
        view_spacegroup(filename=sys.argv[1])
    elif len(sys.argv)==3:
        view_spacegroup(filename=sys.argv[1],symprec=float(sys.argv[2]), angle_tolerance=)
    else:
        print("Error")
