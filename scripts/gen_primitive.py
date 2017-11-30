#!/usr/bin/env python
from ase.io import read
from ase.io import write
from pyDFTutils.ase_utils.geometry import find_primitive
import sys
def gen(src_file ,des_file):
    atoms=read(src_file)
    new_atoms=find_primitive(atoms, symprec=1e-4)
    write(des_file, new_atoms)
    return atoms

if __name__=='__main__':
    gen(sys.argv[1],sys.argv[2])
