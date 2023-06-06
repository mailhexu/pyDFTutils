#!/usr/bin/env python3
from pyDFTutils.siesta.mysiesta import read_siesta_xv
import sys

def xv2vasp():
    """
    Convert siesta XV file to vasp POSCAR file
    """
    with open(sys.argv[1]) as myfile:
        atoms=read_siesta_xv(myfile)
    if len(sys.argv) == 3:
        fout=sys.argv[2]
    else:
        fout='POSCAR'
    atoms.write(fout, format='vasp', vasp5=True)

if __name__ == '__main__':
    xv2vasp()

