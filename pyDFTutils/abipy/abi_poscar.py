#/usr/bin/env python

from abipy.abilab import abiopen
from ase.io import write
import sys

def conv(fname):
    abifile=abiopen(fname)
    atoms=abifile.structure.to_ase_atoms()
    write('POSCAR', atoms, vasp5=True, sort=False)

if __name__=='__main__':
    fname=sys.argv[1]
    conv(fname)
