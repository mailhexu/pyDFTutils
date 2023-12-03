#!/usr/bin/env python
"""
Build a supercell from primitive cell using ase
"""
import argparse
import numpy as np
from ase.io import read, write
from ase.build import make_supercell



def main():
    """
    Build a supercell from primitive cell using ase
    """
    parser = argparse.ArgumentParser(description='Build a supercell from primitive cell using ase')
    parser.add_argument('-i', '--input', type=str, default='POSCAR',
                        help='input file name, default is POSCAR')
    parser.add_argument('-o', '--output', type=str, default='supercell.vasp',
                        help='output file name, default is supercell.vasp')
    # supercell can be 3-integers or 9 integers
    # 3 integers: repeat the cell along a, b, c
    # 9 integers: the 3 integers are the number of repetitions of the cell vectors
    #             the other 6 integers are the 3x3 matrix of the supercell
    #             in terms of the primitive cell vectors
    parser.add_argument('-s', '--supercell', nargs='+', type=int, default=[1, 1, 1],
                        help='supercell size, can be three integers or nine integers (represent 3x3 matrix). default is [1, 1, 1] ')

    args = parser.parse_args()

    # read input file
    atoms = read(args.input)

    # build supercell
    scmat=args.supercell
    if len(scmat)==3:
        scmat=np.diag(scmat)
    elif len(scmat)==9:
        scmat=np.resize(scmat, (3,3))
    else:
        raise ValueError("supercell should be 3 integers or 9 integers.")
    supercell = make_supercell(atoms, scmat)

    # write output file
    write(args.output, supercell, vasp5=True, direct=True, sort=True)

if __name__=="__main__":
    main()
