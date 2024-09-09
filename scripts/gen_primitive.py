#!/usr/bin/env python
from ase.io import read
from ase.io import write
from pyDFTutils.symmetry.view_sym import find_primitive
import sys
import argparse

def gen(src_file ,des_file, symprec=1e-4, angle_tolerance=-1.0, to_primitive=True):
    atoms=read(src_file)
    new_atoms=find_primitive(atoms, symprec=symprec, angle_tolerance=angle_tolerance , to_primitive=to_primitive)
    write(des_file, new_atoms)
    return atoms

if __name__=='__main__':
    #gen(sys.argv[1],sys.argv[2])
    parser = argparse.ArgumentParser(description='generate primitive cell')
    parser.add_argument('src_file', help='source file')
    parser.add_argument('des_file', help='destination file')
    parser.add_argument('-s', '--symprec', type=float, default=1e-4, help='symmetry precision')
    parser.add_argument('-a', '--angle_tolerance', type=float, default=-1.0, help='angle tolerance')
    parser.add_argument('-n', '--nop', action='store_false', help='not to primitive, just standardize')
    args = parser.parse_args()
    gen(args.src_file, args.des_file, args.symprec, args.angle_tolerance, to_primitive=(not args.nop))
    

