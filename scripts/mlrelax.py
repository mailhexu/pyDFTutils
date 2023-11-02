#!/usr/bin/env python
"""
A simple script to relax a structure with matgl.
"""
from __future__ import annotations
import warnings
from ase.io import read, write
from pyDFTutils.mlpot.matgl_wrapper import relax_with_ml
import argparse

def main():
    p=argparse.ArgumentParser( description="Relax a structure with MatGL")
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument("--model", "-m", help="type of model: m3gnet|chgnet|matgl. Default is chgnet", default="chgnet")
    p.add_argument("--sym", "-s", help="whether to impose symmetry constraints on the atoms. Default is False", default=False, action="store_true")
    p.add_argument("--relax_cell", "-r", help="whether to relax the cell shape as well. Default is False", default=False, action="store_true")
    p.add_argument("--fmax", "-f", help="The maximum force allowed on each atom. Default is 0.001", default=0.001, type=float)
    p.add_argument("--cell_factor", "-c", help="The factor by which to scale the unit cell when relaxing the cell shape. Default is 1000", default=1000, type=float)
    p.add_argument("--output_file", "-o", help="The name of the file to write the relaxed structure to. Default is POSCAR_relax.vasp", default="POSCAR_relax.vasp")
    args=p.parse_args()
    atoms=read(args.fname)
    atoms=relax_with_ml(atoms, calc=args.model, sym=args.sym, relax_cell=args.relax_cell, fmax=args.fmax, cell_factor=args.cell_factor)
    write(args.output_file, atoms)

if __name__=="__main__":
    main()
