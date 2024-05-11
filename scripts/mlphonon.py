#!/usr/bin/env python
"""
A simple script to relax a structure with matgl.
"""
from __future__ import annotations
import numpy as np
from ase.io import read, write
from pyDFTutils.mlpot.matgl_wrapper import phonon_with_ml
import argparse

def main():
    p=argparse.ArgumentParser( description="Compute the phonon of a structure with machine learning potential.")
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument("--model", "-m", help="type of model: m3gnet|chgnet|matgl. Default is chgnet", default="chgnet")
    p.add_argument("--relax", "-r", help="relax the structure before computing the phonon.", action="store_true", default=False)
    p.add_argument("--ndim", "-n", help="number of repetitions of the structure in each direction.", nargs=3, type=int, default=[2,2,2])
    p.add_argument("--knames", "-k", help="special kpoints names for the band structure plot.", default=None)
    p.add_argument("--npoints", "-p", help="number of points in the band structure plot.", type=int, default=100)
    p.add_argument("--figname", "-f", help="name of the band structure plot.", default="phonon.pdf")
    args=p.parse_args()
    atoms=read(args.fname)
    atoms=phonon_with_ml(atoms, calc=args.model, relax=args.relax, ndim=np.diag(args.ndim), knames=args.knames, npoints=args.npoints, figname=args.figname)


if __name__=="__main__":
    main()
