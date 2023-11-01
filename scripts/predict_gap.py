#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from pyDFTutils.mlpot.matgl_wrapper import MatGLGapPredictor
import argparse

def get_gap(fname, xc="PBE"):
    atoms=read(fname)
    gap_predictor=MatGLGapPredictor()
    gap=gap_predictor.predict_gap(atoms, xc)
    return gap

if __name__=="__main__":
    p=argparse.ArgumentParser( description="Predict the band gap from a structure with MatGL pre-trained model")
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument("--xc", "-x", help="exchange-correlation functional. Could be PBE/SCAN/HSE/GLLB-SC. Default is PBE", default="PBE")
    args=p.parse_args()
    gap=get_gap(args.fname, args.xc)
    print(f"gap from {args.xc}: {gap:.3f}")