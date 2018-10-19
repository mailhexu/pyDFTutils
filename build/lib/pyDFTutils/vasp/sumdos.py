#!/usr/bin/env python
from pyDFTutils.vasp.vasp_dos import write_all_sum_dos
from argparse import ArgumentParser
if __name__=='__main__':
    parser=ArgumentParser(description='sum the DOSCAR valence dos')
    parser.add_argument('-o','--output',help='output file name',default='sum_dos.txt')
    args=parser.parse_args()
    write_all_sum_dos(output=args.output)
