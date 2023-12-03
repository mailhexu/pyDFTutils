#!/usr/bin/env python
# convert the format of the crystal structure file from one format ot another using ASE


import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description='Convert the format of the crystal structure file from one format ot another using ASE')
    parser.add_argument('-i', '--input', help='input file name', required=True)
    parser.add_argument('-o', '--output', help='output file name', required=True)
    parser.add_argument('-f', '--format', help='output file format', required=True)
    args = parser.parse_args()

    atoms = read(args.input)
    write(args.output, atoms, format=args.format)

if __name__ == '__main__':
    main()