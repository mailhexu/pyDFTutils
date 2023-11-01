#!/usr/bin/env python3
try:
    from ase.io.abinit import read_abinit_in
    from ase.io.cif import write_cif
    from ase.io import write
except:
    print("Ase is not installed. Please install ASE, e.g. with 'python3 -m pip install ase --user' ")

from argparse import ArgumentParser



def convert_to_cif(infile, outfile):
    with open(infile) as myfile:
        atoms = read_abinit_in(myfile)
    write_cif(outfile, atoms)

def main():
    parser = ArgumentParser(description="Converts a Abinit output file to a CIF file. ")
    parser.add_argument("infile", help="Input Abinit output file")
    parser.add_argument("outfile", help="Output CIF file")
    args = parser.parse_args()
    convert_to_cif(args.infile, args.outfile)

if __name__ == "__main__":
    main()

