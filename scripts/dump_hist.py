#!/usr/bin/env python

from pyDFTutils.abinit.abihist_file import AbihistFile
import argparse

def main():
    parser = argparse.ArgumentParser(description='Dump the hist file to other formats of files.')
    parser.add_argument('histfile', help='Path to the hist file')
    parser.add_argument('-p', '--prefix', help='prefix to the output file', type=str )
    parser.add_argument('-f', '--fmt', help='format of the output file', default='vasp', type=str)
    parser.add_argument('-i', '--index', help='index of the history to dump, default is None, which means dump all the hist files. To dump the last history, use -1.', default=None, type=int)
    args = parser.parse_args()


    hist=AbihistFile(args.histfile)
    hist.read()
    hist.dump_atoms(prefix=args.prefix, fmt=args.fmt, i=args.index)


if __name__ == '__main__':
    main()

