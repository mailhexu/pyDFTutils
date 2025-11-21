#!/usr/bin/env python
"""
Octahedra Information Extraction Script

This script extracts and analyzes octahedral structures from crystallographic data files
(typically VASP POSCAR format). It identifies octahedra by finding central atoms (e.g.,
transition metals like Mn, Fe, Co) and their surrounding vertex atoms (e.g., oxygen atoms)
within a specified distance threshold.

The output is a CSV file containing detailed information about each identified octahedron,
which is useful for analyzing distortions, rotations, and other structural properties of
perovskite materials and similar structures with octahedral building units.

Example usage:
    python octainfo.py -f POSCAR -o octahedra_data.csv -c Mn Fe -v O -d 2.5
"""
import argparse
from ase.io import read, write
from pyDFTutils.perovskite.octahedra import write_all_octahedra_info


def main():
    """
    Main function to extract octahedra information from a crystal structure file.

    This function parses command-line arguments, reads the input crystal structure,
    identifies octahedra based on the specified center and vertex atoms within the
    given distance threshold, and writes the octahedra information to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description='Extract and write information about octahedra structures from crystal structures.\n\n'
                    'This script analyzes perovskite materials by identifying octahedral units (e.g., BO6 octahedra '
                    'where B is a metal cation surrounded by 6 oxygen anions). It reads a crystal structure file '
                    '(typically POSCAR format), identifies octahedra based on specified center and vertex atoms, '
                    'and writes detailed information about these octahedra to a CSV output file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f', '--filename', type=str,
                        help='Input crystal structure filename in POSCAR/VASP format. Default: POSCAR',
                        default='POSCAR')
    parser.add_argument('-o', '--output',
                        help='Output filename for the octahedra information in CSV format. Default: octa_info.csv',
                        default='octa_info.csv')
    parser.add_argument('-c', '--center', nargs='+', type=str,
                        help='Element symbol(s) for octahedra center atoms (e.g., Mn Fe Co). '
                             'Multiple elements can be specified. Default: Mn',
                        default=['Mn'])
    parser.add_argument('-v', '--vertex',
                        help='Element symbol for octahedra vertex atoms (e.g., O for oxygen). Default: O',
                        default='O')
    parser.add_argument('-d', '--distance', type=float,
                        help='Maximum distance threshold to identify vertex atoms around center atoms. '
                             'Atoms within this distance will be considered part of the octahedra. Default: 3.0 Å',
                        default=3.0)
    parser.add_argument('-a', '--axis_type',
                        help='Coordinate axis transformation type. Options: None (standard Cartesian), '
                             'rotate45_xy (rotates x and y axes by 45 degrees). Default: None',
                        default=None)
    parser.add_argument('-x', '--xvec', help='Custom x-axis vector as space-separated values (e.g., "1 0 0"). '
                                             'Used for defining custom coordinate system. Default: (1,0,0)',
                                             nargs='+', type=float, default=(1,0,0))
    parser.add_argument('-y', '--yvec', help='Custom y-axis vector as space-separated values (e.g., "0 1 0"). '
                                             'Used for defining custom coordinate system. Default: (0,1,0)',
                                             nargs='+', type=float, default=(0,1,0))
    parser.add_argument('-z', '--zvec', help='Custom z-axis vector as space-separated values (e.g., "0 0 1"). '
                                             'Used for defining custom coordinate system. Default: (0,0,1)',
                                             nargs='+', type=float, default=(0,0,1))
    args = parser.parse_args()
    atoms = read(args.filename)

    # Extract and write octahedra information
    # Parameters:
    # - atoms: ASE Atoms object containing the crystal structure
    # - tuple(args.center): Tuple of center atom element symbols
    # - args.vertex: Vertex atom element symbol
    # - args.distance: Maximum distance threshold for identifying vertex atoms
    # - output: Output CSV filename
    # - axis_type: Coordinate axis transformation type
    # - x, y, z: Custom coordinate axis vectors
    write_all_octahedra_info(
        atoms,
        tuple(args.center),
        args.vertex,
        args.distance,
        output=args.output,
        axis_type=args.axis_type,
        x=args.xvec,
        y=args.yvec,
        z=args.zvec
    )


if __name__ == '__main__':
    main()
