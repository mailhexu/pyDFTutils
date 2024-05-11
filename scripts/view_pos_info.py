#!/usr/bin/env  python
from ase.io import read
from ase.geometry.cell import cell_to_cellpar
import spglib.spglib as spglib
import sys
import argparse


def view_cellpars(filename="POSCAR"):
    atoms = read(filename)
    print("CELLPARS: %s" % cell_to_cellpar(atoms.get_cell()))
    print("Volume: %s" % atoms.get_volume())


def view_spacegroup(
    filename="POSCAR", symprec=1e-4, angle_tolerance=-1.0, view_all=False
):
    atoms = read(filename)

    cell = (cell_vectors, positions, numbers) = (
        atoms.get_cell(),
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
    )
    if not view_all:
        print(
            "SPACEGROUP: %s"
            % spglib.get_spacegroup(
                cell, symprec=symprec, angle_tolerance=angle_tolerance
            )
        )
    else:
        print(
            spglib.get_symmetry_dataset(
                cell, symprec=symprec, angle_tolerance=angle_tolerance
            )
        )


def viewall(filename="POSCAR", symprec=1e-4, angle_tolerance=-1.0):
    view_cellpars(filename=filename)
    view_spacegroup(filename=filename, symprec=symprec, angle_tolerance=angle_tolerance)


if __name__ == "__main__":
    # if len(sys.argv)==1:
    #    view_spacegroup(filename='POSCAR')
    # elif len(sys.argv)==2:
    #    view_spacegroup(filename=sys.argv[1])
    # elif len(sys.argv)==3:
    #    view_spacegroup(filename=sys.argv[1],symprec=float(sys.argv[2]), angle_tolerance=)
    # else:
    #    print("Error")
    parser = argparse.ArgumentParser(
        description="View spacegroup and cellpars of atomic structure file"
    )
    parser.add_argument(
        "filename", type=str, default="POSCAR", help="atomic structure file. Default: POSCAR"
    )
    parser.add_argument("-s", "--symprec", type=float, default=1e-4, help="symprec, default: 1e-4")
    parser.add_argument(
        "-a", "--angle_tolerance", type=float, default=-1.0, help="angle_tolerance, default: -1.0"
    )
    parser.add_argument("-v", "--view_all", action="store_true", help="view all")

    args = parser.parse_args()
    view_spacegroup(
        filename=args.filename,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        view_all=args.view_all,
    )
