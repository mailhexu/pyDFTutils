#!/usr/bin/env  python
from ase.io import read
from ase.geometry.cell import cell_to_cellpar
import spglib.spglib as spglib
import sys
import argparse


def view_cellpars(filename="POSCAR", magnetic_moments=None):
    atoms = read(filename)
    print("CELLPARS: %s" % cell_to_cellpar(atoms.get_cell()))
    print("Volume: %s" % atoms.get_volume())


def parse_magfile(fname):
    mag = np.loadtxt(fname)
    return mag


def atoms_to_spglib_cell(atoms, mag=False, magmom=None):
    cell = atoms.get_cell()
    xred = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    if mag:
        if magmom is None:
            magmom = atoms.get_initial_magnetic_moments()
        return (cell, xred, numbers, magmom)
    else:
        return (cell, xred, numbers)


def is_mag(atoms):
    m = atoms.get_initial_magnetic_moments()
    return np.linalg.norm(m) > 0


def is_collinear_mag(atoms):
    m = atoms.get_initial_magnetic_moments()
    return len(m.shape) == 1


def view_spacegroup(
    atoms=None,
    mag=False,
    symprec=1e-4,
    angle_tolerance=-1.0,
    dataset=False,
    printout=True,
):
    """
    params:
    atoms: ASE atoms object
    symprec: symprec
    angle_tolerance: angle_tolerance
    view_all: view all symmetry operations
    """
    cell = atoms_to_spglib_cell(atoms, mag=mag, magmom=magmom)
    spacegroup = spglib.get_spacegroup(
        cell, symprec=symprec, angle_tolerance=angle_tolerance
    )
    if dataset:
        symmetry_dataset = spglib.get_symmetry_dataset(
            cell, symprec=symprec, angle_tolerance=angle_tolerance
        )

    if printout:
        if not view_all:
            print(f"SPACEGROUP: {spacegroup}")
        else:
            print(symmetry_dataset)
    return spacegroup, symmetry_dataset


def view_magnetic_spacegroup(
    atoms=None,
    magmom=None,
    symprec=1e-4,
    angle_tolerance=-1.0,
    dataset=False,
    printout=True,
):
    """
    params:
    atoms: ASE atoms object
    magmom: magnetic moments
    symprec: symprec
    angle_tolerance: angle_tolerance
    view_all: view all symmetry operations
    """
    cell = atoms_to_spglib_cell(atoms, mag=mag, magmom=magmom)
    symmetry_dataset = spglib.get_magnetic_symmetry_dataset(
        cell, symprec=symprec, angle_tolerance=angle_tolerance
    )
    uni_number = symmetry_dataset["uni_number"]
    magnetic_type = spglib.get_magnetic_spacegroup_type(uni_number)

    if printout:
        if not view_all:
            print(
                f"SPACEGROUP: {uni_number=}, {magnetic_type=}",
            )
        else:
            print(symmetry_dataset)
    return uni_number, symmetry_dataset


def viewall(filename="POSCAR", symprec=1e-4, angle_tolerance=-1.0):
    view_cellpars(filename=filename)
    view_spacegroup(filename=filename, symprec=symprec, angle_tolerance=angle_tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View spacegroup and cellpars of atomic structure file"
    )
    parser.add_argument(
        "filename",
        type=str,
        default="POSCAR",
        help="atomic structure file. Default: POSCAR",
    )
    parser.add_argument(
        "-s", "--symprec", type=float, default=1e-4, help="symprec, default: 1e-4"
    )
    parser.add_argument(
        "-m", "--mag", action="store_true", help="use magnetic symmetry"
    )
    parser.add_argument(
        "-M", "--magfile", type=str, default=None, help="magnetic moments file"
    )
    parser.add_argument(
        "-a",
        "--angle_tolerance",
        type=float,
        default=-1.0,
        help="angle_tolerance, default: -1.0",
    )
    parser.add_argument("-v", "--view_all", action="store_true", help="view all")

    args = parser.parse_args()
    if args.mag:
        magmom = parse_magfile(args.magfile)
        view_magnetic_spacegroup(
            filename=args.filename,
            magmom=magmom,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            view_all=args.view_all,
        )
    else:
        view_spacegroup(
            filename=args.filename,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            view_all=args.view_all,
        )
