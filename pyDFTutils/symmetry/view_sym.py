#!/usr/bin/env  python
import numpy as np
from ase.io import read
from ase.geometry.cell import cell_to_cellpar
from ase import Atoms
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
    dataset: get full dataset
    """
    cell = atoms_to_spglib_cell(atoms, mag=False)
    spacegroup = spglib.get_spacegroup(
        cell, symprec=symprec, angle_tolerance=angle_tolerance
    )
    if dataset:
        symmetry_dataset = spglib.get_symmetry_dataset(
            cell, symprec=symprec, angle_tolerance=angle_tolerance
        )

    if printout:
        if not dataset:
            print(f"SPACEGROUP: {spacegroup}")
        else:
            print(symmetry_dataset)
    if not dataset:
        return spacegroup
    else:
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
    dataset: get full dataset
    """
    cell = atoms_to_spglib_cell(atoms, mag=True, magmom=magmom)
    symmetry_dataset = spglib.get_magnetic_symmetry_dataset(
        cell, symprec=symprec, angle_tolerance=angle_tolerance
    )
    uni_number = symmetry_dataset["uni_number"]
    magnetic_type = spglib.get_magnetic_spacegroup_type(uni_number)
    spacegroup = magnetic_type

    if printout:
        if not dataset:
            print(
                f"SPACEGROUP: {uni_number=}, {magnetic_type=}",
            )
        else:
            print(symmetry_dataset)
    return magnetic_type, symmetry_dataset


def viewall(filename="POSCAR", symprec=1e-4, angle_tolerance=-1.0):
    view_cellpars(filename=filename)
    view_spacegroup(filename=filename, symprec=symprec, angle_tolerance=angle_tolerance)



def find_sym(atoms, symprec=1e-4, angle_tolerance=-1.0):
    data = atoms_to_spglib_cell(atoms, mag=False) 
    return spglib.get_spacegroup(
        data, symprec=symprec, angle_tolerance=angle_tolerance
    )


def get_prim_atoms(atoms, symprec=1e-4, angle_tolerance=-1.0):
    return spglib.find_primitive(
        atoms_to_spglib_cell(atoms), symprec=symprec, angle_tolerance=angle_tolerance
    )


def ref_atoms_mag(atoms):
    """
    substitute atom with magnetic moment to another atom object. Use He Ne Ar Kr Xe Rn as subsititutions. So if you have these atoms , this fucntion can be rather buggy. Do *NOT* use it in that case.
    """
    symbols = atoms.get_chemical_symbols()
    magmoms = atoms.get_initial_magnetic_moments()
    sub_syms = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
    sym_dict = {}
    syms = []
    for sym, mag in zip(symbols, magmoms):
        if sym not in syms:
            syms.append(sym)
            sym_dict[(sym, mag)] = sym
        elif (sym, mag) not in sym_dict:
            sym_dict[(sym, mag)] = sub_syms.pop()
        else:
            pass
    new_sym = ""
    for sym, mag in zip(symbols, magmoms):
        new_sym += sym_dict[(sym, mag)]
    new_atoms = atoms.copy()
    new_atoms.set_chemical_symbols(new_sym)

    return new_atoms, sym_dict


def rev_ref_atoms(atoms, sym_dict):
    rev_dict = {}
    for key in sym_dict:
        rev_dict[sym_dict[key]] = key
    old_symbols = []
    old_magmons = []
    for sym in atoms.get_chemical_symbols():
        old_symbols.append(rev_dict[sym][0])
        old_magmons.append(rev_dict[sym][1])
    old_atoms = atoms.copy()
    old_atoms.set_chemical_symbols(old_symbols)
    old_atoms.set_initial_magnetic_moments(old_magmons)
    return old_atoms


def find_primitive(atoms, symprec=1e-4, angle_tolerance=-1.0, mag_symprec=1e-4, to_primitive=True):
    """
    find the primitive cell. a atoms object is returned.
    """
    # atoms_mag,sym_dict=ref_atoms_mag(atoms)
    #cell, scaled_pos, chem_nums = spglib.find_primitive(
    #    , symprec=symprec, angle_tolerance=angle_tolerance#, mag_symprec=mag_symprec
    #)

    cell, scaled_pos, chem_nums=spglib.standardize_cell(atoms_to_spglib_cell(atoms), symprec=symprec, angle_tolerance=angle_tolerance, to_primitive=False)
    new_atoms = Atoms(numbers=chem_nums, cell=cell, scaled_positions=scaled_pos)
    return new_atoms


def find_primitive_mag(atoms, symprec=1e-4, angle_tolerance=-1.0):
    """
    find the primitive cell withh regard to the magnetic structure. a atoms object is returned.
    """
    atoms_mag, sym_dict = ref_atoms_mag(atoms)
    cell, scaled_pos, chem_nums = spglib.find_primitive(atoms_mag, symprec=symprec)
    chem_sym = "H%d" % (len(chem_nums))
    new_atoms = Atoms(chem_sym)

    new_atoms.set_atomic_numbers(chem_nums)
    new_atoms.set_cell(cell)
    new_atoms.set_scaled_positions(scaled_pos)
    new_atoms = rev_ref_atoms(new_atoms, sym_dict)
    return new_atoms


def get_refined_atoms(atoms, symprec=1e-4, angle_tolerance=-1.0):
    """
    using spglib.refine_cell, while treat atoms with different magnetic moment as different element.
    """
    atoms_mag, sym_dict = ref_atoms_mag(atoms)
    cell, scaled_pos, chem_nums = spglib.refine_cell(
        atoms_mag, symprec=symprec, angle_tolerance=angle_tolerance
    )
    chem_sym = "H%d" % (len(chem_nums))
    new_atoms = Atoms(chem_sym)

    new_atoms.set_atomic_numbers(chem_nums)
    new_atoms.set_cell(cell)
    new_atoms.set_scaled_positions(scaled_pos)
    new_atoms = rev_ref_atoms(new_atoms, sym_dict)
    return new_atoms




def view_symmetry_cli():
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
    parser.add_argument("-d", "--dataset", action="store_true", help="view full dataset")

    args = parser.parse_args()

    atoms = read(args.filename)


    if args.mag:
        magmom = parse_magfile(args.magfile)
        view_magnetic_spacegroup(
            atoms=atoms,
            magmom=magmom,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            dataset=args.dataset,
        )
    else:
        view_spacegroup(
            atoms=atoms,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            dataset=args.dataset,
        )

if __name__ == "__main__":
    view_symmetry_cli()
