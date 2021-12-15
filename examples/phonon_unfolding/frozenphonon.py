#!/usr/bin/env python
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToTHz
from phonopy.file_IO import write_FORCE_CONSTANTS, write_disp_yaml
from phonopy.interface.vasp import write_supercells_with_displacements
from ase import Atoms
import numpy as np
import os


def calculate_phonon(atoms,
                     calc,
                     ndim=np.eye(3),
                     primitive_matrix=np.eye(3),
                     distance=0.01,
                     factor=VaspToTHz,
                     is_symmetry=True,
                     symprec=1e-5,
                     func=None,
                     **func_args):
    """
    """
    if 'magmoms' in atoms.arrays:
        is_mag = True
    else:
        is_mag = False
    # 1. get displacements and supercells
    atoms.set_calculator(calc)
    # bulk = PhonopyAtoms(atoms=atoms)
    if is_mag:
        bulk = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell(),
            magmoms=atoms.arrays['magmoms'], )
    else:
        bulk = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell())

    phonon = Phonopy(
        bulk,
        ndim,
        primitive_matrix=primitive_matrix,
        factor=factor,
        symprec=symprec)
    phonon.generate_displacements(distance=distance)
    disps = phonon.get_displacements()
    for d in disps:
        print("[phonopy] %d %s" % (d[0], d[1:]))
    supercell0 = phonon.get_supercell()
    supercells = phonon.get_supercells_with_displacements()
    #write_supercells_with_displacements(supercell0, supercells)
    write_disp_yaml(disps, supercell0)

    # 2. calculated forces.
    set_of_forces = []
    for iscell, scell in enumerate(supercells):
        cell = Atoms(
            symbols=scell.get_chemical_symbols(),
            scaled_positions=scell.get_scaled_positions(),
            cell=scell.get_cell(),
            pbc=True)
        if is_mag:
            cell.set_initial_magnetic_moments(
                atoms.get_initial_magnetic_moments())
        cell.set_calculator(calc)
        dir_name = "PHON_CELL%s" % iscell
        cur_dir = os.getcwd()
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        os.chdir(dir_name)
        forces = cell.get_forces()
        # print "[Phonopy] Forces: %s" % forces
        # Do something other than calculating the forces with func.
        # func: func(atoms, calc, func_args)
        if func is not None:
            func(cell, calc, **func_args)
        os.chdir(cur_dir)
        drift_force = forces.sum(axis=0)
        # print "[Phonopy] Drift force:", "%11.5f" * 3 % tuple(drift_force)
        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)

    # Phonopy post-process
    phonon.produce_force_constants(forces=set_of_forces)
    force_constants = phonon.get_force_constants()
    write_FORCE_CONSTANTS(force_constants, filename='FORCE_CONSTANTS')
    print('')
    print("[Phonopy] Phonon frequencies at Gamma:")
    for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
        print("[Phonopy] %3d: %10.5f THz" % (i + 1, freq))  # THz
        print("[Phonopy] %3d: %10.5f cm-1" % (i + 1, freq * 33.35))  # cm-1
    return phonon
