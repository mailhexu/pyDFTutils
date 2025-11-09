#!/usr/bin/env python
import copy
from ase import Atoms
from ase.io import read, write
import numpy as np
import os
import pickle
from phonopy import Phonopy, load
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToTHz
from phonopy.file_IO import (write_FORCE_CONSTANTS, write_disp_yaml,
                             parse_FORCE_SETS)
from phonopy.interface.vasp import write_supercells_with_displacements
import copy
#from concurrent.futures import ProcessPoolExecutor
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool


def _check_magnetic(atoms):
    """Check if atoms have magnetic moments."""
    return 'magmoms' in atoms.arrays or 'initial_magmoms' in atoms.arrays


def _create_phonopy_atoms(atoms):
    """Convert ASE atoms to PhonopyAtoms."""
    is_mag = _check_magnetic(atoms)
    
    if is_mag:
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell(),
            magmoms=atoms.arrays['initial_magmoms']
        ), is_mag
    else:
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell()
        ), is_mag


def _setup_phonon(atoms, ndim, primitive_matrix, factor, symprec, distance, is_plusminus):
    """Initialize phonopy object and generate displacements."""
    bulk, is_mag = _create_phonopy_atoms(atoms)
    
    phonon = Phonopy(
        bulk,
        ndim,
        primitive_matrix=primitive_matrix,
        factor=factor,
        symprec=symprec
    )
    phonon.generate_displacements(distance=distance, is_plusminus=is_plusminus)
    
    # Print displacement information
    disps = phonon.displacements
    for d in disps:
        print(("[phonopy] %d %s" % (d[0], d[1:])))
    
    supercell0 = phonon.supercell
    supercells = phonon.supercells_with_displacements
    
    if supercells is None:
        raise ValueError("supercell not got")
    
    write_disp_yaml(disps, supercell0)
    
    return phonon, supercell0, supercells, is_mag


def _load_forces_from_file(forces_set_file):
    """Load forces from FORCE_SETS file."""
    set_of_forces = parse_FORCE_SETS(
        is_translational_invariance=False, 
        filename=forces_set_file
    )
    return set_of_forces


def _load_forces_from_pickle(phonon, restart):
    """
    Load forces from pickle file if it exists.
    
    Returns:
        tuple: (set_of_forces, iskip, is_complete)
            - set_of_forces: list of force arrays
            - iskip: number of force sets to skip
            - is_complete: True if all forces are loaded from pickle
    """
    if not os.path.exists('forces_set.pickle'):
        return [], 0, False
    
    print("[Phonopy] Reading forces from existing 'forces_set.pickle' file")
    with open("forces_set.pickle", 'rb') as myfile:
        set_of_forces = pickle.load(myfile)
    print(f"[Phonopy] Loaded {len(set_of_forces)} force sets from pickle file")
    
    n_expected = len(phonon.supercells_with_displacements)
    
    # Check if we have complete force sets
    if len(set_of_forces) > 0 and len(set_of_forces) == n_expected:
        print("[Phonopy] Complete force sets found, using pickle data")
        return set_of_forces, len(set_of_forces), True
    else:
        print(f"[Phonopy] Incomplete force sets ({len(set_of_forces)}/{n_expected}), will calculate remaining")
        
        if restart:
            iskip = len(set_of_forces)
            print(f"[Phonopy] Restart mode: found {iskip} existing force sets")
            return set_of_forces, iskip, False
        else:
            return [], 0, False


def _prepare_initial_wavecar(supercell0, calc, is_mag, sc_mag):
    """Prepare initial WAVECAR for subsequent calculations."""
    scell = supercell0
    cell = Atoms(
        symbols=scell.symbols,
        scaled_positions=scell.scaled_positions,
        cell=scell.cell,
        pbc=True
    )
    
    if is_mag and sc_mag is not None:
        cell.set_initial_magnetic_moments(sc_mag)
    
    write('Supercell.cif', cell)
    mcalc = copy.deepcopy(calc)
    cell.calc = mcalc
    
    dir_name = "SUPERCELL0"
    cur_dir = os.getcwd()
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    os.chdir(dir_name)
    mcalc.scf_calculation(cell)
    os.chdir(cur_dir)


def _create_force_calculator(supercells, calc, is_mag, sc_mag, mask_force, 
                             prepare_initial_wavecar, func, func_args):
    """
    Create a function to calculate forces for a single supercell.
    
    Returns a closure that captures the necessary variables.
    """
    def calc_force(iscell):
        scell = supercells[iscell]
        cell = Atoms(
            symbols=scell.symbols,
            scaled_positions=scell.scaled_positions,
            cell=scell.cell,
            pbc=True
        )
        
        if is_mag:
            cell.set_initial_magnetic_moments(sc_mag)
        
        cell.calc = copy.deepcopy(calc)
        
        dir_name = "PHON_CELL%s" % iscell
        cur_dir = os.getcwd()
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        if prepare_initial_wavecar:
            os.system('ln -s %s %s' %
                     (os.path.abspath("SUPERCELL0/WAVECAR"),
                      os.path.join(dir_name, 'WAVECAR')))
            os.system('cp %s %s' %
                     (os.path.abspath("SUPERCELL0/siesta.DM"),
                      os.path.join(dir_name, 'siesta.DM')))
        
        os.chdir(dir_name)
        forces = cell.get_forces()
        
        if mask_force is not None:
            forces = forces * np.array(mask_force)[None, :]
        
        print("[Phonopy] Forces: %s" % forces)
        
        # Run custom function if provided
        if func is not None:
            func(cell, calc, **func_args)
        
        os.chdir(cur_dir)
        
        # Apply drift correction
        drift_force = forces.sum(axis=0)
        print("[Phonopy] Drift force:", "%11.5f" * 3 % tuple(drift_force))
        
        for force in forces:
            force -= drift_force / forces.shape[0]
        
        return forces
    
    return calc_force


def _calculate_forces(supercells, calc, is_mag, sc_mag, mask_force,
                     prepare_initial_wavecar, func, func_args, 
                     iskip, set_of_forces, parallel):
    """Calculate forces for displaced supercells."""
    calc_force = _create_force_calculator(
        supercells, calc, is_mag, sc_mag, mask_force,
        prepare_initial_wavecar, func, func_args
    )
    
    if parallel:
        p = Pool()
        new_forces = p.map(calc_force, list(range(iskip, len(supercells))))
        set_of_forces.extend(new_forces)
    else:
        for iscell in range(iskip, len(supercells)):
            print(f"[Phonopy] Calculating forces for supercell {iscell+1}/{len(supercells)}")
            fs = calc_force(iscell)
            set_of_forces.append(fs)
            # Save progress after each calculation
            with open("forces_set.pickle", 'wb') as myfile:
                pickle.dump(set_of_forces, myfile)
    
    return set_of_forces


def _save_results(phonon):
    """Save phonon results to files."""
    force_constants = phonon.force_constants
    write_FORCE_CONSTANTS(force_constants, filename='FORCE_CONSTANTS')
    
    with open('phonon.pickle', 'wb') as myfile:
        pickle.dump(phonon, myfile)
    
    phonon.save(settings={'force_constants': True})


def calculate_phonon(atoms,
                     calc=None,
                     forces_set_file=None,
                     ndim=np.eye(3),
                     primitive_matrix=np.eye(3),
                     distance=0.01,
                     factor=VaspToTHz,
                     is_plusminus='auto',
                     is_symmetry=True,
                     symprec=1e-5,
                     func=None,
                     prepare_initial_wavecar=False,
                     mask_force=[1,1,1],
                     skip=None,
                     restart=True,
                     parallel=True,
                     sc_mag=None,
                     **func_args):
    """
    Calculate phonon properties using frozen phonon method.
    
    Args:
        atoms: ASE Atoms object
        calc: ASE calculator object
        forces_set_file: Path to existing FORCE_SETS file
        ndim: Supercell dimensions (3x3 matrix)
        primitive_matrix: Primitive cell matrix
        distance: Displacement distance for finite differences
        factor: Unit conversion factor (default: VaspToTHz)
        is_plusminus: Generate plus/minus displacements ('auto', True, False)
        is_symmetry: Use symmetry to reduce calculations
        symprec: Symmetry precision
        func: Custom function to run for each displacement
        prepare_initial_wavecar: Prepare initial wavefunction
        mask_force: Mask for force components [x, y, z]
        skip: Skip specific calculations
        restart: Enable restart from pickle file
        parallel: Use parallel force calculation
        sc_mag: Magnetic moments for supercell
        **func_args: Additional arguments for custom function
        
    Returns:
        Phonopy object with calculated force constants
    """
    is_mag = _check_magnetic(atoms)
    print("is_mag: ", is_mag)
    
    if calc is not None:
        atoms.calc = calc
    
    # Setup phonon object and generate displacements
    phonon, supercell0, supercells, is_mag = _setup_phonon(
        atoms, ndim, primitive_matrix, factor, symprec, distance, is_plusminus
    )
    
    # Handle force calculation or loading
    if forces_set_file is not None:
        # Load from FORCE_SETS file
        set_of_forces = _load_forces_from_file(forces_set_file)
        phonon.set_displacement_dataset(set_of_forces)
        phonon.produce_force_constants()
    else:
        # Try loading from pickle, or calculate forces
        set_of_forces, iskip, is_complete = _load_forces_from_pickle(phonon, restart)
        
        if is_complete:
            # All forces loaded from pickle
            phonon.produce_force_constants(forces=np.array(set_of_forces))
            _save_results(phonon)
            return phonon
        
        # Need to calculate forces (some or all)
        if prepare_initial_wavecar and skip is None:
            _prepare_initial_wavecar(supercell0, calc, is_mag, sc_mag)
        
        set_of_forces = _calculate_forces(
            supercells, calc, is_mag, sc_mag, mask_force,
            prepare_initial_wavecar, func, func_args,
            iskip, set_of_forces, parallel
        )
        
        phonon.produce_force_constants(forces=np.array(set_of_forces))
    
    _save_results(phonon)
    return phonon


def compute_phonon_at_qpoints(qpoints, phonopy_yaml='phonopy_params.yaml'):
    """
    Compute phonon eigenvalues and eigenvectors at specified q-points.
    
    This function loads a phonopy object from a saved phonopy_params.yaml file
    (produced by phonon.save(settings={'force_constants': True})) and computes
    the dynamical matrix eigenvalues (frequencies) and eigenvectors at the
    requested q-points.
    
    Args:
        qpoints: List or array of q-points in fractional coordinates.
                 Shape: (nqpoints, 3) or a single q-point (3,)
        phonopy_yaml: Path to phonopy_params.yaml file (default: 'phonopy_params.yaml')
        
    Returns:
        dict: Dictionary containing:
            - 'qpoints': Array of q-points (nqpoints, 3)
            - 'frequencies': Phonon frequencies in THz, shape (nqpoints, nbands)
            - 'eigenvectors': Phonon eigenvectors, shape (nqpoints, nbands, natoms, 3)
                            Complex array with eigenvector components
            - 'nbands': Number of phonon bands (3 * natoms in primitive cell)
            
    Example:
        >>> # Single q-point (Gamma point)
        >>> result = compute_phonon_at_qpoints([0, 0, 0])
        >>> print(result['frequencies'][0])  # Frequencies at Gamma
        >>> print(result['eigenvectors'][0])  # Eigenvectors at Gamma
        
        >>> # Multiple q-points
        >>> qpts = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]
        >>> result = compute_phonon_at_qpoints(qpts)
        >>> print(result['frequencies'].shape)  # (3, nbands)
    """
    # Ensure qpoints is a 2D array
    qpoints = np.atleast_2d(qpoints)
    nqpoints = qpoints.shape[0]
    
    # Load phonopy object from yaml file using phonopy.load()
    print(f"[Phonopy] Loading phonopy data from {phonopy_yaml}")
    phonon = load(phonopy_yaml=phonopy_yaml)
    
    if phonon is None:
        raise ValueError(f"Failed to load Phonopy object from {phonopy_yaml}")
    
    # Check if force constants are available
    if phonon.force_constants is None:
        raise ValueError(
            "Force constants are not available in the loaded phonopy object. "
            "Make sure the phonopy_params.yaml was created with "
            "phonon.save(settings={'force_constants': True})"
        )
    
    # Get number of bands
    primitive = phonon.primitive
    natoms = len(primitive)
    nbands = 3 * natoms
    
    print(f"[Phonopy] Computing phonon at {nqpoints} q-point(s)")
    print(f"[Phonopy] Primitive cell has {natoms} atoms, {nbands} phonon bands")
    
    # Initialize output arrays
    frequencies = np.zeros((nqpoints, nbands), dtype=float)
    # Eigenvectors from phonopy have shape (bands, bands) = (bands, natoms*3)
    # We'll reshape to (bands, natoms, 3) for convenience
    eigenvectors = np.zeros((nqpoints, nbands, natoms, 3), dtype=complex)
    
    # Compute for each q-point
    for iq, qpt in enumerate(qpoints):
        # Set the q-point and compute
        phonon.run_qpoints([qpt], with_eigenvectors=True)
        
        # Get frequencies and eigenvectors
        qpt_dict = phonon.get_qpoints_dict()
        qpt_freqs = qpt_dict['frequencies']
        qpt_eigvecs = qpt_dict['eigenvectors']
        
        frequencies[iq] = qpt_freqs[0]
        
        # Reshape eigenvectors from (bands, bands) to (bands, natoms, 3)
        # eigvecs shape: (bands, natoms*3)
        eigenvectors[iq] = qpt_eigvecs[0].reshape(nbands, natoms, 3)
        
        print(f"[Phonopy] Q-point {iq+1}/{nqpoints}: {qpt} - "
              f"Frequencies: {qpt_freqs[0][0]:.4f} to {qpt_freqs[0][-1]:.4f} THz")
    
    return {
        'qpoints': qpoints,
        'frequencies': frequencies,
        'eigenvectors': eigenvectors,
        'nbands': nbands
    }
