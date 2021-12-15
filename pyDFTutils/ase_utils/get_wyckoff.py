from ase.io import read
import spglib


def get_wyckoff(atoms):
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    magmoms = None
    cell = (lattice, positions, numbers, magmoms)
    ds = spglib.get_symmetry_dataset(atoms, symprec=1e-3, hall_number=0)
    print(ds)


def test():
    atoms = read('POSCAR')
    get_wyckoff(atoms)


test()
