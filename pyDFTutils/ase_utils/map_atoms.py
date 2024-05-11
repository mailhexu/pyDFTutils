import numpy as np
from ase.io import read, write
from ase import Atoms

def distance_pbc(x1, x2, period=1.0):
    """
    distance of two points in periodic boundary conditions
    """
    dist = x1 - x2
    dist = dist - np.round(dist / period) * period
    return np.linalg.norm(dist)



def mapping_atoms(atoms, ref, tol=1e-5, pbc=False):
    """
    Given a atoms object and a reference atoms object, return a mapping between the two.
    It is based on the nearest distance between the two atoms.
    Parmeters:
    atoms: ase.Atoms object
      The atoms object to be mapped to the reference atoms object.
    ref: ase.Atoms object
        The reference atoms object.
    Returns:
    mapping: list
        A list of indices of the atoms object that correspond to the reference atoms object.
    shift: numpy.ndarray
        The shift vector between the two atoms objects.
    """
    # Check if the two atoms objects have the same length
    if len(atoms) != len(ref):
        raise ValueError("The two atoms objects must have the same length.")

    # Calculate the shift vector
    shift = atoms.get_center_of_mass() - ref.get_center_of_mass()
    # Shift the ref object
    ref.translate(shift)
    # Initialize the mapping list
    mapping = []
    xred = atoms.get_scaled_positions()
    ref_xred= ref.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    ref_symbols = ref.get_chemical_symbols()
    # Loop over the atoms in the reference atoms object
    for i, p in enumerate(ref_xred):
        # Check the nearest distance between the two atoms
        dvec = p[None, :]-xred
        if pbc:
            dvec = dvec - np.round(dvec)
        dist = np.linalg.norm(dvec, axis=1)
        #dist = np.linalg.norm(p - ref_xred, axis=1)
        # find the index of the nearest atom with the same symbol
        # filter out the atoms with different symbols
        i_min = np.argmin(dist)
        if dist[i_min] > tol:
            raise ValueError(f"Cannot find close atom with distance below {tol}.")
        if symbols[i_min] != ref_symbols[i]:
            raise ValueError("The two atoms objects are not the same.")
        # check if the two atoms are the same
        mapping.append(i_min)
    # check if there is duplicate mapping
    if len(set(mapping)) != len(mapping):
        raise ValueError("The mapping is not unique.")

    mapped_atoms=Atoms(symbols=[atoms.get_chemical_symbols()[i] for i in mapping],
                       positions=[atoms.get_positions()[i] for i in mapping], cell=atoms.get_cell())
    return mapped_atoms, mapping


def test_mapping_atoms():
    """
    test the mapping_atoms function
    """
    atoms0= Atoms('H2O', positions=[[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    atoms1= Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    mapped_atoms, mapping = mapping_atoms(atoms0, atoms1)
    assert mapping == [1, 0, 2]


if __name__ == "__main__":
    test_mapping_atoms()




