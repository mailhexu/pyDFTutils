import numpy as np
import spglib
from ase.cell import Cell
from ase.dft.kpoints import special_paths,  parse_path_string, bandpath
import numpy as np
from collections import Counter

def kpath(cell, path=None, npoints=None, supercell_matrix=np.eye(3), eps=1e-3):
    mycell=Cell(cell)
    bpath=mycell.bandpath(path=path, npoints=npoints, eps=eps )
    kpts=bpath.kpts
    kpts=[np.dot(kpt, supercell_matrix) for kpt in kpts]
    x,X, knames= bpath.get_linear_kpoint_axis()
    return kpts, x, X, knames


def cubic_kpath(npoints=500,name=True):
    """
    return the kpoint path for cubic
    Parameters:
    ----------------
    npoints: int
        number of points.

    Returns:
    -----------------
    kpts:
        kpoints
    xs:
        x cordinates for plot
    xspecial:
        x cordinates for special k points
    """
    special_path = special_paths['cubic']
    points = special_points['cubic']
    paths = parse_path_string(special_path)
    special_kpts = [points[k] for k in paths[0]]
    kpts, xs, xspecial = bandpath(
        special_kpts, cell=np.eye(3), npoints=npoints)
    if not name:
        return kpts, xs, xspecial
    else:
        return kpts, xs, xspecial,special_path


def get_ir_kpts(atoms, mesh):
    """
    Gamma-centered IR kpoints. mesh : [nk1,nk2,nk3].
    """
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    cell = (lattice, positions, numbers)
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        mesh, cell, is_shift=[0, 0, 0])
    #print("%3d ->%3d %s" % (1, mapping[0], grid[0].astype(float) / mesh))
    #print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
    #print(grid[np.unique(mapping)] / np.array(mesh, dtype=float))
    return grid[np.unique(mapping)] / np.array(mesh, dtype=float)


def ir_kpts(atoms, mp_grid, is_shift=[0, 0, 0], verbose=True, ir=True):
    """
    generate kpoints for structure
    Parameters:
    ------------------
    atoms: ase.Atoms
      structure
    mp_grid: [nk1,nk2,nk3]
    is_shift: shift of k points. default is Gamma centered.
    ir: bool
    Irreducible or not.
    """
    cell = (atoms.get_cell(), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    # print(spglib.get_spacegroup(cell, symprec=1e-5))
    mesh = mp_grid
    # Gamma centre mesh
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        mesh, cell, is_shift=is_shift)
    if not ir:
        return (np.array(grid).astype(float) + np.asarray(is_shift) / 2.0
                ) / mesh, [1.0 / len(mapping)] * len(mapping)
    # All k-points and mapping to ir-grid points
    # for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
    #    print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh))
    cnt = Counter(mapping)
    ids = list(cnt.keys())
    weight = list(cnt.values())
    weight = np.array(weight) * 1.0 / sum(weight)
    ird_kpts = [(grid[id].astype(float) + np.asarray(is_shift) / 2.0) / mesh
                for id in ids]

    # Irreducible k-points
    # print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
    # print(grid[np.unique(mapping)] / np.array(mesh, dtype=float))
    return ird_kpts, weight


if __name__ == '__main__':
    from .ase_utils.cubic_perovskite import gen_primitive
    atoms = gen_primitive(mag_order='G')
    print(get_ir_kpts(atoms, [4, 4, 4]))
"""
exit()
lattice = np.array([[0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0]]) * 5.4
positions = [[0.875, 0.875, 0.875],
             [0.125, 0.125, 0.125]]
numbers= [1,] * 2
cell = (lattice, positions, numbers)
print(spglib.get_spacegroup(cell, symprec=1e-5))
mesh = [4, 4, 4]

#
# Gamma centre mesh
#
mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
print("%3d ->%3d %s" % (1, mapping[0], grid[0].astype(float) / mesh))
print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
print(grid[np.unique(mapping)] / np.array(mesh, dtype=float))
exit()
#print  np.unique(mapping)
#print "grid:", grid
# All k-points and mapping to ir-grid points
for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
    print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh))
    # Irreducible k-points
    print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
    print(grid[np.unique(mapping)] / np.array(mesh, dtype=float))
    continue
    # With shift
    #
    #mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[1, 1, 1])

    # All k-points and mapping to ir-grid points
    for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
        print("%3d ->%3d %s" % (i, ir_gp_id, (gp + [0.5, 0.5, 0.5]) / mesh))
        # Irreducible k-points
        print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
"""
