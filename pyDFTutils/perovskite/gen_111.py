#!/usr/bin/env python
from pyDFTutils.perovskite.frozen_mode import gen_distorted_perovskite
from pyDFTutils.ase_utils import vesta_view
from pyDFTutils.ase_utils import substitute_atoms
from ase.io import write

def gen():
    atoms = gen_distorted_perovskite(
        name='NdNiO3',
        cell=[3.79, 3.79, 3.79],
        #supercell_matrix=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
        #supercell_matrix=[[1, 1, 1.0], [-1, 1, 1.0], [0, -2, 1.0]],
        supercell_matrix=[[1, 1, 2.0], [-1, 1, 2.0], [0, -2, 2.0]],
        out_of_phase_rotation=0.4,
        in_phase_rotation=0.3,
        #in_phase_tilting=0.3,
        #JT_a=1.35,
        #JT_d=1.05,
        #breathing=0.5
    )
    #atom=substitute_atoms(atoms,['Ni5','Ni10'],['Ga','Ga'])
    return atoms

write('P21c.cif',gen())
vesta_view(gen())
