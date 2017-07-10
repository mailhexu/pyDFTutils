#!/usr/bin/env python
from pyDFTutils.wannier90.wannier import wannier_input, t2g_orbs, eg_orbs
from pyDFTutils.abinit.myabinit import default_abinit_calculator, Abinit
from ase.io import read
from ase.units import Ha
#from pyDFTutils.ase_utils.kpoints import cubic_kpath,
import numpy as np
import abipy.abilab as abilab


def calc_wannier(atoms):
    calc = default_abinit_calculator(
        mag_order='PM', is_metal=False, kpts=[4, 4, 4],pps='jth',ecut=20*Ha)
    calc.set(fband=2, nbdbuf=3)
    atoms.set_calculator(calc)

    #calc.scf_calculation(atoms, dos=True)
    #calc.ldos_calculation(atoms)

    gsr= abilab.abiopen('abinito_GSR.nc')
    kpoints=gsr.kpoints.frac_coords

    efermi = gsr.energy_terms.e_fermie

    # Wannier calculation
    winp = wannier_input(atoms)
    winp.add_basis('O', 'p')
    winp.add_basis('Ti', 'd')

    winp.set(
        kmesh_tol=1e-06,
        search_shells=24,
        num_iter=300,
        write_xyz=True,
        hr_plot=True,
        bands_plot_format='gnuplot',
        bands_plot=True,
        guiding_centres=True,
        mp_grid=[4,4,4],
        num_bands = 30,
    )
    winp.set_kpoints(kpoints)
    winp.set_energy_window([-5,9],[-4.5,4],shift_efermi=efermi)
    

    kpath = np.array([[0.0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1],
                      [0, 1, 0]]) / 2.0
    #kpath=np.dot(kpath,np.array([[1,-1,0],[1,1,0],[0,0,2]]))
    winp.set_kpath(kpath, ['GM', 'X', 'M', 'GM', 'R', 'X'], npoints=410)
    calc.set(autoparal=0)
    calc.wannier_calculation(atoms, wannier_input=winp)


def run():
    atoms = read('POSCAR.vasp')
    calc_wannier(atoms)


run()
