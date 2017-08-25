#!/usr/bin/env python

from pyDFTutils.vasp.myvasp import myvasp, default_pps
from pyDFTutils.vasp.vasp_utils import read_efermi
from pyDFTutils.ase_utils.geometry import gen_disped_atoms
from ase.io import read
from pyDFTutils.wannier90.wannier import wannier_input,run_wannier
import os


def calc():
    atoms = read('POSCAR.vasp')
    d_atoms = gen_disped_atoms(atoms, 'Ti1', distance=0.005, direction='all')
    # original
    pwd = os.getcwd()
    path='orig'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    calc_wannier(atoms)
    os.chdir(pwd)

    # displaced
    pwd = os.getcwd()
    path='disp_Ti_x'
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    calc_wannier(d_atoms[0])
    os.chdir(pwd)




def calc_wannier(atoms):
    mycalc = myvasp(
        xc='PBE',
        gga='PS',
        setups=default_pps,
        ispin=2,
        icharg=0,
        kpts=[6, 6, 6],
        gamma=True,
        prec='normal',
        istart=1,
        lmaxmix=4,
        encut=500)
    mycalc.set(lreal='Auto', algo='normal')

    atoms.set_calculator(mycalc)
    # electronic
    mycalc.set(ismear=-5, sigma=0.1, nelm=100, nelmdl=-6, ediff=1e-7)
    mycalc.set(ncore=1, kpar=3)
    mycalc.scf_calculation()

    mycalc.set(
        lwannier90=True,
        lwrite_unk=False,
        lwrite_mmn_amn=True,
        ncore=1,
        kpar=3)
    wa = wannier_input(atoms=atoms)
    efermi = read_efermi()
    wa.set(
        mp_grid=[6, 6, 6],
        num_bands=28,
        guiding_centres=True,
        num_iter=100,
        kmesh_tol=1e-9,
        search_shells=24,
        write_xyz=True,
        hr_plot=True,
    )
    wa.set_energy_window([-70,0.5],[-67.4,0.4],shift_efermi=efermi)
    wa.add_basis('Ba','s')
    wa.add_basis('Ba','p')
    wa.add_basis('Ti','s')
    wa.add_basis('Ti','p')
    wa.add_basis('O','s')
    wa.add_basis('O','p')
    wa.write_input()

    mycalc.set(nbands=28)
    mycalc.scf_calculation()
    run_wannier(spin='up')
    run_wannier(spin='dn')
    #mycalc.ldos_calculation()

calc()
