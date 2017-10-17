#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

import os
import abipy.data as abidata
import abipy.abilab as abilab
from abipy.abilab import AbinitInput
from abipy.abilab import Structure
from ase.units import eV, Ha
from abipy.abio.factories import scf_input, ebands_input, dos_from_gsinput, ioncell_relax_from_gsinput, scf_for_phonons, phonons_from_gsinput, piezo_elastic_inputs_from_gsinput, scf_piezo_elastic_inputs, ebands_from_gsinput
import abipy.flowtk as flowapi
from pyDFTutils.abipy.input_utils import find_all_pp, set_Hubbard_U, set_spinat, to_abi_structure


def build_structure(name, mag='PM'):
    from ase_utils3.cubic_perovskite import gen_primitive
    return gen_primitive(name=name, latticeconstant=4, mag_order='PM')


def make_scf_input(atoms,
                   ecut=26,
                   xc='LDA',
                   is_metal=True,
                   spin_mode='PM',
                   pp_family='gbrv',
                   pp_labels={},
                   ldau_type=0,
                   Hubbard_U_dict={},
                   **kwargs):
    """
    structure: ase atoms object.
    """

    spin_mode_dict = {'PM': 'unpolarized', 'FM': 'polarized', 'AFM': 'afm'}
    if spin_mode in spin_mode_dict:
        spin_mode = spin_mode_dict[spin_mode]
    structure, magmoms = to_abi_structure(atoms, magmoms=True)

    default_vars = dict(
        accuracy=5,
        ecut=ecut,
        iscf=7,
        iprcel=0,
        #fband=1.4,
        nbdbuf=4,
        autoparal=1,
        paral_kgb=0,
        paral_rf=0,
        nstep=80,
        prtvol=10,
        npulayit=18,
        nline=17,
        diemac=4,
        diemix=0.6,
        nnsclo=4,
        usexcnhat=1)

    if pp_family in ['gbrv', 'jth']:
        default_vars.update(dict(usexcnhat=1, pawecutdg=ecut * 2))

    non_metal_vars = {'diemac': 5, 'occopt': 1}

    metal_vars = {'diemac': 5e5, 'occopt': 3}

    PM_vars = {'nsppol': 1, 'nspinor': 1, 'nspden': 1}

    FM_vars = {
        'occopt': 3,
        'nsppol': 2,
    }

    AFM_vars = {'nsppol': 1, 'nspinor': 1, 'nspden': 2}

    if is_metal:
        default_vars.update(metal_vars)
    else:
        default_vars.update(non_metal_vars)
    if spin_mode == 'afm' and not is_metal:
        default_vars.update(AFM_vars)
    elif spin_mode == 'afm' and is_metal:
        default_vars.update(FM_vars)
    elif spin_mode == 'unpolarized':
        default_vars.update(PM_vars)
    else:
        default_vars.update(FM_vars)
    default_vars.update(kwargs)

    pseudos = find_all_pp(
        structure, xc, family=pp_family, label_dict=pp_labels)

    scf_inp = scf_input(
        structure=structure, pseudos=pseudos, ecut=ecut, spin_mode=spin_mode)

    scf_inp.set_vars(default_vars)

    if spin_mode != 'unpolarized':
        scf_inp = set_spinat(scf_inp, magmoms)

    print(scf_inp.to_string())

    return scf_inp


def test_scf():
    atoms = build_structure(name='BaTiO3', mag='PM')
    scf_inp = make_scf_input(atoms, spin_mode='unpolarized',is_metal=False)
    ebands_inp = ebands_from_gsinput(scf_inp)

    dos_inp = dos_from_gsinput(scf_inp, dos_kppa=400)

    #scf_task = flowapi.ScfTask(scf_inp)
    #ebands_task = flowapi.NscfTask(ebands_inp)
    #dos_task = flowapi.NscfTask(dos_inp)
    band_work = flowapi.BandStructureWork(
        scf_inp, ebands_inp, dos_inp, workdir=None)

    #phonon_work=flowapi.PhononWork()
    #phonon_work.from_scf_input()

    flow = flowapi.Flow('BaTiO3_scf')
    flow.register_work(band_work)
    #flow.build_and_pickle_dump()
    #flow.plot_networkx()
    #flow.show_status()
    #flow.make_scheduler().start()

    flow_phbands = flowapi.PhononFlow.from_scf_input(
    'BaTiO3_phonon', scf_inp, ph_ngqpt=(2, 2, 2), with_becs=True)
    flow_phbands.make_scheduler().start()
    flow_phbands.show_summary()

#test_scf()


def relax():
    pass


def phonon(dirname):
    scf_inp = make_scf_input()
    flow_phbands = flowapi.PhononFlow.from_scf_input(
        dirname, scf_inp, ph_ngqpt=(2, 2, 2), with_becs=True)
    #flow_phbands.plot_networkx()
