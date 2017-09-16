#!/usr/bin/env python
from ase_utils.myvasp import myvasp
import re

default_pps_1 = {
    'Pr': 'Pr_3',
    'Ni': 'Ni',
    'Yb': 'Yb_3',
    'Pd': 'Pd',
    'Pt': 'Pt',
    'Ru': 'Ru_pv',
    'S': 'S',
    'Na': 'Na_pv',
    'Nb': 'Nb_sv',
    'Nd': 'Nd_3',
    'C': 'C',
    'Li': 'Li_sv',
    'Pb': 'Pb_d',
    'Y': 'Y_sv',
    'Tl': 'Tl_d',
    'Lu': 'Lu_3',
    'Rb': 'Rb_sv',
    'Ti': 'Ti_sv',
    'Te': 'Te',
    'Rh': 'Rh_pv',
    'Tc': 'Tc_pv',
    'Ta': 'Ta_pv',
    'Be': 'Be',
    'Sm': 'Sm_3',
    'Ba': 'Ba_sv',
    'Bi': 'Bi_d',
    'La': 'La',
    'Ge': 'Ge_d',
    'Po': 'Po_d',
    'Fe': 'Fe',
    'Br': 'Br',
    'Sr': 'Sr_sv',
    'Pm': 'Pm_3',
    'Hf': 'Hf_pv',
    'Mo': 'Mo_sv',
    'At': 'At_d',
    'Tb': 'Tb_3',
    'Cl': 'Cl',
    'Mg': 'Mg',
    'B': 'B',
    'F': 'F',
    'I': 'I',
    'H': 'H',
    'K': 'K_sv',
    'Mn': 'Mn_pv',
    'O': 'O',
    'N': 'N',
    'P': 'P',
    'Si': 'Si',
    'Sn': 'Sn_d',
    'W': 'W_pv',
    'V': 'V_sv',
    'Sc': 'Sc_sv',
    'Sb': 'Sb',
    'Os': 'Os',
    'Dy': 'Dy_3',
    'Se': 'Se',
    'Hg': 'Hg',
    'Zn': 'Zn',
    'Co': 'Co',
    'Ag': 'Ag',
    'Re': 'Re',
    'Ca': 'Ca_sv',
    'Ir': 'Ir',
    'Eu': 'Eu_3',
    'Al': 'Al',
    'Ce': 'Ce_3',
    'Cd': 'Cd',
    'Ho': 'Ho_3',
    'As': 'As',
    'Gd': 'Gd_3',
    'Au': 'Au',
    'Zr': 'Zr_sv',
    'Ga': 'Ga_d',
    'In': 'In_d',
    'Cs': 'Cs_sv',
    'Cr': 'Cr_pv',
    'Tm': 'Tm_3',
    'Cu': 'Cu',
    'Er': 'Er_3'
}
default_pps = {}
for p in default_pps_1:
    v = default_pps_1[p]
    default_pps[p] = v[len(p):]


def vasp_calculator(ecut=500, xc='LDA', nk=6, mag_order='FM', **kwargs):
    """
    default vasp calculator.

    Parameters
    ------------
    ecut: float
        energy cutoff
    xc: string
        XC functional
    nk: int
        k-point mesh  nk*nk*nk. use kpts=[nk1,nk2,nk3] for unequal nk's.
    mag_order: string
        FM|PM|A|G|C|
    **kwargs:
        args passed to myvasp.set function.

    Returns
    ------------
    A myvasp calculator object. derived from ase.calculator.vasp
    """
    mycalc = myvasp(xc=xc, setups=default_pps, kpts=[nk, nk, nk], gamma=True)
    mycalc.set(prec='Accurate',
               istart=1,
               lmaxmix=6,
               encut=ecut,
               ediff=1e-7,
               ismear=-5,
               sigma=0.05)
    mycalc.set(icharg=2)
    if mag_order == 'PM':
        mycalc.set(ispin=1)
    else:
        mycalc.set(ispin=2)
    mycalc.set(nsw=100, ibrion=1, isif=3, ediffg=-1e-4)
    mycalc.set(lreal=False, npar=2)
    mycalc.set(ldau=False,
               ldautype=2,
               ldau_luj={'Fe': {
                   'L': 2,
                   'U': U,
                   'J': 0.0
               }})
    mycalc.set(algo='normal',
               maxmix=60,
               nfree=5,
               nelm=100,
               nelmdl=-5,
               nelmin=6)
    mycalc.set(**kwargs)
    return mycalc


def read_efermi(filename='OUTCAR'):
    """
    read the fermi energy from OUTCAR.
    """
    text = open(filename, 'r').read()
    m = re.search('fermi\s*\:\s*([\d.]*)', text)
    if m:
        t = m.group(1)
    else:
        return None
    return float(t)


def read_nbands(filename='OUTCAR'):
    """
    read the number of bands form OUTCAR.
    """
    text = open(filename, 'r').read()
    m = re.search('NBANDS\s*=\s*(\d*)', text)
    if m:
        t = m.group(1)
    else:
        return None
    return int(t)
