from pyDFTutils.ase_utils.frozenphonon import calculate_phonon
from pyDFTutils.siesta.mysiesta import get_species, MySiesta
from ase.io import read, write
import numpy as np
from phonopy import load, Phonopy
import matplotlib.pyplot as plt
import os
import copy
from siesta_flow.pdos import gen_pdos_figure, plot_layer_pdos 

def do_relax_calculation(atoms, calc, MaxForceTol=1e-2, MaxStressTol=0.1, NumCGSteps=1000,VariableCell=False, TypeOfRun='cg', path='.'):
    calc.label=f'{path}/relax/siesta'
    atoms=calc.relax(atoms, MaxForceTol=MaxForceTol, MaxStressTol=MaxStressTol, NumCGSteps=NumCGSteps, VariableCell=VariableCell,TypeOfRun=TypeOfRun)
    write(f'{path}/Results/relaxed.vasp', atoms, vasp5=True) 

    os.system(f'cp {path}/relax/siesta.out {path}/Results/siesta_relax.out') 
    os.system(f'cp {path}/relax/siesta.fdf {path}/Results/siesta_relax.fdf') 
    os.system(f'cp {path}/relax/siesta.XV {path}/Results/siesta.XV') 
    return atoms

def do_scf_calculation(atoms, calc, dos=True, band_structure=True, potential=False, UseDM=True, path='./'):
    if not os.path.exists(f'{path}/dos'):
        os.makedirs(f'{path}/dos')
    pwd=os.getcwd()
    if os.path.exists(f"{path}/dos/siesta.DM"):
        pass
    elif os.path.exists(f"{path}/relax/siesta.DM"):
        os.system(f"cp {pwd}/{path}/relax/siesta.DM {path}/dos/siesta.DM")
    dos_calc=copy.deepcopy(calc)
    dos_calc.label=f'{path}/dos/siesta'
    fdf=dos_calc['fdf_arguments']
    fdf.update({'DM.UseSaveDM':UseDM})
    if dos:
        fdf.update({'WriteEigenvalues': '.true.', 
    		'ProjectedDensityOfStates': ['-70.00 30.0 0.015 3000 eV'],
            'PDOS.kgrid_Monkhorst_Pack': ['7 0 0 0.0',
                                          '0 7 0 0.0',
                                          '0 0 7 0.0']})

    if band_structure:
        fdf.update({'BandLinesScale': 'pi/a',
                   'BandLines':['1  0.0 0.0 0.0 \Gamma',
                               '22 3.0 0.0 0.0 X',
                               '22 3.0 3.0 0.0 M',
                               '33 0.0 0.0 0.0 \Gamma',
                               '39 3.0 3.0 3.0 R',
                               '33 3.0 0.0 0.0 X']
                   })
    if potential:
        fdf.update({'SaveElectrostaticPotential': ' .true.',
                    'SaveRho': '.true.',
                    'SaveTotalCharge': '.true.',
                    'SaveIonicCharge': '.true.',
                    'SaveDeltaRho': '.true.',
                    'SaveTotalPotential': ' .true.',
                    })

    dos_calc.set_fdf_arguments(fdf)
    print(fdf)
    dos_calc.calculate(atoms)
    os.system(f'cp {path}/dos/siesta.out {path}/Results/siesta_scf.out') 
    os.system(f'cp {path}/dos/siesta.fdf {path}/Results/siesta_scf.fdf') 
    os.system(f'cp {path}/dos/siesta.PDOS {path}/Results/siesta.PDOS') 
    os.system(f'cp {path}/dos/siesta.DOS {path}/Results/siesta.DOS') 
    #os.system(f'cp {path}/dos/siesta.VH {path}/Results/siesta.VH') 
    os.system(f'cp {path}/dos/siesta.PDOS siesta.PDOS') 

    symbols=atoms.get_chemical_symbols()
    sdict={}
    for s in symbols:
        if s not in sdict:
            sdict[s]=f"{s}.{len(sdict)+1}"
    for s in set(symbols):
        gen_pdos_figure('siesta.PDOS', sdict[s], 0,-1,9 ,output_path=f'./{path}/Results', xlim=(-5,5), ylim=(-20,20))
    os.system(f'mv {path}/pdos*.dat {path}/Results/')

    for iatom in range(len(atoms)):
        gen_pdos_figure(f'siesta.PDOS', iatom+1, 0,-1,9 ,output_path=f'./{path}/Results', xlim=(-5,5), ylim=(-8,8))
    os.system(f'mv {path}/pdos*.dat {path}/Results/')
do_phonon_calculation=calculate_phonon

#def do_phonon_calculation(atoms, calc):
#        phonon_calc=copy.deepcopy(calc)
#        calculate_phonon(atoms, calc=phonon_calc, ndim=np.diag([2.,  2.,  2.]),parallel=False,symprec=1e-3)
#        #calculate_phonon(atoms, calc=phonon_calc, ndim=np.array([[-1.,  1.,  1.],
#        #   [ 1., -1.,  1.],
#        #   [ 1.,  1., -1.]]), parallel=False,symprec=1e-3)

