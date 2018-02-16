import numpy as np
import os
from pyDFTutils.perovskite.frozen_mode import gen_P21c_perovskite
from pyDFTutils.vasp.myvasp import myvasp, default_pps
from pyDFTutils.ase_utils import vesta_view

def test_distortion():
    amp=np.random.random(6)
    atoms=gen_P21c_perovskite(
        'NdNiO3',
        cell=[3.785415]*3,
        supercell_matrix=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
        modes=dict(
        #R2_m_O1=amp[0]*0.1,  # R2-[O1:c:dsp]A2u(a), O, breathing
        #R3_m_O1=-0.0023,  # R3-[O1:c:dsp]A2u(a), O JT inplane-stagger, out-of-plane antiphase
        #R3_m_O2=amp[1],  # R3-[O1:c:dsp]A2u(b), O, out-of-plane-stagger, inplane antiphase
        #R4_m_A1=amp[2],  # R4-[Nd1:a:dsp]T1u(a), A
        #R4_m_A2=0.0,  # R4-[Nd1:a:dsp]T1u(b), A
        #R4_m_A3=0.0,  # R4-[Nd1:a:dsp]T1u(c), A
        #R4_m_O1=0.0,  # R4-[O1:c:dsp]Eu(a), O
        #R4_m_O2=0.0,  # R4-[O1:c:dsp]Eu(b), O
        #R4_m_O3=0.0,  # R4-[O1:c:dsp]Eu(c), O

        R5_m_O1=0.81457,  # R5-[O1:c:dsp]Eu(a), O a-
        R5_m_O2=0.81457,  # R5-[O1:c:dsp]Eu(b), O b-
        R5_m_O3=0.81457,  # R5-[O1:c:dsp]Eu(c), O c- (not in Pnma)
        #X3_m_A1=0.0,  # X3-[Nd1:a:dsp]T1u(a), What's this..
        #X3_m_O1=0.0,  # X3-[O1:c:dsp]A2u(a)

        #X5_m_A1=0.3219,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode
        #Z5_m_A2=0.3219,  # [Nd1:a:dsp]T1u(b), A , save as above
        #X5_m_O1=-0.06299,  # [Nd1:a:dsp]T1u(a), O , Antiferro mode
        #Z5_m_O2=-0.06299,  # [Nd1:a:dsp]T1u(b), O , same as above

        #M2_p_O1=0.739,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation
        #M3_p_O1=0.00412,  # M3+[O1:c:dsp]A2u(a), O, D-type JT inplane stagger
        #M5_p_O1=0.00015,  # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting
        #M5_p_O2=-0.00015,  # M5+[O1:c:dsp]Eu(b), O, Out of phase tilting, -above
        #M4_p_O1=0.0 , # M4+[O1:c:dsp]A2u(a), O, in-plane-breathing (not in P21/c)
            )
        )
    vesta_view(atoms)
        #write("NNO_br%s.vasp"%br,atoms, vasp5=True)
    return amp, atoms

test_distortion()
def calc_energy():
    for i in range(100):
        dirname='calc_%s'%i
        os.mkdir(dirname)
        cwd=os.getcwd()
        os.chdir(dirname)
        amp, atoms=test_distortion()
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
        #energy=atoms.get_energy()
        os.chdir(cwd)
        with open('myfile.txt','a') as  myfile:
            myfile.write("%s, %s\n"%(amp))


#calc_energy()
