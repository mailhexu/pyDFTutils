#!/usr/bin/env python
import numpy as np
from collections import namedtuple, OrderedDict
from ase.data import atomic_numbers, atomic_masses

nmode = namedtuple('nmode', [
    'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz', 'O1x', 'O1y', 'O1z', 'O2x', 'O2y',
    'O2z', 'O3x', 'O3y', 'O3z'
])  # Note: A, B, Oz, Ox, Oy

IR_dict = OrderedDict()

zvec = nmode._make([0.0] * 15)

# Gamma point
D1_1 = zvec._replace(Ay=1)
D1_2 = zvec._replace(By=1)
D1_3 = zvec._replace(O3y=1)
D1_4 = zvec._replace(O1y=1, O2y=1)

D2 = zvec._replace(O1y=1, O2y=-1)

D5_1 = zvec._replace(Ax=1)
D5_2 = zvec._replace(Bx=1)
D5_3 = zvec._replace(O1x=1)
D5_4 = zvec._replace(O2x=1)
D5_5 = zvec._replace(O3x=1)

D5_6 = zvec._replace(Az=1)
D5_7 = zvec._replace(Bz=1)
D5_8 = zvec._replace(O1z=1)
D5_9 = zvec._replace(O2z=1)
D5_10 = zvec._replace(O3z=1)

IR_dict['Gamma'] = {
    D1_1: '$\Delta_1$',
    D1_2: '$\Delta_1$',
    D1_3: '$\Delta_1$',
    D1_4: '$\Delta_1$',
    D2: '$\Delta_2$',
    D5_1: '$\Delta_5$',
    D5_2: '$\Delta_5$',
    D5_3: '$\Delta_5$',
    D5_4: '$\Delta_5$',
    D5_5: '$\Delta_5$',
    D5_6: '$\Delta_5$',
    D5_7: '$\Delta_5$',
    D5_8: '$\Delta_5$',
    D5_9: '$\Delta_5$',
    D5_10: '$\Delta_5$',
}

# Z (0,0,1/2) for generating distorted structures.
Z2p_1 = zvec._replace(Az=1)
Z2p_2 = zvec._replace(O1z=1)

Z5p_1 = zvec._replace(Ax=1)
Z5p_2 = zvec._replace(Ay=1)
Z5p_3 = zvec._replace(O1x=1)
Z5p_4 = zvec._replace(O1y=1)

# X point
X1_1 = zvec._replace(By=1)
X1_2 = zvec._replace(O1y=1, O2y=1)

X2p_1 = zvec._replace(Ay=1)
X2p_2 = zvec._replace(O3y=1)

X3 = zvec._replace(O1y=1, O2y=-1)

X5_1 = zvec._replace(Bx=1)
X5_2 = zvec._replace(Bz=1)
X5_3 = zvec._replace(O1x=1)
X5_4 = zvec._replace(O1z=1)
X5_5 = zvec._replace(O2x=1)
X5_6 = zvec._replace(O2z=1)

X5p_1 = zvec._replace(Ax=1)
X5p_2 = zvec._replace(Az=1)
X5p_3 = zvec._replace(O3x=1)
X5p_4 = zvec._replace(O3z=1)

IR_dict['X'] = {
    X1_1: '$M_1$',
    X1_2: '$M_1$',
    X2p_1: '$M_2\prime$',
    X2p_2: '$M_2\prime$',
    X3: '$M_3$',
    X5_1: '$M_5$',
    X5_2: '$M_5$',
    X5_3: '$M_5$',
    X5_4: '$M_5$',
    X5_5: '$M_5$',
    X5_6: '$M_5$',
    X5p_1: '$M_5\prime$',
    X5p_2: '$M_5\prime$',
    X5p_3: '$M_5\prime$',
    X5p_4: '$M_5\prime$',
}
# M point

M = nmode._make([0.0] * 15)
M1 = M._replace(O3x=1, O2y=1)

M2 = M._replace(O2x=1, O3y=-1)

M3 = M._replace(O3x=1, O2y=-1)

M4 = M._replace(O2x=1, O3y=1)

M2p = M._replace(Az=1)

M3p_1 = M._replace(Bz=1)

M3p_2 = M._replace(O1z=1)

M5_1 = M._replace(O3z=1)

M5_2 = M._replace(O2z=1)

M5p_1 = M._replace(Bx=1)

M5p_2 = M._replace(By=1)

M5p_3 = M._replace(Ay=1)

M5p_4 = M._replace(Ax=1)

M5p_5 = M._replace(O1x=1)

M5p_6 = M._replace(O1y=1)

IR_dict['M'] = {
    M1: '$M_1$',
    M2: '$M_2$',
    M3: '$M_3$',
    M4: '$M_4$',
    M2p: '$M_2\prime$',
    M3p_1: '$M_3\prime$',
    M3p_2: '$M_3\prime$',
    M5_1: '$M_5$',
    M5_2: '$M_5$',
    M5p_1: '$M_5\prime$',
    M5p_2: '$M_5\prime$',
    M5p_3: '$M_5\prime$',
    M5p_4: '$M_5\prime$',
    M5p_5: '$M_5\prime$',
    M5p_6: '$M_5\prime$',
}
# R point

# Breathing mode
R = nmode._make([0.0] * 15)
R2p = R._replace(O1z=1, O2x=1, O3y=1)

R12p_1 = R._replace(O1z=1, O3y=1, O2x=-2)

R12p_2 = R._replace(O1z=1, O3y=-1)

R25_1 = R._replace(O1y=1, O3z=-1)

R25_2 = R._replace(O1x=1, O2z=-1)

R25_3 = R._replace(O3x=1, O2y=-1)

R25p_1 = R._replace(Bx=1)

R25p_2 = R._replace(By=1)

R25p_3 = R._replace(Bz=1)

R15_1 = R._replace(Ax=1)

R15_2 = R._replace(Ay=1)

R15_3 = R._replace(Az=1)

R15_4 = R._replace(O1y=1, O3z=1)

R15_5 = R._replace(O1x=1, O2z=1)

R15_6 = R._replace(O3x=1, O2y=1)

IR_dict['R'] = {
    R2p: r'$\Gamma_2\prime$',
    R12p_1: r'$\Gamma_{12}\prime$',
    R12p_2: r'$\Gamma_{12}\prime$',
    R25_1: r'$\Gamma_{25}$',
    R25_2: r'$\Gamma_{25}$',
    R25_3: r'$\Gamma_{25}$',
    R25p_1: r'$\Gamma_{25}\prime$',
    R25p_2: r'$\Gamma_{25}\prime$',
    R25p_3: r'$\Gamma_{25}\prime$',
    R15_1: r'$\Gamma_{15}$',
    R15_2: r'$\Gamma_{15}$',
    R15_3: r'$\Gamma_{15}$',
    R15_4: r'$\Gamma_{15}$',
    R15_5: r'$\Gamma_{15}$',
    R15_6: r'$\Gamma_{15}$',
}


def label_zone_boundary(qname,
                        phdisp=None,
                        masses=None,
                        evec=None,
                        notation='IR'):
    IR_translation = {}

    IR_translation['Gamma'] = {
        '$\Delta_1$': r'$\Gamma_4^-$',
        '$\Delta_2$': r'',
        '$\Delta_5$': r'',
    }

    IR_translation['R'] = {
        r'$\Gamma_2\prime$': '$R_2^-$',
        r'$\Gamma_{12}\prime$': '$R_3^-$',
        r'$\Gamma_{25}$': '$R_5^-$',
        r'$\Gamma_{25}\prime$': '$R_5^+$',
        r'$\Gamma_{15}$': '$R_4^-$',
    }

    IR_translation['X'] = {
        '$M_1$': '$X_1^+$',
        '$M_2\prime$': '$X_3^-$',
        '$M_3$': '$X_2^+$',
        '$M_5$': '$X_5^+$',
        '$M_5\prime$': '$X_5^-$',
    }

    IR_translation['M'] = {
        '$M_1$': '$M_1^+$',
        '$M_2$': '$M_3^+$',
        '$M_3$': '$M_2^+$',
        '$M_4$': '$M_4^+$',
        '$M_2\prime$': '$M_3^-$',
        '$M_3\prime$': '$M_2^-$',
        '$M_5$': '$M_5^+$',
        '$M_5\prime$': '$M_5^-$',
    }

    if phdisp is not None or masses is not None:
        evec = np.array(phdisp) * np.sqrt(np.kron(masses, [1, 1, 1]))
        evec = np.real(evec) / np.linalg.norm(evec)

    mode = None
    for m in IR_dict[qname]:
        mvec = np.real(m)
        mvec = mvec / np.linalg.norm(mvec)
        evec = evec / np.linalg.norm(evec)
        p = np.abs(np.dot(np.real(evec), mvec))
        if p > 0.4:  #1.0 / np.sqrt(2):
            print("-------------")
            #print("Found! p= %s" % p)
            #print("eigen vector: ", nmode._make(mvec))
            if notation.lower() == 'cowley':
                mode = IR_dict[qname][m]
            elif notation.upper() == 'IR':
                #print(IR_translation[qname])
                mode = IR_translation[qname][IR_dict[qname][m]]
            elif notation.lower() == 'both':
                mode = (IR_dict[qname][m],
                        IR_translation[qname][IR_dict[qname][m]])
            else:
                raise ValueError('notation should be Cowley|IR|both')
            #print("mode: ", mode, m)
    if mode is None:
        print("==============")
        print("eigen vector: ", nmode._make(evec))
    return mode


def label_Gamma(phdisp=None, masses=None, evec=None):
    if phdisp is not None and masses is not None:
        evec = np.array(phdisp) * np.sqrt(np.kron(masses, [1, 1, 1]))
        evec = np.real(evec) / np.linalg.norm(evec)

    basis_dict = mass_to_Gamma_basis_3d(masses, eigen_type='eigen_vector')
    mode = None

    for m in basis_dict:
        mvec = np.real(m)
        mvec = mvec / np.linalg.norm(mvec)
        evec = evec / np.linalg.norm(evec)
        p = np.abs(np.dot(np.real(evec), mvec))
        if p > 0.5:  #1.0 / np.sqrt(2):
            mode = basis_dict[m]
    if mode is None:
        print("==============")
        print("eigen vector: ", nmode._make(evec))
    return mode


def mass_to_Gamma_basis(masses=None, symbols=None, eigen_type='eigen_vector'):
    """
    return the slater-last-axe basis.
    type: eigenvector |eigendisplacement
    Notice that the sequence of the atoms are A, B, O_inplane, O_inplane, O_out_of_plane.
    """

    if masses is None and symbols is None:
        raise ValueError("Either masses or symbols of A, B, C should be given")
    if symbols is not None:
        masses = [atomic_masses[atomic_numbers[elem]] for elem in symbols]
    masses = np.sqrt(masses)
    A, B, O = masses[0], masses[1], masses[2]

    # acoustic
    s = np.array([1, 1, 1, 1, 1])
    acoustic = s / np.linalg.norm(s)

    #slater: A0, B+, O//- O|_-
    s = np.array([0, O * 3, -B, -B, -B])
    slater = s / np.linalg.norm(s)

    # axe
    s = np.array([0, 0, 2, -1, -1])
    axe = s / np.linalg.norm(s)

    # last :  A- B- O+
    s = np.array([B + O * 3, -A, -A, -A, -A])
    last = s - np.dot(s, slater) * slater
    last = last / np.linalg.norm(last)

    # Gamma4 :
    s = np.array([0, 0, 0, 1, -1])
    g4 = s / np.linalg.norm(s)

    if eigen_type == 'eigen_vector':
        slater = slater * masses
        slater = slater / np.linalg.norm(slater)
        axe = axe * masses
        axe = axe / np.linalg.norm(axe)
        last = last * masses
        last = last / np.linalg.norm(last)
        acoustic = acoustic * masses
        acoustic = acoustic / np.linalg.norm(acoustic)
        g4 = g4 * masses
        g4 = g4 / np.linalg.norm(g4)

    #return acoustic, slater, axe, last, g4
    return {
        'acoustic': acoustic,
        'slater': slater,
        'axe': axe,
        'last': last,
        'g4': g4
    }


def mass_to_Gamma_basis_3d(masses=None, symbols=None, eigen_type='eigen_vector'):
    """
    in 3 directions.
    """
    if masses is None and symbols is None:
        raise ValueError("Either masses or symbols of A, B, C should be given")
    evecs = mass_to_Gamma_basis(masses=masses, symbols=symbols, eigen_type=eigen_type)
    M = nmode._make([0.0] * 15)

    a = evecs['acoustic']
    s = evecs['slater']
    ax = evecs['axe']
    l = evecs['last']
    g = evecs['g4']

    # acoustic mode.
    v = np.zeros(15, dtype=float)
    v[0::3] = a
    Ax = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[1::3] = a
    Ay = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[2::3] = a
    Az = nmode._make(v)

    # Slater mode
    v = np.zeros(15, dtype=float)
    v[0::3] = s
    Sx = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[1::3] = s
    Sy = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[2::3] = s
    Sz = nmode._make(v)

    # Axe mode
    r1, r2 = ax[2], ax[3]  # O1, O2 -- Oz Ox
    Axex = M._replace(O1x=r2, O2x=r1, O3x=r2)
    Axey = M._replace(O1y=r2, O2y=r2, O3y=r1)
    Axez = M._replace(O1z=r1, O2z=r2, O3z=r2)

    # Last mode
    v = np.zeros(15, dtype=float)
    v[0::3] = l
    Lx = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[1::3] = l
    Ly = nmode._make(v)

    v = np.zeros(15, dtype=float)
    v[2::3] = l
    Lz = nmode._make(v)

    # Gamma4 silent mode
    r1, r2 = g[3], g[4]
    G4x = M._replace(O1x=r1, O3x=r2)  # Ox, Oy in z direction
    G4y = M._replace(O1y=r1, O2y=r2)  # Ox, Oz in y direction
    G4z = M._replace(O2z=r1, O3z=r2)  # Ox, Oy in z direction

    mode_dict = {
        Ax: 'Ax',
        Ay: 'Ay',
        Az: 'Az',
        Sx: 'Sx',
        Sy: 'Sy',
        Sz: 'Sz',
        Axex: 'Axex',
        Axey: 'Axey',
        Axez: 'Axez',
        Lx: 'Lx',
        Ly: 'Ly',
        Lz: 'Lz',
        G4x: 'G4x',
        G4y: 'G4y',
        G4z: 'G4z'
    }
    return mode_dict

def Gamma_modes(elems):
    mdict=mass_to_Gamma_basis_3d(masses=None, symbols=elems, eigen_type='eigen_vector')
    return dict(zip(['G_'+x for x in mdict.values()], mdict.keys()))

def elem_to_Gamma_basis(elems, eigen_type='eigen_displacement'):
    """
    Input A, B, X and get Slater, axe, Last mode in perovskite structure.
    eigen_type: eigen_displacement or eigen_vector. Default is eigen displacement.
    """
    masses = [atomic_masses[atomic_numbers[elem]] for elem in elems]
    return mass_to_Gamma_basis(masses, eigen_type=eigen_type)

def test():
    from ase import Atoms
    atoms=Atoms('BaTiO3')
    masses=atoms.get_masses()
    m=mass_to_Gamma_basis_3d(masses, eigen_type='eigen_vector')
    for k in m:
        print(m[k],k)
#test()
