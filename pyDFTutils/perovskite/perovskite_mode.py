#!/usr/bin/env python
import numpy as np
from collections import namedtuple, OrderedDict

nmode = namedtuple('nmode', [
    'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz', 'O1x', 'O1y', 'O1z', 'O2x', 'O2y',
    'O2z', 'O3x', 'O3y', 'O3z'
])

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


def label(qname, phdisp, masses, notation='IR'):
    IR_dict = {}

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
    #with open('names.txt','w') as myfile:
    #    for q in IR_translation:
    #        myfile.write('## %s\n\n'%q)
    #        myfile.write('|Cowley | ? |\n|------|-----|\n')
    #        for cname in IR_translation[q]:
    #            myfile.write('| '+cname+' | '+IR_translation[q][cname]+' |\n')
    #        myfile.write("\n")
    evec = np.array(phdisp) * np.sqrt(np.kron(masses, [1, 1, 1]))
    evec = np.real(evec) / np.linalg.norm(evec)

    mode = None
    for m in IR_dict[qname]:
        #print m
        mvec = np.real(m)
        mvec = mvec / np.linalg.norm(mvec)
        #print mvec
        p = np.abs(np.dot(np.real(evec), mvec))
        #print p
        if p > 0.5:  #1.0 / np.sqrt(2):
            print("-------------")
            print("Found! p= %s" % p)
            print("eigen vector: ", nmode._make(mvec))
            if notation == 'Cowley':
                mode = IR_dict[qname][m]
            else:
                print(IR_translation[qname])
                mode = IR_translation[qname][IR_dict[qname][m]]
            print("mode: ", mode, m)
        #return IR_dict[m]
    if mode is None:
        print("==============")
        print("eigen vector: ", nmode._make(evec))
    #return None
    return mode
