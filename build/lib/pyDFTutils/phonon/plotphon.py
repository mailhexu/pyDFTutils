#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from ase.units import Bohr
import os.path
from pyDFTutils.ase_utils.kpoints import cubic_kpath
from collections import namedtuple


def plot_phon_from_nc(fname, title='BaT', output_filename='phonon.png'):
    """
    read phonon frequencies from .nc file.
    """
    ds = Dataset(fname, mode='r')

    #ds.variables[u'space_group'][:]
    #print ds.variables[u'primitive_vectors'][:]
    #print ds.variables.keys()
    qpoints = ds.variables['qpoints'][:]
    phfreqs = ds.variables['phfreqs'][:] * 8065.6
    phdisps = ds.variables['phdispl_cart'][:]
    masses = ds.variables['atomic_mass_units'][:]
    masses = list(masses) + [masses[-1]] * 2

    IR_modes = label_all(qpoints, phfreqs, phdisps, masses)
    #return

    print((phdisps[0, 0, :, :] / Bohr))
    print((phdisps[0, 0, :, 0] / Bohr))
    print((get_weight(phdisps[0, 0, :, :], masses)))
    phfreqs = fix_gamma(qpoints, phfreqs)

    weights_A = np.empty_like(phfreqs)
    weights_B = np.empty_like(phfreqs)
    weights_C = np.empty_like(phfreqs)

    nk, nm = phfreqs.shape

    for i in range(nk):
        for j in range(nm):
            weights_A[i, j], weights_B[i, j], weights_C[i, j] = get_weight(
                phdisps[i, j, :, :], masses)

    #for i in range(1):
    # plt.plot(weights_B[:, i], linewidth=0.1, color='gray')
    #plt.plot(weights_A[:, i], linewidth=0.1, color='gray')
    #plt.show()
    #return
    axis = None
    kpath = cubic_kpath()
    kslist = [kpath[1]] * 15
    xticks = [['$\Gamma$', 'X', 'M', '$\Gamma$', 'R', 'X'], kpath[2]]
    axis = plot_band_weight(
        kslist,
        phfreqs.T,
        weights_A.T,
        axis=axis,
        color='red',
        style='alpha',
        xticks=xticks,
        title=title)
    axis = plot_band_weight(
        kslist,
        phfreqs.T,
        weights_B.T,
        axis=axis,
        color='green',
        style='alpha',
        xticks=xticks,
        title=title)
    axis = plot_band_weight(
        kslist,
        phfreqs.T,
        weights_C.T,
        axis=axis,
        color='blue',
        style='alpha',
        xticks=xticks,
        title=title)

    tick_mode = {'R': -2, 'X': -1, 'M': 2,'Gamma':0}
    for qname in IR_modes:
        for mode in IR_modes[qname]:
            #print(mode)
            shiftx = lambda x: x-0.2 if x>0.2 else x+0.01
            axis.annotate(
                mode[1], (shiftx(xticks[1][tick_mode[qname]]) , mode[0] + 5),
                fontsize='x-small',
                color='black',wrap=True)
    plt.savefig(output_filename,dpi=300)
    plt.show()
    return qpoints, phfreqs, phdisps, masses

    #print ds.variables[u'phfreqs'][:]
    #print ds.variables[u'phdispl_cart'][:]

    #for k in ds.variables:
    #    print "--------------\n"
    #    print k
    #    print ds.variables[k][:]


def label_all(qpoints, phfreqs, phdisps, masses):
    special_qpoints = {
        #'Gamma': [0, 0.013333, 0],
        'X': [0, 0.5, 0],
        'M': [0.5, 0.5, 0],
        'R': [0.5, 0.5, 0.5]
    }
    mode_dict = {}
    for i, qpt in enumerate(qpoints):
        # print qpt
        for qname in special_qpoints:
            if np.isclose(
                    qpt, special_qpoints[qname], rtol=1e-5, atol=1e-3).all():
                mode_dict[qname] = []
                print("====================================")
                print(qname)
                phdisps_q = phdisps[i]
                for j, disp in enumerate(phdisps_q):
                    disp = disp[:, 0] + 1.0j * disp[:, 1]
                    mode = label(qname, disp, masses)
                    freq = phfreqs[i][j]
                    mode_dict[qname].append([freq, mode])
    print(mode_dict)
    return mode_dict


def label(qname, phdisp, masses, notation='IR'):
    nmode = namedtuple('nmode', [
        'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz', 'O1x', 'O1y', 'O1z', 'O2x', 'O2y',
        'O2z', 'O3x', 'O3y', 'O3z'
    ])
    IR_dict = {}

    IR_translation={}

    IR_translation['Gamma']={
        '$\Delta_1$':r'$\Gamma_4^-$',
        '$\Delta_2$':r'',
        '$\Delta_5$':r'',
    }
 
    IR_translation['R']={
        r'$\Gamma_2\prime$':'$R_2^-$',
        r'$\Gamma_{12}\prime$':'$R_3^-$',
        r'$\Gamma_{25}$':'$R_5^-$',
        r'$\Gamma_{25}\prime$':'$R_5^+$',
        r'$\Gamma_{15}$':'$R_4^-$',
    }
    
    IR_translation['X']={
        '$M_1$':'$X_1^+$',
        '$M_2\prime$':'$X_3^-$',
        '$M_3$':'$X_2^+$',
        '$M_5$':'$X_5^+$',
        '$M_5\prime$':'$X_5^-$',
    }

    IR_translation['M']={
        '$M_1$':'$M_1^+$',
        '$M_2$':'$M_3^+$',
        '$M_3$':'$M_2^+$',
        '$M_4$':'$M_4^+$',
        '$M_2\prime$':'$M_3^-$',
        '$M_3\prime$':'$M_2^-$',
        '$M_5$':'$M_5^+$',
        '$M_5\prime$':'$M_5^-$',
        }
    #with open('names.txt','w') as myfile:
    #    for q in IR_translation:
    #        myfile.write('## %s\n\n'%q)
    #        myfile.write('|Cowley | ? |\n|------|-----|\n')
    #        for cname in IR_translation[q]:
    #            myfile.write('| '+cname+' | '+IR_translation[q][cname]+' |\n')
    #        myfile.write("\n")

    zvec=nmode._make([0.0] * 15)
    
    # Gamma point
    D1_1=zvec._replace(Ay=1)
    D1_2=zvec._replace(By=1)
    D1_3=zvec._replace(O3y=1)
    D1_4=zvec._replace(O1y=1, O2y=1)

    D2 =zvec._replace(O1y=1, O2y=-1)

    D5_1=zvec._replace(Ax=1)
    D5_2=zvec._replace(Bx=1)
    D5_3=zvec._replace(O1x=1)
    D5_4=zvec._replace(O2x=1)
    D5_5=zvec._replace(O3x=1)

    D5_6=zvec._replace(Az=1)
    D5_7=zvec._replace(Bz=1)
    D5_8=zvec._replace(O1z=1)
    D5_9=zvec._replace(O2z=1)
    D5_10=zvec._replace(O3z=1)

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
        D5_10:'$\Delta_5$',
    }


    # X point
    X1_1 = nmode._make([0.0] * 15)
    X1_1 = X1_1._replace(By=1)

    X1_2 = nmode._make([0.0] * 15)
    X1_2 = X1_2._replace(O1y=1, O2y=1)

    X2p_1 = nmode._make([0.0] * 15)
    X2p_1 = X2p_1._replace(Ay=1)

    X2p_2 = nmode._make([0.0] * 15)
    X2p_2 = X2p_2._replace(O3y=1)

    X3 = nmode._make([0.0] * 15)
    X3 = X3._replace(O1y=1, O2y=-1)

    X5_1 = nmode._make([0.0] * 15)
    X5_1 = X5_1._replace(Bx=1)

    X5_2 = nmode._make([0.0] * 15)
    X5_2 = X5_2._replace(Bz=1)

    X5_3 = nmode._make([0.0] * 15)
    X5_3 = X5_3._replace(O1x=1)

    X5_4 = nmode._make([0.0] * 15)
    X5_4 = X5_4._replace(O1z=1)

    X5_5 = nmode._make([0.0] * 15)
    X5_5 = X5_5._replace(O2x=1)

    X5_6 = nmode._make([0.0] * 15)
    X5_6 = X5_6._replace(O2z=1)

    X5p_1 = nmode._make([0.0] * 15)
    X5p_1 = X5_1._replace(Ax=1)

    X5p_2 = nmode._make([0.0] * 15)
    X5p_2 = X5_2._replace(Az=1)

    X5p_3 = nmode._make([0.0] * 15)
    X5p_3 = X5_3._replace(O3x=1)

    X5p_4 = nmode._make([0.0] * 15)
    X5p_4 = X5_4._replace(O3z=1)

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
    M1 = nmode._make([0.0] * 15)
    M1 = M1._replace(O3x=1, O2y=1)

    M2 = nmode._make([0.0] * 15)
    M2 = M2._replace(O2x=1, O3y=-1)

    M3 = nmode._make([0.0] * 15)
    M3 = M3._replace(O3x=1, O2y=-1)

    M4 = nmode._make([0.0] * 15)
    M4 = M4._replace(O2x=1, O3y=1)

    M2p = nmode._make([0.0] * 15)
    M2p = M2p._replace(Az=1)

    M3p_1 = nmode._make([0.0] * 15)
    M3p_1 = M3p_1._replace(Bz=1)

    M3p_2 = nmode._make([0.0] * 15)
    M3p_2 = M3p_2._replace(O1z=1)

    M5_1 = nmode._make([0.0] * 15)
    M5_1 = M5_1._replace(O3z=1)

    M5_2 = nmode._make([0.0] * 15)
    M5_2 = M5_2._replace(O2z=1)

    M5p_1 = nmode._make([0.0] * 15)
    M5p_1 = M5p_1._replace(Bx=1)

    M5p_2 = nmode._make([0.0] * 15)
    M5p_2 = M5p_2._replace(By=1)

    M5p_3 = nmode._make([0.0] * 15)
    M5p_3 = M5p_3._replace(Ay=1)

    M5p_4 = nmode._make([0.0] * 15)
    M5p_4 = M5p_4._replace(Ax=1)

    M5p_5 = nmode._make([0.0] * 15)
    M5p_5 = M5p_5._replace(O1x=1)

    M5p_6 = nmode._make([0.0] * 15)
    M5p_6 = M5p_6._replace(O1y=1)

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
    G2p = nmode._make([0.0] * 15)
    G2p = G2p._replace(O1z=1, O2x=1, O3y=1)

    G12p_1 = nmode._make([0.0] * 15)
    G12p_1 = G12p_1._replace(O1z=1, O3y=1, O2x=-2)

    G12p_2 = nmode._make([0.0] * 15)
    G12p_2 = G12p_2._replace(O1z=1, O3y=-1)

    G25_1 = nmode._make([0.0] * 15)
    G25_1 = G25_1._replace(O1y=1, O3z=-1)

    G25_2 = nmode._make([0.0] * 15)
    G25_2 = G25_2._replace(O1x=1, O2z=-1)

    G25_3 = nmode._make([0.0] * 15)
    G25_3 = G25_3._replace(O3x=1, O2y=-1)

    G25p_1 = nmode._make([0.0] * 15)
    G25p_1 = G25p_1._replace(Bx=1)

    G25p_2 = nmode._make([0.0] * 15)
    G25p_2 = G25p_2._replace(By=1)

    G25p_3 = nmode._make([0.0] * 15)
    G25p_3 = G25p_3._replace(Bz=1)

    G15_1 = nmode._make([0.0] * 15)
    G15_1 = G15_1._replace(Ax=1)

    G15_2 = nmode._make([0.0] * 15)
    G15_2 = G15_2._replace(Ay=1)

    G15_3 = nmode._make([0.0] * 15)
    G15_3 = G15_3._replace(Az=1)

    G15_4 = nmode._make([0.0] * 15)
    G15_4 = G15_4._replace(O1y=1, O3z=1)

    G15_5 = nmode._make([0.0] * 15)
    G15_5 = G15_5._replace(O1x=1, O2z=1)

    G15_6 = nmode._make([0.0] * 15)
    G15_6 = G15_6._replace(O3x=1, O2y=1)

    IR_dict['R'] = {
        G2p: r'$\Gamma_2\prime$',
        G12p_1: r'$\Gamma_{12}\prime$',
        G12p_2: r'$\Gamma_{12}\prime$',
        G25_1: r'$\Gamma_{25}$',
        G25_2: r'$\Gamma_{25}$',
        G25_3: r'$\Gamma_{25}$',
        G25p_1: r'$\Gamma_{25}\prime$',
        G25p_2: r'$\Gamma_{25}\prime$',
        G25p_3: r'$\Gamma_{25}\prime$',
        G15_1: r'$\Gamma_{15}$',
        G15_2: r'$\Gamma_{15}$',
        G15_3: r'$\Gamma_{15}$',
        G15_4: r'$\Gamma_{15}$',
        G15_5: r'$\Gamma_{15}$',
        G15_6: r'$\Gamma_{15}$',
    }
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


def fix_gamma(qpoints, phfreqs):
    for i, qpt in enumerate(qpoints):
        if np.isclose(qpt, [0.0, 0.0, 0.0], rtol=1e-5, atol=1e-3).all():
            print("Fix")
            if i == 0:
                phfreqs[i] = phfreqs[i + 1]
            else:
                phfreqs[i] = phfreqs[i - 1]
    return phfreqs


def get_weight(disp, masses):
    ms = np.kron(masses, [1, 1, 1])
    disp = np.array(disp)
    disp = disp[:, 0] + disp[:, 1] * 1j
    w = np.real(disp.conj() * disp * ms)
    wA = sum(w[0:3])
    wB = sum(w[3:6])
    wC = sum(w[6:])
    s = sum(w)
    wA, wB, wC = wA / s, wB / s, wC / s
    return wA, wB, wC


def test_read_freq_nc():
    fname = 'BaTiO3/abinit_ifc.out_PHBST.nc'
    read_phon_freq_nc(fname)


def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=0,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=2,
                     xticks=None,
                     title=None):
    if axis is None:
        fig, a = plt.subplots()
        plt.tight_layout(pad=2.19)
        plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
    else:
        a = axis
    if title is not None:
        a.set_title(title)

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - 66,
                  np.array(ekslist).flatten().max() + 66)

    if wkslist is not None:
        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            lwidths = np.array(wkslist[i]) * width
            #lwidths=np.ones(len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lc = LineCollection(
                    segments,
                    linewidths=[2] * len(x),
                    colors=[
                        colorConverter.to_rgba(
                            color, alpha=np.abs(lwidth / (width + 0.001)))
                        for lwidth in lwidths
                    ])

            a.add_collection(lc)
    plt.ylabel('Frequency (cm$^{-1}$)')
    if axis is None:
        for ks, eks in zip(kslist, ekslist):
            plt.plot(ks, eks, color='gray', linewidth=0.001)
        a.set_xlim(0, xmax)
        a.set_ylim(yrange)
        if xticks is not None:
            plt.xticks(xticks[1], xticks[0])
        for x in xticks[1]:
            plt.axvline(x, color='gray', linewidth=0.5)
        if efermi is not None:
            plt.axhline(linestyle='--', color='black')
    return a


#plot_phon_from_nc(
#    'BaTiO3/abinit_ifc.out_PHBST.nc',
#    title='BaTiO3',
#    output_filename='phonon.png')
