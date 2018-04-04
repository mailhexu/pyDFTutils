#!/usr/bin/env python

from netCDF4 import Dataset
from abipy.abilab import abiopen
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
import os.path


def get_weight(disp, masses):
    """
    disp (complex)
    """
    ms = np.kron(masses, [1, 1, 1])
    natoms = len(masses)
    disp = np.array(disp)
    #disp = disp[:, 0] + disp[:, 1] * 1j
    w = np.real(disp.conj() * disp*ms)
    s = sum(w)
    return w.reshape([natoms, 3]).sum(axis=1) / s

def fix_gamma(qpoints, phfreqs):
    for i, qpt in enumerate(qpoints):
        if np.isclose(qpt, np.array([0.0, 0.0, 0.0]), rtol=1e-5, atol=1e-3).all():
            if i == 0:
                phfreqs[i] = phfreqs[i + 1]
            else:
                phfreqs[i] = phfreqs[i - 1]
    return phfreqs



def colordict_elem_to_atom(colordict_elem, symbols):
    """
    colordict_elem: color-elem pair dictionary
    symbols: symbols of atoms
    """
    cdict = {}
    for color in colordict_elem:
        cdict[color] = []
    rdict = dict(zip(colordict_elem.values(), colordict_elem.keys()))
    for i, symbol in enumerate(symbols):
        if symbol in rdict:
            cdict[rdict[symbol]].append(i)
    return cdict


def auto_colordict_elem(symbols):
    """
    symbols to colordict. If number os species >3, all in blue.
    """
    if len(set(symbols)) > 3:
        return {'blue': list(range(len(set(symbols))))}
    colors = ['red', 'green', 'blue']
    ss = []
    for symbol in symbols:
        if symbol not in ss:
            ss.append(symbol)
    cdict = dict(zip(colors, ss))
    return cdict


def plot_phon_from_nc(fname,
                      title=None,
                      axes=None,
                      colordict_elem=None,
                      colordict_atom=None,
                      output_filename='phonon.png',
                      show=False):
    f = abiopen(fname)
    nqpts = f.phbands.num_qpoints
    x_qpts = np.arange(nqpts)
    ticks, labels = f.phbands._make_ticks_and_labels(qlabels=None)
    qpoints = f.phbands.qpoints.frac_coords
    phfreqs = f.phbands.phfreqs * 8065.6
    phfreqs = fix_gamma(qpoints, phfreqs)
    phdisp = f.phbands.phdispl_cart
    atoms = f.structure.to_ase_atoms()
    natoms = len(atoms)
    masses = atoms.get_masses()
    symbols = atoms.get_chemical_symbols()
    ntypesp = f.structure.ntypesp

    nk, nm = phfreqs.shape

    if colordict_elem == 'auto':
        colordict_elem = auto_colordict_elem(symbols)
    if colordict_elem is not None:
        colordict_atom = colordict_elem_to_atom(colordict_elem, symbols)
    colordict = colordict_atom

    if axes is None:
        fig, axes = plt.subplots()

    if colordict_elem is None and colordict_atom is None:
        for i in range(nm):
            axes.plot(f.phbands.phfreqs[:, i])

    if colordict is not None:
        weights = np.zeros([natoms, nk, nm], dtype=float)
        for i in range(nk):
            for j in range(nm):
                #print( get_weight(
                #    phdisp[i, j, :], masses).shape)
                weights[:, i, j] = get_weight(phdisp[i, j, :], masses)
        weight_color = {}
        for color in colordict:
            weight_color[color] = np.sum(weights[np.array(
                colordict[color], dtype=int), :, :],
                                         axis=0)
            #print(weight_color[color])

    return weight_color
    kslist = [x_qpts] * (3 * len(symbols))

    xticks = [labels, ticks]

    #for i in range(nm):
    #    axes.plot(f.phbands.phfreqs[:, i], color='gray', linewidth=0.01)
    axes.set_xlim(x_qpts[0], x_qpts[-1])

    print(colordict)
    #print(symbols)
    for color in colordict:
        axes = plot_band_weight(
            kslist,
            phfreqs.T,
            wkslist=weight_color[color].T*3,
            axis=axes,
            color=color,
            style='alpha',
            xticks=xticks, )
    return axes


def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=0,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=1,
                     xticks=None,
                     cmap=mpl.cm.coolwarm,
                     weight_min=-4,
                     weight_max=4,
                     show=False):
    if axis is None:
        fig, a = plt.subplots()
    else:
        a = axis
    if efermi is not None:
        ekslist = np.array(ekslist) - efermi

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - 0.66,
                  np.array(ekslist).flatten().max() + 0.66)

    if wkslist is not None:
        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            #lwidths=np.ones(len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lwidths = np.array(wkslist[i]) * width
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lwidths = np.array(wkslist[i]) * width
                lc = LineCollection(
                    segments,
                    linewidths=[2.5] * len(x),
                    colors=[
                        colorConverter.to_rgba(
                            color, alpha=lwidth / (width + 0.001) / 1.2)
                        for lwidth in lwidths
                    ])
            elif style == 'color' or style == 'colormap':
                lwidths = np.array(wkslist[i]) * 1
                norm = mpl.colors.Normalize(vmin=weight_min, vmax=weight_max)
                #norm = mpl.colors.SymLogNorm(linthresh=0.03,vmin=weight_min, vmax=weight_max)
                m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                #lc = LineCollection(segments,linewidths=np.abs(norm(lwidths)-0.5)*1, colors=[m.to_rgba(lwidth) for lwidth in lwidths])
                lc = LineCollection(
                    segments,
                    linewidths=lwidths,
                    colors=[m.to_rgba(lwidth) for lwidth in lwidths])

            a.add_collection(lc)
    for ks, eks in zip(kslist, ekslist):
        a.plot(ks, eks, color='gray', linewidth=0.01)
    a.set_xlim(0, xmax)
    #a.set_ylim(yrange)
    if xticks is not None:
        a.set_xticks(xticks[1])
        a.set_xticklabels(xticks[0])
        for x in xticks[1]:
            a.axvline(x, color='gray', linewidth=0.2)
    if efermi is not None:
        a.axhline(efermi, linestyle='--', color='black')
    if show:
        plt.show()
    return a


#plot_phon_from_nc(
#    "/Users/hexu/phondb/phbst_nodipdip/run.abo_PHBST.nc",
#    colordict_elem='auto')

#plot_phon_from_nc("/Users/hexu/phon_web/Data/db_data/PbTiO3_59d9e9136a71af96976874ae/phonon/phbst/phbst_dipdip/run.abo_PHBST.nc",
#    colordict_elem='auto')

#plt.show()
