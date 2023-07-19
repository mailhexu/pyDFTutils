import numpy as np
import matplotlib.pyplot as plt
import os
from pyDFTutils.ase_utils import symbol_number
from pyDFTutils.siesta.mysiesta import read_xv

plt.rcParams.update({"font.size": 15})
ldict = {-1: "all", 0: "s", 1: "p", 2: "d", 3: "f"}
lmdict = {
    0: {0: "s"},
    1: {-1: "py", 0: "pz", 1: "px", 9: "p-all"},
    2: {-2: "dxy", -1: "dyz", 0: "dz2", 1: "dxz", 2: "dx2-y2", 9: "d-all"},
    -1: {9: "all"},
}


def read_efermi(pdos_fname):
    """
    read efermi from pdos_fname
    """
    with open(pdos_fname) as myfile:
        lines = myfile.readlines()
        efermi = float(lines[3].strip()[:-15].split()[2])
    return efermi


def get_pdos_data(pdos_fname, iatom=0, n=0, l=-1, m=9, fmpdos=True):
    """
    get pdos data from pdos_fname
    Parameters:
    -----------
    pdos_fname: str
        the name of pdos file
    iatom: int
        the index of atom, starting from 1. 0 means all atoms. Or the name of atom.
    n: int
        the index of orbital, starting from 1. 0 means all orbitals.
    l: int
        the index of angular momentum starting from 0. -1 means all angular momentum.
    m: int
        the index of magnetic quantum number. 9 means all magnetic quantum number.
    Returns:
    --------
    outfile: str
        the name of output file
    efermi: float
        the value of fermi energy
    """
    outfile = f"pdos_{iatom}_{n}{lmdict[l][m]}.dat"
    inp = f"""{pdos_fname}
{outfile}
{iatom}
{n}
{l}
{m}
"""
    # For example:
    # LaAlO3_SrTiO3_layer.PDOS
    # LAO_STO_pdos_Ti_3d.dat
    # Ti
    # 3
    # 2
    # 9
    with open("pdos_tmp_input.txt", "w") as myfile:
        myfile.write(inp)
    if fmpdos or (not os.path.exists(outfile)):
        if os.path.exists(outfile):
            os.remove(outfile)
        os.system("fmpdos < pdos_tmp_input.txt")
    efermi = read_efermi(pdos_fname)
    return outfile, efermi


def plot_dos_for_species(
    pdos_fname,
    xvfile="siesta.XV",
    label=None,
    n=3,
    l=2,
    m=9,
    xlim=(-10, 10),
    ylim=(None, None),
    conv_n=1,
    ax=None,
    figname=None,
    fmpdos=True,
    show=False
):
    """
    plot pdos for species
    Parameters:
    -----------
    pdos_fname: str
        the name of pdos file
    xvfile: str
        the name of xv file
    label: str
        the label of atom
    n, l, m: int
        the index of orbital, angular momentum and magnetic quantum number
    xlim: tuple
        the range of x axis
    ylim: tuple
        the range of y axis
    figname: str
        the name of output figure
    Returns:
    --------
    None
    """
    atoms = read_xv(xvfile)
    symnum = symbol_number(atoms)
    iatom = symnum[label] + 1
    outfile, efermi = get_pdos_data(pdos_fname, iatom=iatom, n=n, l=l, m=m, fmpdos=fmpdos)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    plot_pdos_ax(outfile, efermi, xlim=xlim, ylim=ylim, ax=ax, conv_n=conv_n)
    if figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()


def plot_pdos_ax(fname, efermi, ax=None, conv_n=1, xlim=(-10, 10), ylim=(None, None)):
    data = np.loadtxt(fname)
    plt.rc("font", size=16)
    n = conv_n  # 为了pdos线更平滑
    if data.shape[1] == 2:
        data[:, 1] = np.convolve(
            data[:, 1], np.array([1.0 / n] * n), mode="same"
        )  # convolution process
        # d=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same')[:-4] #convolution process
        ax.plot(data[:, 0] - efermi, data[:, 1], label=fname)
    if data.shape[1] == 3:
        data[:, 1] = np.convolve(
            data[:, 1], np.array([1.0 / n] * n), mode="same"
        )  # convolution process
        data[:, 2] = np.convolve(
            data[:, 2], np.array([1.0 / n] * n), mode="same"
        )  # convolution process
        # d=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same')[:-4] #convolution process
        ax.plot(data[:, 0] - efermi, data[:, 1], label=fname + "spin up")
        ax.plot(data[:, 0] - efermi, -data[:, 2], label=fname + "spin down")
        ax.axhline(color="black")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # plt.ylim(0, 15 )
    ax.axvline(color="red", linestyle="--")
    ax.set_xlabel("Energy (eV)")
    # ax.set_ylabel('DOS')
    # plt.title(figname)
    # plt.tight_layout()
    # plt.savefig(figname)
    # plt.show()
    # plt.close() #plt.show() have a function of close. plt.close() means close the figure.


def plot_pdos(fname, figname, efermi, xlim=(-10, 10), ylim=(None, None)):
    """
    plot pdos figure
    Parameters:
    -----------
    fname: str
        the name of pdos file
    figname: str
        the name of output figure
    efermi: float
        the value of fermi energy
    xlim: tuple
        the range of x axis
    ylim: tuple
        the range of y axis

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()
    plot_pdos_ax(fname, efermi, ax=ax, xlim=xlim, ylim=ylim)
    plt.title(figname)
    plt.tight_layout()
    plt.savefig(figname)
    # plt.show()
    plt.close()


def plot_total_dos(fname, efermi, xlim=(-6, 6), ylime=(0, 60)):
    data = np.loadtxt(fname)
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x - efermi, y)
    plt.axvline(0, color="red")
    plt.xlabel("$E-E_f$ (eV)")
    plt.ylabel("DOS")
    plt.show()
    plt.savefig("total_dos.png")


# core function
def gen_pdos_figure(
    pdos_fname, iatom, n, l, m, xlim=(-10, 10), ylim=(None, None), output_path="./"
):
    outfile, efermi = get_pdos_data(pdos_fname, iatom, n, l, m)
    figname = os.path.join(output_path, f"pdos_{iatom}_{n}{lmdict[l][m]}.png")
    plot_pdos(fname=outfile, figname=figname, efermi=efermi, xlim=xlim, ylim=ylim)


def plot_layer_pdos(
    pdos_fname, figname, iatoms, n, l, m, xlim=(-10, 10), ylim=(None, None)
):
    natoms = len(iatoms)
    fig, axes = plt.subplots(natoms, 1, sharex=True)
    for i, iatom in enumerate(iatoms):
        outfile, efermi = get_pdos_data(pdos_fname, iatom, n, l, m)
        plot_pdos_ax(outfile, efermi, ax=axes[i], conv_n=5, xlim=xlim, ylim=ylim)
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(figname)

    plt.show()


if __name__ == "__main__":
    plot_dos_for_species(
        pdos_fname="siesta.PDOS",
        xvfile="siesta.XV",
        label="Ni1",
        n=3,
        l=2,
        m=9,
        xlim=(-10, 10),
        ylim=(None, None),
        figname="pdos_Si.png",
        conv_n=5
    )
