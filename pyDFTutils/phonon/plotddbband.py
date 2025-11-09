import abipy
from abipy.abilab import abiopen
from pyDFTutils.ase_utils.kpoints import kpath, get_path_special_points
import matplotlib.pyplot as plt

def plot_phonon_band(ddb_fname, knames=None, dipdip=False, units="cm-1", show=True, savefig="phonon.pdf"):
    f=abiopen(ddb_fname)
    atoms= f.structure.to_ase_atoms()
    # get the phonon band structure
    cell=atoms.get_cell()
    knames, kpoints = get_path_special_points(cell, knames)
    print("Qpoint path:")
    for kname, kpt in zip(knames[0], kpoints[0]):
        print(f"{kname}: {kpt}")
    ph, _dos = f.anaget_phbst_and_phdos_files(nqsmall=0, dipdip=1, qptbounds=kpoints[0], ndivsm=40)
    fig=ph.plot_phbands(units=units, show=False)
    plt.tight_layout()
    plt.savefig(savefig) 

    if show:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot phonon band structure")
    parser.add_argument("ddb_fname", help="DDB file name")
    parser.add_argument("--show", action="store_true", help="Show the plot")
    parser.add_argument("--savefig", help="Save the plot to file", default="phonon.pdf")
    args = parser.parse_args()
    plot_phonon_band(args.ddb_fname, show=args.show, savefig=args.savefig)

if __name__ == "__main__":
    main()
