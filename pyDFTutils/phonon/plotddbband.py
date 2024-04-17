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

plot_phonon_band(ddb_fname="O.DDB" )
