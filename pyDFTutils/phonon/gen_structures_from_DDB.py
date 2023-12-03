"""
Generate structure with frozen phonon displacements from a DDB file with abipy
"""
import numpy as np
from abipy.dfpt.ddb import DdbFile
import copy


def get_frozen_phonon_structure(ddb_file, ampdict, scale_matrix,  **kwargs):
    """
    Get the structure with frozen phonon displacements from a DDB file

    Args:
        ddb_file (str): Path to the DDB file
        ampdict (dict): Dictionary with the qpoint and the index of the phonon mode as key, and the amplitude of the displacement as value
          for example: {qpt1: {mode_index1: amplitude11, mode_index2: amplitude12}, qpt2: {mode_index1: amplitude21, mode_index2: amplitude22}}
        Args:
        eta: pre-factor multiplying the displacement. Gives the value in Angstrom of the
        largest displacement.
        scale_matrix: the scaling matrix of the supercell. If None a scaling matrix suitable for
            the qpoint will be determined.
        max_supercell: mandatory if scale_matrix is None, ignored otherwise. Defines the largest
            supercell in the search for a scaling matrix suitable for the q point.
        **kwargs: Keyword arguments to pass to anaget_phmodes_at_qpoints. The list of argments and the defaults are:
                qpoints=None,
                asr=2,
                chneut=1,
                dipdip=1,
                dipquad=1,
                quadquad=1,
                ifcflag=0,
                ngqpt=None,
                workdir=None,
                mpi_procs=1,
                manager=None,
                verbose=0,
                lo_to_splitting=False,
                spell_check=True,
                directions=None,
                anaddb_kwargs=None,
                return_input=False,
    Returns:
        Structure: Structure with frozen phonon displacements
    """
    ddb = DdbFile(ddb_file)
    qlist=list(ampdict.keys())
    phmodes = ddb.anaget_phmodes_at_qpoints(qlist, **kwargs)
    qpt0=qlist[0]
    mode0=phmodes.get_frozen_phonons(qpt0, nmode=0, eta=0, scale_matrix=scale_matrix)
    structure0=mode0.structure
    displacements=mode0.displ
    for qpt, amps in ampdict.items():
        for mode_index, amplitude in amps.items():
            s=copy.deepcopy(phmodes).get_frozen_phonons(qpt, nmode=mode_index, eta=amplitude, scale_matrix=scale_matrix)
            displacements+=s.displ
            print("=========\n",amplitude)
            print("=========\n",s.displ)
    structure=structure0.copy()
    # displace the atoms in the structure
    atoms=structure.to_ase_atoms()
    atoms.set_positions(atoms.get_positions()+displacements)
    return atoms


def test():
    G=(0,0,0)
    X=(0,0.0000001,0.5)
    M=(0.5,0.5,0)
    R=(0.5,0.5,0.50001)
    #atoms=get_frozen_phonon_structure("all.DDB", {R: {0: 0.2, 1:0.2, 2:0.2}}, scale_matrix=np.eye(3)*2, ifcflag=1)
    atoms=get_frozen_phonon_structure("all.DDB", {X: {0: 0.2, 1:0.2 }}, scale_matrix=np.eye(3)*2, ifcflag=1)
    atoms.write("test.vasp")

if __name__ == "__main__":
    test()
