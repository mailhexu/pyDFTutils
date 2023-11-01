import numpy as np
import torch
#torch.set_default_device("cuda")
from ase.constraints import ExpCellFilter, UnitCellFilter, StrainFilter
from ase.optimize import FIRE, BFGSLineSearch, BFGS, QuasiNewton
from ase import Atoms
from ase.io import read, write, Trajectory
from phonopy.units import VaspToTHz
from pyDFTutils.ase_utils.frozenphonon import calculate_phonon
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry
from pymatgen.io.ase import AseAtomsAdaptor



def ase_to_pymatgen(atoms):
    """
    ase atoms transform into pymatgen structure
    """
    return AseAtomsAdaptor.get_structure(atoms)


def relax_with_ml(
    atoms,
    calc=None,
    relax_cell=True,
    sym=True,
    traj_file="relax.traj",
    output_file="POSCAR_relax.vasp",
    fmax=0.001,
    cell_factor=1000,
):
    """
    Perform relaxation of atomic positions and/or cell shape using the FIRE algorithm.

    Args:
        atoms (ase.Atoms): The atoms object to be relaxed.
        calc (ase.Calculator): The calculator object to be used for energy and force calculations.
        relax_cell (bool, optional): Whether to relax the cell shape as well. Defaults to True.
        sym (bool, optional): Whether to impose symmetry constraints on the atoms. Defaults to True.
        traj_file (str, optional): The name of the file to write the trajectory to. Defaults to "relax.traj".
        output_file (str, optional): The name of the file to write the relaxed structure to. Defaults to "POSCAR_relax.vasp".
        fmax (float, optional): The maximum force allowed on each atom. Defaults to 0.001.
        cell_factor (float, optional): The factor by which to scale the unit cell when relaxing the cell shape. Defaults to 1000.

    Returns:
        ase.Atoms: The relaxed atoms object.
    """
    catoms = atoms.copy()
    if isinstance(calc, str):
        calc=init_calc(model_type=calc)
    elif calc is None:
        calc=init_calc(model_type="chgnet")
    catoms.calc = calc
    if sym:
        catoms.set_constraint(FixSymmetry(catoms))
    if relax_cell:
        ecf = UnitCellFilter(catoms, cell_factor=cell_factor)
        opt = FIRE(ecf)
        opt.run(fmax=fmax*10)
        opt = BFGS(ecf)
        opt.run(fmax=fmax)
    else:
        opt = FIRE(catoms)
    traj = Trajectory(traj_file, "w", catoms)
    opt.attach(traj)
    opt.run(fmax=fmax)
    return catoms

def init_calc(model_type="chgnet", model_path=None):
    if model_type.lower()=="matgl":
        import matgl
        from matgl.ext.ase import M3GNetCalculator as M3GCalc
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calc = M3GCalc(potential=pot, stress_weight=1.0)
    elif model_type.lower()=="m3gnet":
        from m3gnet.models import M3GNet, Potential
        from m3gnet.models import M3GNetCalculator as M3GCalc
        if model_path is None:
            potential=Potential(M3GNet.load())
        else:
            potential=Potential(M3GNet.from_dir(model_path))
        calc=M3GCalc(potential=potential, compute_stress=True)
    elif model_type.lower()=="chgnet":
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        calc=CHGNetCalculator(model=None)
    else:
        raise ValueError("")
    return calc




def relax_with_matgl(atoms, **kwargs):
    """
    Use the MatGL potential to relax the atomic positions and cell shape of the given atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to relax.
    **kwargs : dict
        Additional keyword arguments to pass to the `relax_with_ml` function.

    Returns
    -------
    relaxed_atoms : ase.Atoms
        The relaxed atoms object.
    """
    import matgl
    from matgl.ext.ase import M3GNetCalculator as M3GCalc
    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = M3GCalc(potential=pot, stress_weight=1.0)
    return relax_with_ml(atoms, calc, **kwargs)



XCdict={"PBE":0, "GLLB-SC":1, "HSE":2, "SCAN":3}

class MatGLStructureRelaxer():
    def __init__(self):
        self._pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calc = M3GCalc(potential=self._pot, stress_weight=1.0)

    def relax(self, atoms, **kwargs):
        return relax_with_ml(atoms, self._calc, **kwargs)


class MatGLGapPredictor():
    def __init__(self):
        #matgl.clear_cache()
        self._model= matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    def predict_gap(self, atoms, xc):
        struct=ase_to_pymatgen(atoms)
        graph_attrs = torch.tensor([XCdict[xc]])
        # For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
        bandgap = self._model.predict_structure(
            structure=struct, state_feats=graph_attrs
        )
        #print(f"The predicted {xc} band gap for CsCl is {float(bandgap):.3f} eV.")
        return float(bandgap)

def predict_gap(atoms, xc="PBE"):
    p=MatGLGapPredictor()
    return p.predict_gap(atoms, xc)
