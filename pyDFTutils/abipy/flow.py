#!/usr/bin/env python
from __future__ import unicode_literals, division, print_function
import numpy as np
from abipy.abio.inputs import *
from pymatgen.io.abinit.works import BecWork, PhononWork
from pymatgen.io.abinit.tasks import ScfTask
from pymatgen.io.abinit.flows import PhononFlow
_TOLVARS = set([
    'toldfe',
    'tolvrs',
    'tolwfr',
    'tolrff',
    "toldff",
    "tolimg", # ?
    "tolmxf",
    "tolrde",
])


class myScfTask(AbinitInput):
    def make_strain_and_bec_perts_inputs(self, bec=True, strain=True, tolerance=None, kptopt_gamma=2):
        if tolerance is None:
            tolerance = {"tolvrs": 1.0e-12}
        if len(tolerance) != 1 or any(k not in _TOLVARS for k in tolerance):
            raise self.Error("Invalid tolerance: {}".format(str(tolerance)))

        if bec:
            perts_bec = self.abiget_irred_phperts(qpt=(0, 0, 0))
        else:
            perts_bec=[]

        if strain:
            perts_strain = self.abiget_irred_strainperts(kptopt=2)
            print(perts_strain)
        else:
            perts_strain=[]

        perts=[]
        for pert in perts_bec:
            if pert not in perts:
                perts.append(pert)
        for pert in perts_strain:
            if pert not in perts:
                perts.append(pert)

        # Build list of datasets (one input per perturbation)
        multi = MultiDataset.replicate_input(input=self, ndtset=len(perts))

        for pert, inp in zip(perts, multi):
            rfdir = 3 * [0]
            rfdir[pert.idir -1] = 1
            if pert.ipert <= len(self.structure):
                inp.set_vars(rfphon=1,             # Activate the calculation of the atomic dispacement perturbations
                             rfelfd=3*int(bec),
                             rfatpol=[pert.ipert, pert.ipert],
                             rfdir=rfdir,
                             nqpt=1,               # One wavevector is to be considered
                             qpt=(0, 0, 0),        # q-wavevector.
                             kptopt=kptopt_gamma,             # No symmetries
                             iscf=7,
                             paral_kgb=0
                             )
            elif pert.ipert == len(self.structure) + 3:
                inp.set_vars(rfstrs=1,             # Activate the calculation of the strain perturbations (uniaxial)
                             #rfelfd=3*int(bec),
                             rfdir=rfdir,
                             nqpt=1,               # One wavevector is to be considered
                             qpt=(0, 0, 0),        # q-wavevector.
                             kptopt=kptopt_gamma,             # No symmetries
                             )

class BecStrainWork(BecWork):
    """
    Work for the computation of the Born effective charges, and strain

    This work consists of DDK tasks and phonon + electric field perturbation + strain perturbation,
    It provides the callback method (on_all_ok) that calls mrgddb to merge the
    partial DDB files produced by the work.
    """

    @classmethod
    def from_scf_task(cls, scf_task, ddk_tolerance=None):
        """Build a BecWork from a ground-state task."""
        if not isinstance(scf_task, ScfTask):
            raise TypeError("task %s does not inherit from GsTask" % scf_task)

        new = cls() #manager=scf_task.manager)

        # DDK calculations
        multi_ddk = scf_task.input.make_ddk_inputs(tolerance=ddk_tolerance)

        ddk_tasks = []
        for ddk_inp in multi_ddk:
            ddk_task = new.register_ddk_task(ddk_inp, deps={scf_task: "WFK"})
            ddk_tasks.append(ddk_task)

        # Build the list of inputs for electric field perturbation and phonons
        # Each bec task is connected to all the previous DDK task and to the scf_task.
        bec_deps = {ddk_task: "DDK" for ddk_task in ddk_tasks}
        bec_deps.update({scf_task: "WFK"})

        str_bec_inputs = scf_task.input.make_strain_and_bec_perts_inputs() #tolerance=efile
        for bec_inp in str_bec_inputs:
             new.register_bec_task(bec_inp, deps=bec_deps)

        return new


class myPhononFlow(PhononFlow):
    """
        1) One workflow for the GS run.
        2) nqpt works for phonon calculations. Each work contains
           nirred tasks where nirred is the number of irreducible phonon perturbations
           for that particular q-point.
    """

    @classmethod
    def from_scf_input(cls,
                       workdir,
                       scf_input,
                       ph_ngqpt,
                       with_becs=True,
                       with_strain=True,
                       tolerance=None,
                       manager=None,
                       allocate=True):
        """
        Create a `PhononFlow` for phonon calculations from an `AbinitInput` defining a ground-state run.
        Args:
            workdir: Working directory of the flow.
            scf_input: :class:`AbinitInput` object with the parameters for the GS-SCF run.
            ph_ngqpt: q-mesh for phonons. Must be a sub-mesh of the k-mesh used for
                electrons. e.g if ngkpt = (8, 8, 8). ph_ngqpt = (4, 4, 4) is a valid choice
                whereas ph_ngqpt = (3, 3, 3) is not!
            with_becs: True if Born effective charges are wanted.
            manager: :class:`TaskManager` object. Read from `manager.yml` if None.
            allocate: True if the flow should be allocated before returning.
        Return:
            :class:`PhononFlow` object.
        """
        flow = cls(workdir, manager=manager)

        # Register the SCF task
        flow.register_scf_task(scf_input)
        scf_task = flow[0][0]

        # Make sure k-mesh and q-mesh are compatible.
        scf_ngkpt, ph_ngqpt = np.array(scf_input["ngkpt"]), np.array(ph_ngqpt)

        if any(scf_ngkpt % ph_ngqpt != 0):
            raise ValueError("ph_ngqpt %s should be a sub-mesh of scf_ngkpt %s"
                             % (ph_ngqpt, scf_ngkpt))

        # Get the q-points in the IBZ from Abinit
        qpoints = scf_input.abiget_ibz(
            ngkpt=ph_ngqpt, shiftk=(0, 0, 0), kptopt=1).points

        # Create a PhononWork for each q-point. Add DDK and E-field if q == Gamma and with_becs.
        for qpt in qpoints:
            if np.allclose(qpt, 0) and with_becs and with_strain:
                ph_work = BecStrainWork.from_scf_task(scf_task)
            elif np.allclose(qpt, 0) and with_becs and (not with_strain):
                ph_work = BecWork.from_scf_task(scf_task)
            else:
                ph_work = PhononWork.from_scf_task(scf_task, qpoints=qpt, tolerance=tolerance)
            flow.register_work(ph_work)
        if allocate:
            flow.allocate()

        return flow
