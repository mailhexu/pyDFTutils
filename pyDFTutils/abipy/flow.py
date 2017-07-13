#!/usr/bin/env python
from __future__ import unicode_literals, division, print_function
import numpy as np
from pymatgen.io.abinit.works import BecWork, PhononWork
from pymatgen.io.abinit.flows import PhononFlow


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
            if np.allclose(qpt, 0) and with_becs:
                ph_work = BecWork.from_scf_task(scf_task)
            else:
                ph_work = PhononWork.from_scf_task(
                    scf_task, qpoints=qpt, tolerance=tolerance)

            flow.register_work(ph_work)

        if allocate:
            flow.allocate()

        return flow
