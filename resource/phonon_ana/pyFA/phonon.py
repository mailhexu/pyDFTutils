#!/usr/bin/env python
"""
phonon analysis
and interface with phonopy
"""
import yaml
import numpy as np
from ase.data import atomic_masses

conversion_factor = {'THz': 15.633302, 'cm-1': 33.35641 * 15.633302}


def read_dynamic_matrix(fname="qpoints.yaml"):
    """
    read dynamic matrix from qpoints.yaml file
    """
    data = yaml.load(open(fname))
    dynmat = []
    dynmat_data = data['phonon'][0]['dynamical_matrix']
    for row in dynmat_data:
        vals = np.reshape(row, (-1, 2))
        dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
    dynmat = np.array(dynmat)
    return dynmat


def fcm_to_dym(fcm, masses):
    natoms = fcm.shape[0]
    dym = np.zeros([natoms, natoms])
    for i in range(natoms):
        for j in range(natoms):
            dym[i, j] = fcm[i, j] / np.sqrt(masses[i] * masses[j])
    return dym


def dym_to_freq(dynmat, unit='cm-1',eigenvalues=True):
    eigvals, eigvecs, = np.linalg.eigh(dynmat)
    frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
    freqs = frequencies * conversion_factor[unit]
    if eigenvalues:
        return freqs, eigvecs, eigvals
    else:
        return freqs, eigvecs


def eig_vec_to_eig_disp(eig_vec, masses, dimension=3):
    s_masses = 1.0 / np.sqrt(np.kron(masses, np.ones(dimension)))
    eig_disp = eig_vec * s_masses
    return eig_disp


def check_zero_sum(vec, masses):
    return np.dot(vec, masses)


def mass_to_basis(masses, norm=1):
    """
    return the slater-last-axe basis.
    """
    masses = np.sqrt(masses)
    A, B, O = masses[0], masses[1], masses[2]

    #slater: A0, B+, O//- O|_-
    s = np.array([0, O * 3, -B, -B, -B])
    slater = s / np.linalg.norm(s)

    # axe
    s = np.array([0, 0, 2, -1, -1])
    axe = s / np.linalg.norm(s)

    # last :  A- B- O+
    s = np.array([B + O * 3, -A, -A, -A, -A])
    last = s - np.dot(s, slater) * slater
    last = last / np.linalg.norm(last)

    if norm == 'eigvec':
        sum_m = sum(np.sqrt(masses))
        slater /= sum_m
        axe /= sum_m
        last /= sum_m

    return slater, axe, last


def test_PTO():
    freql = []
    sl = []
    al = []
    ll = []
    nl = np.arange(0.00, 0.41, 0.1)
    for i in nl:
        fname = 'PTO%s_fcm.txt' % i
        fcm = np.loadtxt(fname)
        masses = [
            atomic_masses[atomic_numbers[name]]
            for name in ['Pb', 'Ti', 'O', 'O', 'O']
        ]
        dynmat = fcm_to_dym(fcm, masses)
        freqs, vec = dym_to_freq(dynmat)
        freql.append(freqs)

        #phonon eigen displacement dis=vec/sqrt(m)
        vec = np.array(vec) / np.sqrt(masses)
        #vec=fcm_to_vec(fcm)
        print "Freqs: ", freqs
        print "Vec: ", vec
        print "vec dot:", np.dot(masses, vec)
        slater, axe, last = mass_to_basis(masses)
        print "slater: ", slater
        print "axe: ", axe
        print "last: ", last
        s = np.abs(np.dot(slater, vec))
        a = np.abs(np.dot(axe, vec))
        l = np.abs(np.dot(last, vec))
        print "Slater, axe, last: ", s, a, l
        s, a, l = np.array([s, a, l]) / (s + a + l)
        s, a, l = np.array([s**2, a**2, l**2]) / (s**2 + a**2 + l**2)
        sl.append(s)
        al.append(a)
        ll.append(l)
    plt.figure()
    plt.plot(nl, np.array(sl) * 100, label='Slater', linewidth=2)
    plt.plot(nl, np.array(al) * 100, label='Axe', linewidth=2)
    plt.plot(nl, np.array(ll) * 100, label='Last', linewidth=2)
    plt.xlabel('$n_e$ (e/u.c.)')
    plt.ylabel('Percentage (%)')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    plt.legend()
    plt.savefig('mode.pdf')
    plt.figure()
    plt.plot(nl, freql)
    plt.show()
