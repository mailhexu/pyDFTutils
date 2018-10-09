#!/usr/bin/env python
import numpy as np
import scipy.linalg as sl
import cmath


class mytb(object):
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        self.dim_k = dim_k
        self.dim_r = dim_r
        self.lat = lat
        self.orb = orb
        self.norb = len(orb)
        self.per = per
        self.nspin = nspin
        self.ham = {}
        self.ham_R = {}

        G = np.zeros((3, 3))
        G[0] = 2 * np.pi * np.cross(lat[1], lat[2]) / (np.inner(
            lat[0], np.cross(lat[1], lat[2])))
        G[1] = 2 * np.pi * np.cross(lat[2], lat[0]) / (np.inner(
            lat[1], np.cross(lat[2], lat[0])))
        G[2] = 2 * np.pi * np.cross(lat[0], lat[1]) / (np.inner(
            lat[2], np.cross(lat[0], lat[1])))

        self.G = G

    def set_onsite(self, val, ind_i):
        self.ham[(ind_i, ind_i, (0, ) * self.dim_r)] = val
        self.ham_R[(ind_i, ind_i, (0, ) * self.dim_r)] = val

    def set_hop_dis(self, val, i, j, dis):
        self.ham[(i, j, tuple(dis))] = val

    def set_hop(self, val, i, j, R, allow_conjugate_pair=True):
        dis = np.dot(self.lat,
                     R) + np.array(self.orb[j]) - np.array(self.orb[i])
        self.ham[(i, j, tuple(dis))] = val
        self.ham_R[i, j, tuple(R)] = val

    def is_hermitian(self, H):
        return np.isclose(H, H.T.conj(), atol=1e-6).all()

    def make_hamk(self, k):
        """
        build k space hamiltonian.
        """
        k = np.dot(self.G, np.asarray(k))
        hamk = np.zeros((self.norb, self.norb, ), dtype='complex')
        for key in self.ham:
            i, j, dis = key
            val = self.ham[key]
            hamk[i, j] += val * cmath.exp(1j * np.inner(k, dis))

        # if not self.is_hermitian(hamk):
        #    print "Hamk is not Hermitian"
        hamk = (hamk + hamk.T.conj()) / 2.0
        return hamk

    def solve_k(self, k, eigvec=False):
        hamk = self.make_hamk(k)
        if eigvec:
            eigval, eigvec = sl.eigh(hamk)
            eigvec=np.linalg.qr(eigvec)[0]
            return eigval, eigvec
        else:
            eigval = sl.eigvalsh(hamk)
            return eigval

    def solve_all(self, k_vec):
        eigvals = []
        for k in k_vec:
            eigval = self.solve_k(k)
            eigvals.append(eigval)
        return np.array(eigvals).T


def test():
    tb = mytb(3, 3, lat=np.eye(3), orb=[(0, 0, 0)])
    tb.set_onsite(1, 0)
    tb.set_hop(1, 0, 0, (0, 0, 1))
    print "real space H:\n", tb.ham
    print tb.make_hamk([0, 0, 0.2])

    print tb.solve_k([0, 0, 0.2])
    kpath = [(0, 0, x) for x in np.arange(0, 0.5, 0.02)]
    eigvals = tb.solve_all(kpath)
    print eigvals[:, 0]

    import matplotlib.pyplot as plt
    plt.plot(eigvals[0])
    plt.show()


if __name__ == '__main__':
    test()
