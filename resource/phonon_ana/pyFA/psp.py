#!/usr/bin/env python
from data.electron_configuration import configurations, core_states
from ase.units import Ha
import copy
from data.ONCV_PBEsol_conf import ONCV_PBEsol_conf


class econf():
    def __init__(self, symbol):
        """
        reference electronic configuration of elements.
        """
        self.symbol = symbol
        self.whole_conf = [[None] + list(Ha2eV(x))[:-1] + [None]
                           for x in configurations[symbol][1]]

    def whole_configuration(self):
        return self.whole_conf

    def valence_configuration(self, ncore=None, valence=0):
        if ncore is None:
            ncore = core_states(self.symbol)
        # electronic configuration of 0 valence atom.
        neutral_conf = self.whole_conf[ncore:]
        v_conf = copy.copy(neutral_conf)
        v_conf = atomconf_to_ionconf(v_conf, valence)
        return v_conf

    def wannier_conf(self, ncore=None, valence=0):
        vconf = self.valence_configuration(ncore=ncore, valence=valence)
        c = []
        for s in vconf:
            m, n, l, occ, spin = s
            c.append([None, l, n, occ, None])
        return c


def get_oncv_econf(elem, fname):
    with open(fname) as myfile:
        lines = myfile.readlines()
        for line in lines[-100:]:
            if line.strip().startswith(elem):
                try:
                    e = line.strip().split()[0]
                except:
                    pass
                if elem == e:
                    w = line.strip().split()
                    z = float(w[1])
                    nc = int(w[2])
                    nv = int(w[3])
        return elem, z, nc, nv


def atomconf_to_ionconf(neutral_conf, valence):
    v_conf = copy.copy(neutral_conf)
    if valence < 0:
        m, l, n, occ, spin = v_conf[-1]
        occ -= valence
        if occ < 0 or occ > (2 * l + 1) * 2:
            raise ValueError("occupation cannot be <0 or > 2l+1")
        v_conf[-1] = (m, l, n, occ, spin)
    elif valence > 0:
        ind = -1
        nel = valence
        while nel > 0:
            m, l, n, occ, spin = v_conf[ind]
            if occ < nel:
                nel -= occ
                occ = 0
                v_conf[ind] = m, l, n, occ, spin
            else:
                occ -= nel
                nel = 0
                v_conf[ind] = m, l, n, occ, spin
                break
            ind -= 1
    return v_conf


def gen_conf_dict():
    evlist = [('Be', 2), ('Mg', 2), ('Ca', 2), ('Sr', 2), ('Ba', 2), ('Pb', 2),
              ('Bi', 3), ('Li', 1), ('Na', 1)]
    conf_dict = {}
    for ev in evlist:
        elem, val = ev
        conf = econf(elem).wannier_conf(valence=val)
        conf_dict[ev] = conf
    print conf_dict[('Li', 1)]
    return conf_dict


def gen_ion_conf_dict(evlist, atom_conf_dict):
    conf_dict = {}
    # in case a dict is "wrongly" used, which is often the case.
    if isinstance(evlist,dict):
        evlist=zip(evlist.keys(),evlist.values())
    for ev in evlist:
        elem, val = ev
        conf = atomconf_to_ionconf(atom_conf_dict[elem], val)
        conf_dict[ev] = conf
    return conf_dict


def Ha2eV(x):
    """
    (n,l,occ,energy) energy Ha->eV.
    """
    n, l, occ, energy = x
    return (n, l, occ, energy * Ha)


def test():
    ec = econf('O')
    print(ec.whole_configuration())
    print(ec.valence_configuration(valence=-2))
    print(econf('P').valence_configuration(valence=-3))
    print(econf('Ba').valence_configuration(valence=4))
    print(econf('Pb').valence_configuration(valence=4))
    print(econf('Ca').valence_configuration(valence=4))
    print('Ca', ONCV_PBEsol_conf['Be'])
    print(gen_ion_conf_dict([('Be', 1)], ONCV_PBEsol_conf))


#test()
#print gen_conf_dict()
