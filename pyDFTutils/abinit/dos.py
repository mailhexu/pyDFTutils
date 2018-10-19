from ase.units import Ha, eV
import re
import numpy as np
import os
import matplotlib.pyplot as plt

class abinit_ldos(object):

    def __init__(self, fname):
        self.fname=fname
        self._parse()

    def _parse(self):
        # nsppol
        with open(self.fname) as myfile:
            lines=myfile.readlines()
        _inblock=False
        self._data=[]
        for i, line in enumerate(lines):
            line=line.strip()
            if line.startswith("# nsppol"):
                line=line.replace(',', ' ')
                self.nspin = int(line.split()[3])
            if line.startswith("# Fermi energy"):
                self.efermi= float(line.split()[-1])
            if line.startswith('# energy(Ha)'):
                _inblock=True
                continue
            if _inblock and line=='':
                _inblock=False
            if _inblock:
                data_line= list(map(float, line.split()))
                self._data.append(data_line)
        self._data=np.array(self._data)

    def get_efermi(self):
        return self.efermi*Ha

    def get_energy(self, shift_fermi=False):
        if shift_fermi:
            return self._data[:,0]* Ha
        else:
            return self._data[:,0]* Ha - self.get_efermi()

    def get_ldos(self, l):
        odict={'s':0, 'p':1, 'd':2, 'f':3, 'g':4}
        if l in odict:
            return self._data[:, odict[l]+1]/Ha
        elif 0<=l<5:
            return self._data[:, l+1]/Ha
        else:
            raise ValueError("l should be spdfg or 0-4")

    def plot_dos(self):
        odict={'s':0, 'p':1, 'd':2, 'f':3, 'g':4}
        for l in odict:
            plt.plot(self.get_energy(shift_fermi=True), self.get_ldos(l), label=l)
            plt.xlim(-10, 6)
            plt.ylim(0,10)
        plt.legend()
        plt.show()

        



def test():
    from paths import path_UJ
    fname=os.path.join(path_UJ(U=4, J=0), 'abinito_DOS_AT0007')
    dos=abinit_ldos(fname)
    print(dos.get_energy())
    print(dos.get_ldos(0))
    dos.plot_dos()

#test()
