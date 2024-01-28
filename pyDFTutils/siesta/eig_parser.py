
"""
Parse the siesta.dos file and return the band gap
example file of siesta.eig
 -0.890619152E+01
        284 1        123
         1  -0.506230786E+02  -0.506184022E+02  -0.506149142E+02  -0.506102295E+02  -0.309721969E+02  -0.309453032E+02  -0.309360701E+02  -0.309075280E+02  -0.308782108E+02  -0.308341749E+02
            -0.307872470E+02  -0.307580254E+02  -0.307269192E+02  -0.306978840E+02  -0.306435382E+02  -0.306323943E+02  -0.269035853E+02  -0.266792940E+02  -0.266018146E+02  -0.264623370E+02
            -0.258801940E+02  -0.258557526E+02  -0.257683228E+02  -0.257090654E+02  -0.257012242E+02  -0.256751448E+02  -0.254952340E+02  -0.254847379E+02  -0.158289624E+02  -0.155390140E+02
            -0.154944254E+02  -0.153093963
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def get_nelectron(fname):
    """
    search for the total number of electrons in the siesta.out file
    The line is like: 
      Total number of electrons:   128.000000"""
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "Total number of electrons" in line:
            nelectron = int(round(float(line.split()[4])))
            return nelectron

    

class EigParser():
    def __init__(self, prefix):
        self.prefix=prefix
        self.filename = f"{prefix}.EIG"
        self.eigs = []
        self.nkpt = 0
        self.nband = 0
        with open(self.filename, 'r') as f:
            self.lines = f.readlines()
        self.read_eig()

    def read_eig(self):
        lines=self.lines
        self.efermi= [float(x) for x in lines[1].split()]
        self.nband = int(lines[1].split()[0])
        self.nspin= int(lines[1].split()[1])
        self.nkpt= int(lines[1].split()[2])
        for line in lines[2:]:
            tokens = line.strip().split()
            try:
                i=int(tokens[0])
                if i>1:
                    #print(eig)
                    self.eigs.append(eig)
                    #print(i-1, len(eig))
                eig=[]
                eig+=[float(x) for x in tokens[1:]]
            except:
                eig+=[float(x) for x in tokens[:]]
        self.eigs.append(eig)

        self.eigs=np.array(self.eigs)
        #print(f"{self.nkpt=}, {self.nband=}, {self.nspin=}")
        #print(f"{self.eigs.shape}")

    def get_band_gap(self):
        nelectron= get_nelectron(f"{self.prefix}.out")
        nocc = nelectron // 2
        homo = np.max(self.eigs[:, nocc-1])
        lumo = np.min(self.eigs[:, nocc])
        gap = lumo - homo
        return gap


def test():
    factors=[1.0, 0.5, 0.0]

    modes=["R3-", "R5-", "X5+", "M2-", "M5-"]
    for mode in modes:
        gaps=[]
        for f in factors:
            print(f'scf_{mode}_f{f}U3.5')
            p=EigParser(prefix=f'scf_{mode}_f{f}U3.5/scf/siesta')
            gap=p.get_band_gap()
            #print(f"{mode=}, {f=}: {gap=:.3f}")
            gaps.append(gap)
        plt.plot(factors, gaps, label=mode, marker="o")
    plt.legend()
    plt.xlabel("$\lambda$")
    plt.ylabel("Bandgap (eV)")
    plt.savefig("bandgap_vs_mode.png")
    plt.savefig("bandgap_vs_mode.pdf")
    #plt.show()
            
def get_bandgap(prefix):
    p=EigParser(prefix=prefix)
    gap=p.get_band_gap()
    return gap

if __name__=="__main__":
    gap=get_bandgap(prefix=sys.argv[1])
    print(f"Found gap: {gap} eV.")
