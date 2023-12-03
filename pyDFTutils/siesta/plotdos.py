import numpy as np
import matplotlib.pyplot as plt

def plot_siesta_dos(fname="siesta.DOS", efermi=-6.77):
    """
    The function plot_dos plots the DOS from a SIESTA calculation.
    The DOS file has two columns: energy and DOS
    """
    d=np.loadtxt(fname)
    plt.plot(d[:,0]-efermi,d[:,1])
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.xlim(-10,10)
    plt.axhline(0, color='black')
    plt.show()


if __name__ == "__main__":
    plot_dos(fname=sys.argv[1], efermi=sys.argv[2])


