import numpy as np
from pyDFTutils.unfolding.DDB_unfolder import nc_unfolder, DDB_unfolder
import matplotlib.pyplot as plt

def test():
    sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    ax=DDB_unfolder('./out_DDB', sc_mat=sc_mat, kpath_bounds=[[0,0,0],[0,.5,0], [.5,.5,0],[0,0,0],[.5,.5,.5]], knames=['A','B','C','D','E']) 
    plt.savefig('sc261.png')
    plt.show()

test()
