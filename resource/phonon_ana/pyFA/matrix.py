import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import copy


def matrix_elem_contribution(mat, vec, asr=True):
    """
    contribution of each matrix entries to the nth eigenvalue.
    For a vector Phi= [C1...Cn], the result is R_ij= Ci.conj() * Hij * Cj
    Here we assume Phi is normalized. We also assume all the matrix entries are independent of each other.

    """
    #vec=np.ones_like(vec)
    contri = np.outer(np.conj(vec), vec) * mat
    #for i in range(vec.shape[0]):
    #    contri[i, i] /= 2.0
    if asr:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                contri[i, j] -= vec[i].conj() * vec[i] * mat[i, j] / 2
                contri[i, j] -= vec[j].conj() * vec[j] * mat[i, j] / 2
    return contri


def matrix_elem_sensitivity(vec, asr=True):
    """
    for a eigen vector C, $d\lambda/dH_{mn}=C_m*C_n*(2-\delta_{mn})$
    """
    #c = np.outer(np.conj(vec), vec )
    #vec=np.ones_like(vec)
    #vec/=np.linalg.norm(vec)
    c = np.outer(np.conj(vec), vec)
    #c += c.T
    #c=np.outer(evec,evec*2)
    #print evec.shape
    #for i in range(evec.shape[0]):
    #    c[i, i] /= 2.0
    if asr:
        for i in range(vec.shape[0]):
            for j in range(vec.shape[0]):
                c[i, j] -= vec[i].conj() * vec[i] / 2
                c[i, j] -= vec[j].conj() * vec[j] / 2
    return c


def mat_heatmap(data,
                ax=None,
                xlabel=None,
                ylabel=None,
                savefig=None,
                show=True,
                title=None,
                **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    data = np.real(data[::-1, ::1])
    # print data.shape
    # heatmap = ax.pcolor(data, cmap=plt.cm.coolwarm)
    sns.heatmap(data, ax=ax, center=0, **kwargs)  #cmap=plt.cm.seismic,
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    if data.shape[0] == 5:
        xlabels = ['A', 'B', r'$O_{\bot}$', r'$O_{\bot}$', r'$O_{\parallel}$']
        ylabels = ['A', 'B', r'$O_{\bot}$', r'$O_{\bot}$', r'$O_{\parallel}$']
    elif data.shape[0] == 15:
        xlabels = product(
            ['A', 'B', r'$O_{1}$', r'$O_{2}$', r'$O_{3}$'],
            ['x', 'y', 'z'])
        xlabels = [l[0] + l[1] for l in xlabels]
        ylabels = copy.copy(xlabels)
    ax.set_yticklabels(xlabels, minor=False)
    ax.set_xticklabels(ylabels, minor=False)
    if title is not None:
        ax.set_title(title, y=1.1)
    ax.xaxis.tick_top()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    mat_heatmap(np.ones((5, 5)))
