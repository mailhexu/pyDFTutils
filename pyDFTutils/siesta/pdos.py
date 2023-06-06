import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 15})
ldict={-1:'all',0:'s', 1:'p', 2:'d', 3:'f'}
lmdict={0:{0:'s'},
        1:{-1:'py',
            0:'pz',
            1:'px',
            9:'p-all'},
        2:{-2:'dxy', -1:'dyz', 0:'dz2', 1:'dxz', 2:'dx2-y2',9:'d-all'},
        -1:{9:'all'}
        }

def read_efermi(pdos_fname):
    with open(pdos_fname) as myfile:
        lines=myfile.readlines()
        efermi=float(lines[3].strip()[:-15].split()[2])
    return efermi

def get_pdos_data(pdos_fname, iatom, n, l, m):
    outfile=f"pdos_{iatom}_{n}{lmdict[l][m]}.dat"
    inp=f"""{pdos_fname}
{outfile}
{iatom}
{n}
{l}
{m}
"""
    # For example:
    #LaAlO3_SrTiO3_layer.PDOS
    #LAO_STO_pdos_Ti_3d.dat
    #Ti
    #3
    #2
    #9
    if os.path.exists(outfile):
        os.remove(outfile)
    with open('pdos_tmp_input.txt', 'w') as myfile:
        myfile.write(inp)
    os.system("fmpdos < pdos_tmp_input.txt")
    efermi=read_efermi(pdos_fname)
    return outfile, efermi


def plot_pdos_ax(fname, efermi, ax=None, conv_n=1, xlim=(-10,10), ylim=(None, None)):
    data=np.loadtxt(fname)
    plt.rc('font', size=16)
    n=conv_n #为了pdos线更平滑
    if data.shape[1]==2:
        data[:,1]=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same') #convolution process 
        #d=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same')[:-4] #convolution process 
        ax.plot(data[:,0]-efermi, data[:,1],label=fname)
    if data.shape[1]==3:
        data[:,1]=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same') #convolution process 
        data[:,2]=np.convolve(data[:,2], np.array([1.0/n]*n),mode='same') #convolution process 
        #d=np.convolve(data[:,1], np.array([1.0/n]*n),mode='same')[:-4] #convolution process 
        ax.plot(data[:,0]-efermi, data[:,1],label=fname+'spin up')
        ax.plot(data[:,0]-efermi, -data[:,2],label=fname+'spin down')
        ax.axhline(color='black')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #plt.ylim(0, 15 )
    ax.axvline(color='red') 
    ax.set_xlabel('Energy (eV)') 
    #ax.set_ylabel('DOS')
    #plt.title(figname)
    #plt.tight_layout()
    #plt.savefig(figname)
    #plt.show()
    #plt.close() #plt.show() have a function of close. plt.close() means close the figure.

def plot_pdos(fname, figname, efermi, xlim=(-10, 10), ylim=(None,None)):
    fig, ax=plt.subplots()
    plot_pdos_ax(fname, efermi, ax=ax, xlim=xlim, ylim=ylim)
    plt.title(figname)
    plt.tight_layout()
    plt.savefig(figname)
    #plt.show()
    plt.close() 

def plot_total_dos(fname, efermi, xlim=(-6,6), ylime=(0,60)):
    data=np.loadtxt(fname)
    x=data[:,0]
    y=data[:,1]
    plt.plot(x-efermi,y)
    plt.axvline(0,color='red')
    plt.xlabel('$E-E_f$ (eV)')
    plt.ylabel('DOS')
    plt.show()
    plt.savefig('total_dos.png')

#core function
def gen_pdos_figure(pdos_fname, iatom, n, l, m, xlim=(-10, 10), ylim=(None,None),output_path='./'):
    outfile, efermi = get_pdos_data(pdos_fname, iatom, n, l, m)
    figname=os.path.join(output_path,f"pdos_{iatom}_{n}{lmdict[l][m]}.png")
    plot_pdos(fname=outfile, figname=figname,efermi=efermi, xlim=xlim, ylim=ylim)


def plot_layer_pdos(pdos_fname, figname, iatoms, n, l, m, xlim=(-10, 10), ylim=(None,None)):
    natoms=len(iatoms)
    fig, axes=plt.subplots(natoms,1, sharex=True)
    for i, iatom in enumerate(iatoms):
        outfile, efermi=get_pdos_data(pdos_fname, iatom, n, l, m)
        plot_pdos_ax(outfile, efermi, ax=axes[i], conv_n=5, xlim=xlim, ylim=ylim)
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(figname)

    plt.show()

