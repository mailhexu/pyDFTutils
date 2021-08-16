#!python
from pyDFTutils.vasp.vasp_dos import plot_all_ldos
import argparse

def plot_ldos():
    parser=argparse.ArgumentParser(description='Plot the local dos')
    parser.add_argument('-f','--filename',type=str,help='DOSCAR filename',default='DOSCAR')
    parser.add_argument('-n','--ispin',type=int,help='number os spins, 1 or 2',default=2)
    parser.add_argument('-o','--output',type=str,help='output dir')
    parser.add_argument('--xmin',type=float,help='xmin',default=-15.0)
    parser.add_argument('--xmax',type=float,help='xmax',default=5.0)
    parser.add_argument('--ymin',type=float,help='ymin',default=-2.0)
    parser.add_argument('--ymax',type=float,help='xmax',default=2.0)

    args=parser.parse_args()

    plot_all_ldos(filename=args.filename,ispin=args.ispin,ymin=args.ymin,ymax=args.ymax,xmin=args.xmin,xmax=args.xmax, has_f=False)

if __name__=='__main__':
    plot_ldos()
