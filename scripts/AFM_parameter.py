#!/usr/bin/env python
from pyDFTutils.perovskite.octahedra import even_or_odd_path,symnum_to_sym
from ase.io.vasp import read_vasp
import argparse
def test():
    parser=argparse.ArgumentParser(description='Decide AFM structure 1 -1')
    parser.add_argument('-f','--filename',type=str,help='POSCAR filename',default='POSCAR.vasp')
    parser.add_argument('-n','--nodes',type=str,help='symbol of element on the node',nargs='+')
    parser.add_argument('-o','--origin',type=str,help='From which symbol_number')
    parser.add_argument('--xmin',type=float,help='xmin',default=0.5)
    parser.add_argument('--xmax',type=float,help='xmax',default=4.8)
    args=parser.parse_args()

    #plot_all_pdos(element_types=args.element,filename=args.filename,ispin=args.ispin,ymin=args.ymin,ymax=args.ymax,xmin=args.xmin,xmax=args.xmax,output_dir=args.output)

    atoms=read_vasp(args.filename)
    symnums,vals= even_or_odd_path(atoms,args.origin,args.nodes,first_neighbor_min=args.xmin,first_neighbor_max=args.xmax)
    for sym in args.nodes:
        vs=[]
        for i,(s,v) in enumerate(zip(symnums,vals)):
            if symnum_to_sym(s)==sym:
                vs.append(v)
        print('%s: %s'%(sym,vs))


if __name__ == '__main__':
    test()
