#!/usr/bin/env python
import argparse
from ase.io import read, write
from pyDFTutils.perovskite.octahedra import write_all_octahedra_info


def main():
    parser=argparse.ArgumentParser(description='write all the octahedra info')
    parser.add_argument('-f','--filename',type=str,help='POSCAR filename',default='POSCAR')
    parser.add_argument('-o','--output',help='output filename',default='octa_info.csv')
    parser.add_argument('-c','--center',nargs='+',type=str,help='octahedra symbol list',default='Mn')
    parser.add_argument('-v','--vertex',help='vertex symbol', default='O')
    parser.add_argument('-d','--distance',type=float,help='max distance', default=3.0)
    parser.add_argument('-a','--axis_type',help='axis type: None or rotate45_xy',default=None)
    parser.add_argument('-x','--xvec',help='x vector',nargs='+',type=float,default=(1,0,0))
    parser.add_argument('-y','--yvec',help='y vector',nargs='+',type=float,default=(0,1,0))
    parser.add_argument('-z','--zvec',help='z vector',nargs='+',type=float,default=(0,0,1))
    args=parser.parse_args()
    atoms=read(args.filename)

    write_all_octahedra_info(atoms,tuple(args.center), args.vertex, args.distance, output=args.output,axis_type=args.axis_type,x=args.xvec,y=args.yvec,z=args.zvec)



if __name__=='__main__':
    main()
