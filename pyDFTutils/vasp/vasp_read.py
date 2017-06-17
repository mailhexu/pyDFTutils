
from ase.io import read

def born_read(fname='OUTCAR',poscar='POSCAR'):
    atoms=read(poscar)
    natoms=len(atoms.get_chemical_symbols())

    with open(fname) as myfile:
        lines=myfile.readlines()
    print(len(lines))
    bec=[[] for i in range(natoms)]

    bornsection=False
    counter=-3
    for line in lines:
        if line.find('BORN EFFECTIVE CHARGE')!= -1 and line.find('cummulative')==-1:
            bornsection=True
        if bornsection:
            counter=counter+1
            if 0<=counter<natoms:
                b=[float(x) for x in line.strip().split()[3:6]]
                bec[counter].append(b)
            if counter==natoms:
                bornsection=False
                counter=-3
    return bec

if __name__ == '__main__':
    print(born_read())
