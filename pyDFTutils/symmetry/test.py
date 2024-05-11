from ase import Atoms

def test():
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
    m=atoms.get_initial_magnetic_moments()
    print(m)
    return atoms

if __name__ == '__main__':
    test()

