import os
def move_wannier_files(parent=None, orig='.', dest='Wannier'):
    fnames=['POSCAR','OUTCAR', 'basis.txt', 'wannier90.*']
    if parent is not None:
        orig=os.path.join(parent, orig)
        dest=os.path.join(parent, dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
    for fname in fnames:
        os.system('cp %s %s'%(os.path.join(orig, fname), dest ))

if __name__=="__main__":
    move_wannier_files(parent=None, orig='.', dest='Wannier')
