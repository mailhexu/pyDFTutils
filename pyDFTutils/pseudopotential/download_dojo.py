import os

def download_dojo_dirname(dirname):
    fname=f"{dirname}.tgz"
    url = f"http://www.pseudo-dojo.org/pseudos/{fname}"
    print(url)
    os.system(f'wget {url}')
    if not os.path.exists(dirname):
        os.makedirs(f'{dirname}')
    os.system(f'mv {fname} {dirname}')
    cwd = os.getcwd()
    os.chdir(dirname)
    os.system(f'tar xf {fname}')
    os.chdir(cwd)


def download_dojo_04():
    typ= 'nc'
    ver = '04'
    for rel in ['fr', 'sr']:
        for ver in ['04']:
            for xc in ['pbesol', 'pbe', 'pw']:
                for acc in ['standard', 'stringent']:
                    for fmt in ['psml', 'psp8']:
                        dirname=f"{typ}-{rel}-{ver}_{xc}_{acc}_{fmt}"
                        download_dojo_dirname(dirname)

def download_dojo_3plus():
    typ= 'nc'
    ver = '04'
    for rel in ['sr']:
        for ver in ['04-3plus']:
            for xc in ['pbesol', 'pbe']:
                for acc in ['standard']:
                    for fmt in ['psml', 'psp8']:
                        dirname=f"{typ}-{rel}-{ver}_{xc}_{acc}_{fmt}"
                        download_dojo_dirname(dirname)

def download_dojo():
    download_dojo_04()
    download_dojo_3plus()

if __name__=='__main__':
    download_dojo()
