import os


class PPFinder():
    def __init__(self):
        pass

    def get_pp_path(self, element, xc, label, rel):
        pass

class DojoFinder():
    def __init__(self, path=None):
        if path is None:
            self.path = os.environ['DOJO_PATH']
        else:
            self.path = path

    def get_pp_path(self,
                    xc : str,
                    typ='NC',
                    rel='sr',
                    version='04',
                    accuracy='standard',
                    fmt='psml'):
        typ=typ.lower()
        xc=xc.lower()
        dirname = os.path.join(self.path, f"{typ}-{rel}-{version}_{xc}_{accuracy}_{fmt}")
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"File Not found: {dirname}")
        return dirname

    def get_pp_fname(self,
                    element,
                    xc : str,
                    typ='NC',
                    rel='sr',
                    version='04',
                    accuracy='standard',
                    fmt='psml'):
        fname = os.path.join(self.get_pp_path(xc=xc, typ=typ, rel=rel, version=version, accuracy=accuracy, fmt=fmt), f"{element}.{fmt}")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File Not found: {fname}")
        return fname



def test():
    finder=DojoFinder(path=os.path.expanduser('~/projects/pp/dojo'))
    fname=finder.get_pp_path(element='Sr', xc='pbesol')
    fname=finder.get_pp_path(element='Srg', xc='pbe')

if __name__=='__main__':
    test()
