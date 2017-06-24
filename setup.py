#!/usr/bin/env python
from setuptools import setup

setup(
    name='pyDFTutils',
    version='0.1',
    description='utils for DFT/TB calculation',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=['pyDFTutils'],
    install_requires=['numpy','matplotlib','scipy','ase', 'spglib', 'pythtb', 'phonopy'],
    scripts=['scripts/plotldos.py','scripts/plotpdos.py','scripts/AFM_parameter.py','scripts/view_pos_info.py',
        'scripts/run_wannier.py'
        
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GPLv3 license',
    ])
