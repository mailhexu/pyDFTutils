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
    install_requires=['ase', 'spglib', 'pythtb', 'phonopy'],
    scripts=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GPLv3 license',
    ])
