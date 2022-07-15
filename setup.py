#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='pyDFTutils',
    version='0.1.4',
    description='utils for DFT/TB calculation',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy',
                      'ase', 'spglib',  'phonopy', 'pathos'],
    scripts=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        #'License :: OSI Approved :: BSD License',
        #'License :: OSI Approved :: GPLv3 License',
    ])
