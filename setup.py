#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='snap-tools',
    version='0.0.1',
    description='Tools for converting Keras models to other specs.',
    author='Triage Technologies',
    author_email='steven@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[],
    install_requires=[
        'tensorflow',
        'keras',
        'h5py',
        'pillow',
        #'coremltools' <- NOTE!: coremltools is currently Python 2.7 only
    ]
)
