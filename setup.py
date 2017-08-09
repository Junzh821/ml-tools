#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='ml-tools',
    version='0.0.5',
    description='Tools for common machine learning tasks using Tensorflow and Keras.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[
        'bin/keras_to_tensorflow',
    ],
    install_requires=[
        'tensorflow',
        'keras',
        'h5py',
        'pillow',
        'tensorflow-serving-client>=0.0.5',
        #'coremltools' <- NOTE!: coremltools is currently Python 2.7 only
    ]
)
