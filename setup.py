#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = ('numpy', 'opencv-python', 'solt==0.1.8', 'pyyaml',
                'torch>=1.3.1', 'tqdm', 'scikit-learn', 'tensorboard', 'dill', 'matplotlib',
                'pandas', 'pretrainedmodels', 'pillow==6.1',
                'segmentation-models-pytorch')

setup_requirements = ()

test_requirements = ('pytest',)

description = """Deep Learning framework for reproducible science. From Finland with love."""

setup(
    author="Hoang Nguyen, Abu Mohammed Raisuddin, Aleksei Tiulpin",
    author_email='huy.nguyen@oulu.fi, abu.raisuddin@oulu.fi, aleksei.tiulpin@oulu.fi,',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    description="Deep Learning Framework for Reproducible Science",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='data augmentations, deeep learning',
    name='collagen',
    packages=find_packages(include=['collagen']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MIPT-Oulu/Collagen',
    version='0.0.1',
    zip_safe=False,
)
