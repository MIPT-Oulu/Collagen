#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = ('numpy', 'opencv-python', 'solt')

setup_requirements = ()

test_requirements = ('pytest',)

description = """Deep Learning framework for reproducible science. 
Supports all modern practices for training deep neural networks. From Finland with love.
"""

setup(
    author="Aleksei Tiulpin, Egor Panfilov, Hoang Nguyen",
    author_email='aleksei.tiulpin@oulu.fi, egor.panfilov@oulu.fi, huy.nguyen@oulu.fi',
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
    name='solt',
    packages=find_packages(include=['solt']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MIPT-Oulu/collagen',
    version='0.0.1',
    zip_safe=False,
)
