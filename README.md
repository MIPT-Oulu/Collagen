# Collagen: Deep Learning Framework for Reproducible Experiments
[![Build Status](https://travis-ci.com/MIPT-Oulu/Collagen.svg?branch=master)](https://travis-ci.com/MIPT-Oulu/Collagen)

## About

## Install
You can get the most stable version from pip:

```
pip install collagen
```
The fresh-most release of the framework can also be installed from the master branch.

Collagen supports distributed training with mixed precision out-of-the box. Consider using APEX to
use this feature:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
```
