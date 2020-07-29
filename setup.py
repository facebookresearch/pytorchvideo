#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="pytorchvideo",
    version="0.1",
    author="Facebook AI",
    url="unknown",  # https://github.com/facebookresearch/pytorchvideo
    description="",
    python_requires=">=3.6",
    install_requires=["fvcore", "torch", "torchvision"],
    packages=find_packages(exclude=("scripts", "tests")),
)
