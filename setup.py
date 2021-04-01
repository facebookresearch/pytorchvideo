#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

from setuptools import find_namespace_packages, find_packages, setup


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "pytorchvideo", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


setup(
    name="pytorchvideo",
    version=get_version(),
    license='Apache 2.0',
    author="Facebook AI",
    url="https://github.com/facebookresearch/pytorchvideo",
    description="A video research library providing pytorch-based components.",
    python_requires=">=3.7",
    install_requires=[
        "fvcore>=0.1.4",
        "av",
        "parameterized",
        "opencv-python",
        "iopath",
    ],
    extras_require={
        "test": ["coverage", "pytest"],
        "dev": [
            "black==20.8b1",
            "sphinx",
            "isort==4.3.21",
            "flake8==3.8.1",
            "flake8-bugbear",
            "flake8-comprehensions",
            "pre-commit",
            "nbconvert",
            "bs4",
            "autoflake==1.4",
        ]
    },
    packages=find_packages(exclude=("scripts", "tests")),
)
