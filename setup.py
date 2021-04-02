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

    # Used by CI to build nightly packages. Users should never use it.
    # To build a nightly wheel, run:
    # BUILD_NIGHTLY=1 python setup.py sdist
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%Y%m%d")
        # pip can perform proper comparison for ".post" suffix,
        # i.e., "1.1.post1234" >= "1.1"
        version = version + ".post" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))

    return version


def get_name():
    name = "pytorchvideo"
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        name += "-nightly"
    return name


setup(
    name=get_name(),
    version=get_version(),
    license="Apache 2.0",
    author="Facebook AI",
    url="https://github.com/facebookresearch/pytorchvideo",
    description="A video research library providing pytorch-based components.",
    python_requires=">=3.7",
    install_requires=[
        "fvcore>=0.1.4",
        "av",
        "parameterized",
        "iopath",
    ],
    extras_require={
        "test": ["coverage", "pytest", "opencv-python"],
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
        ],
    },
    packages=find_packages(exclude=("scripts", "tests")),
)
