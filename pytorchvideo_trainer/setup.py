#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup


setup(
    name="pytorchvideo_trainer",
    version="0.0.1",
    license="Apache 2.0",
    author="Facebook AI",
    url="https://github.com/facebookresearch/pytorchvideo",
    description="PyTorch-Lightning trainer powering PyTorchVideo models.",
    python_requires=">=3.8",
    install_requires=[
        "submitit",
        "pytorchvideo>=0.1.5",
    ],
    extras_require={
        "test": ["coverage", "pytest", "opencv-python"],
        "dev": [
            "opencv-python",
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
        "opencv-python": [
            "opencv-python",
        ],
    },
    packages=find_packages(exclude=("scripts", "tests")),
)
