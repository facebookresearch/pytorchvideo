#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="pytorchvideo",
    version="0.1",
    author="Facebook AI",
    url="unknown",  # https://github.com/facebookresearch/pytorchvideo
    description="",
    python_requires=">=3.6",
    install_requires=["fvcore", "termcolor>=1.1", "simplejson", "tqdm", "psutil"],
    extras_require={"all": ["shapely", "psutil"]},
    packages=find_packages(exclude=("configs", "tests")),
)
