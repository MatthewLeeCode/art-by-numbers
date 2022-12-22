# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="art-by-numbers",
    version="0.0.1",
    description="Art-by-Numbers is a Python library for creating craft projects using numbered sections for painting, coloring, or other art projects.",
    license="MIT",
    author="Matthew Lee",
    packages=find_packages(),
    install_requires=[],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ]
)
