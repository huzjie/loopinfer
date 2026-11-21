# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="loopinfer",
    version="1.0.0",
    description="Looped Transformer (Looped MoE) inference runtime",
    packages=find_packages(include=["loopinfer", "loopinfer.*"]),
    python_requires=">=3.8",
)
