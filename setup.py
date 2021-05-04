#!/usr/bin/env python

import os
import re
import setuptools

pkg_name = "dem"

# Get long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Requirements
with open("requirements.txt", "r") as fh:
    install_requires = fh.read()

# Get version from hdviz/__init__.py
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, pkg_name, "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

setuptools.setup(
    name=pkg_name,
    version=version,
    author="Juho Timonen",
    author_email="juho.timonen@iki.fi",
    description="Generative flow models.",
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtimonen/diffeq_match",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=["pip>=19.0.3"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
