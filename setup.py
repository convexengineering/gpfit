"""Standard Python setup script for gpfit"""
from setuptools import setup

LICENSE = """The MIT License (MIT)

Copyright (c) 2021 Convex Engineering Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

setup(
    name="gpfit",
    description="Package for fitting geometric programming models to data",
    author="Convex Engineering Group",
    author_email="gpkit@mit.edu",
    url="https://github.com/convexengineering/gpfit",
    python_requires=">=3.4",
    install_requires=[
        "numpy",
        "scipy",
        "gpkit",
        "matplotlib>=3.3.2",
        "pytest>=6.2.5",
        "pytest-mpl>=0.13",
    ],
    version="0.2.0",
    packages=["gpfit", "gpfit.maths", "gpfit.tests", "gpfit.tests.baseline",
              "gpfit.xfoil"],
    include_package_data=True,
    license=LICENSE,
)
