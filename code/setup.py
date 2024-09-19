from setuptools import find_packages, setup

setup(
    name="utils",
    version="0.0.1",
    description="utility for ppga project",
    package_dir={"": "./"},
    packages=find_packages(where="./"),
    author="Federico Bustaffa",
)
