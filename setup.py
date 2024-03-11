from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ae-imputer",
    version="0.0.1",
    description="a python package used for missing data imputation via autoencoders",
    packages=["aeimputer"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stonegor/ae-imputer",
    author="stonegor",
    author_email="sstonegor@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["scikit-learn >= 0.22",
                      "numpy >= 1.20.0",
                      "pandas",
                      "torch"],
    extras_require={
      "dev" : ["twine>=4.0.2"],  
    },
    python_requires=">=3.8.0",
)