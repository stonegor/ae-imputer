from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ae-imputer",
    version="0.0.10",
    description="An id generator that generated various types and lengths ids",
    package_dir={"": "aeimputer"},
    packages=find_packages(where="aeimputer"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stonegor/ae-imputer",
    author="stonegor",
    author_email="donskikh.egor@yandex.ru",
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
    python_requires=">=3.8.0",
)