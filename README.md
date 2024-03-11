![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
# ae-imputer
ae-imputer is a python package used for missing data imputation via autoencoders.

As of now, only numerical values are supported for imputation.

The method used is based on the paper:

[John T. McCoy, Steve Kroon, Lidia Auret: Variational Autoencoders for Missing Data Imputation with Application to a Simulated Milling Circuit, IFAC-PapersOnLine, 2018](https://www.sciencedirect.com/science/article/pii/S2405896318320949)

## Installing

Note that  **ae-imputer** uses **PyTorch** for all of its underlying AutoEncoder implementations.

Requirements:

* Python 3.8 or greater
* numpy
* scikit-learn
* pytorch

```bash
pip install ae-imputer
```

## Usage

The ae-imputer package is designed to match sklearn imputers calling API. 

```python
import numpy as np
from aeimputer import AEImputer

X = [[1,2,3],[2,np.nan,4],[np.nan,5,6],[np.nan,2,3],[2,3,4],[4,5,6]]
imputer = AEImputer(n_layers=5)

X_imputed = imputer.fit_transform(X)
```
It is recommended to normalize your data before fitting and imputation.
Unlike the example above, AEImputer is meant to be used with much larger amounts of data,
in order to properly utilyze its capabilities.

There are a number of parameters that can be set for the AEImputer class; the
major ones are as follows:

 -  ``model_type`` : 'variational' or 'vanilla', default='variational;
        Type of AutoEncoder architecture to use.

 -  ``n_layers`` : int, default=3
        The number of layers in the AutoEncoder network.

    ``hidden_dims`` : list of int, default=None
        The number of neurons for each hidden layer in the AutoEncoder network. If None, will be 
        determined automatically.`hidden_dims`` : list of int, default=None
        The number of neurons for each hidden layer in the AutoEncoder network. If None, will be 
        determined automatically.

    ``preimpute_at_train`` : bool, default = False
        AEImputer uses only complete rows of data during fitting by default.
        If set True the missing values will be imputed with 'preimpute_strategy' before training.
        Advised, if the fraction of missing rows is significant
        
    ``max_epochs`` : int, default=1000
        The maximum number of epochs to train the AutoEncoder.

    ``lr`` : float, default=1e-3
        The learning rate for the optimizer during training.


```bibtex
@article{MCCOY2018141,
    title = {Variational Autoencoders for Missing Data Imputation with Application to a Simulated Milling Circuit},
    journal = {IFAC-PapersOnLine},
    volume = {51},
    number = {21},
    pages = {141-146},
    year = {2018},
    note = {5th IFAC Workshop on Mining, Mineral and Metal Processing MMM 2018},
    issn = {2405-8963},
    doi = {https://doi.org/10.1016/j.ifacol.2018.09.406},
    url = {https://www.sciencedirect.com/science/article/pii/S2405896318320949},
    author = {John T. McCoy and Steve Kroon and Lidia Auret},
}
```
