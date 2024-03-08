import numpy as np
from aeimputer import VAEImputer
X = [[1,2,3],[2,np.nan,4],[np.nan,5,6],[np.nan,2,3],[2,3,4],[4,5,6]]
imputer = VAEImputer(n_layers=10, init_nan='noise')
X_imputed = imputer.fit_transform(X, verbose=True)
print(X_imputed)