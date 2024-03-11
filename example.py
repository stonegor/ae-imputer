import numpy as np
from aeimputer import AEImputer
X = [[1,2,3],[2,np.nan,4],[np.nan,5,6],[np.nan,2,3],[2,3,4],[4,5,6]]
imputer = AEImputer(n_layers=5)
X_imputed = imputer.fit_transform(X)
print(X_imputed)