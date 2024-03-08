import numpy as np
from aeimputer import AEImputer
X = [[1,2,3],[2,np.nan,4],[np.nan,5,6]]
imputer = AEImputer()
X_imputed = imputer.fit_transform(X, verbose=True)
print(X_imputed)