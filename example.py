import numpy as np
from sklearn.datasets import make_classification
from aeimputer import AEImputer
X, _ = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42, shuffle=False, n_classes=2)
# Introduce missing values in 10% of the data
rng = np.random.RandomState(42)
mask = rng.rand(*X.shape) < 0.01
X[mask] = np.nan
imputer = AEImputer(preimpute_at_train=True)
X_imputed = imputer.fit_transform(X)
print(X_imputed)