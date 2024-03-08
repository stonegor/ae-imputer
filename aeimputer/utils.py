import torch
import pandas as pd
import numpy as np

def linear_generator(start, end, n_steps: int, return_int = True):
  if n_steps > 0:
    step = (end - start) / (n_steps-1)
    current = start
    for _ in range(n_steps):
      if return_int:
        yield round(current)
      else:
        yield current
      current += step
  else:
    raise ValueError("n must be positive")

def format_input(X, missing_values):
  if isinstance(X, list):
      X = np.array(X)
  elif isinstance(X, pd.DataFrame):
      X = np.array(X.values)
  if isinstance(X, np.ndarray):
      X = X.copy()
  else:
      raise TypeError("Expected X to be np.ndarray, list or pd.DataFrame.")
      
  if len(X.shape) != 2:
      raise ValueError(f"Expected X to be of shape (batch, features), got {X.shape} instead.")

  X = X.astype(np.float32)

  X[X == missing_values] = np.nan  

  return X 