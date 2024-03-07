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

