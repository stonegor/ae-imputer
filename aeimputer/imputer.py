import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
import pandas as pd

from .ae import AutoEncoder
from .vae import VariationalAutoEncoder
from .utils import linear_generator, format_input
from .early_stopper import EarlyStopper


# TODO: Categorical variables support
# Normalize input and rescale back output
# allow GPU support

class _BaseImputer:
    def __init__(self, missing_values):
        self.missing_values = missing_values

    def fit(self, X, verbose = False):
        raise NotImplementedError("Abstract method")

    def transform(self, X, verbose = False) -> np.ndarray:
        raise NotImplementedError("Abstract method")

    def fit_transform(self, X, verbose=False) -> np.ndarray:
        self.fit(X,verbose)
        return self.transform(X,verbose)
    
class AEImputer(_BaseImputer):
    def __init__(self, missing_values = np.nan, n_layers = 3, hidden_dims = None, latent_dim_percentage = 'auto',max_epochs = 1000, lr = 1e-3, patience = 10, min_delta = 0.0001, max_impute_iters = 15, init_nan = 'mean', device = 'cpu', batch_size = 32):
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.latent_dim_percentage = latent_dim_percentage
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.max_impute_iters = max_impute_iters
        self.init_nan = init_nan
        self.device = device
        self.batch_size = batch_size
        super().__init__(missing_values)

        if self.device == 'cuda' and not torch.cuda.is_available():
           self.device = 'cpu'
           warnings.warn("device = 'cuda' is specified, but no avaliable cuda devices were found. switching to cpu. (torch.cuda.is_available() is False)")
        

    def fit(self, X, verbose = False):

        X = format_input(X, self.missing_values)
        
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
      
        self.in_features = X.shape[1]
        incomplete_rows_mask = np.isnan(X).any(axis=1)

        dataset = TensorDataset(torch.tensor(X[~incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                   
        # if no hidden_dims is specified, reduce layer dimensionality linearly from in_features 
        # to a fraction of in_features defined by latent_dim_percentage

        if self.latent_dim_percentage == 'auto':
            latent_dim = int(self.in_features**0.75)
        elif not isinstance(self.latent_dim_percentage, float):
            raise TypeError("Expected latent_dim_percentage to be of type float or 'auto'")
        
        if self.hidden_dims == None:          
            self.hidden_dims = list(linear_generator(self.in_features, latent_dim, n_steps = self.n_layers + 1))[1:]
        
        self.model = AutoEncoder(self.in_features, self.n_layers, self.hidden_dims).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)        
        criterion = nn.MSELoss() 

        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

        # AutoEncoder training loop
        for epoch in range(self.max_epochs):
            running_loss = 0.0
            for data in dataloader:
                batch = data[0]
                optimizer.zero_grad()
                
                _, batch_reconstruction = self.model(batch)
                reconstruction_loss = criterion(batch_reconstruction, batch)

                reconstruction_loss.backward()
                optimizer.step()

                running_loss += reconstruction_loss.item()

            if verbose:
                print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
 
            if early_stopper.early_stop(running_loss/len(dataloader)):  
                if verbose:
                    print(f"Loss converged on {epoch} Epoch.")           
                break
        
    def transform(self, X, verbose = False):
        
        X = format_input(X, self.missing_values)   
        

        nan_mask = np.isnan(X)
        incomplete_rows_mask = nan_mask.any(axis=1)
        
        self.model.eval()

        if self.init_nan == 'noise':
            X[nan_mask] = np.random.randn(*X[nan_mask].shape) 
            
        elif self.init_nan in ('mean','median','most_frequent'):
            simple_imputer = SimpleImputer(strategy=self.init_nan)
            X = simple_imputer.fit_transform(X)

        else:
            raise TypeError(f"Expected init_nan to be in ['noise','mean','median','most_frequent'], got {self.init_nan} instead")
        
        dataset = TensorDataset(torch.tensor(X[incomplete_rows_mask]), torch.tensor(nan_mask[incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.MSELoss() 
        
        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

        imputed_batches = []

        for data in dataloader:

            batch, nan_mask = data
            
            for epoch in range(self.max_impute_iters):
                

                _, imputed_batch = self.model(batch) 
                
                # Replace the missing values with the reconstructed values, leaving the observed values unchanged;
                batch[nan_mask] = imputed_batch[nan_mask]  
                   
                reconstruction_loss = criterion(imputed_batch, batch)     
                
                if verbose:
                    print(f"Epoch {epoch+1}, Loss: {reconstruction_loss}")

                if early_stopper.early_stop(reconstruction_loss):  
                    if verbose:
                        print(f"Loss converged on {epoch} Epoch.")           
                    break
                
            imputed_batches.append(batch)
        
        X[incomplete_rows_mask] = torch.vstack(imputed_batches).detach().numpy()
        return X

