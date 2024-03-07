from .ae import AutoEncoder
from .vae import VariationalAutoEncoder
from .utils import linear_generator
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# TODO: Categorical variables support, early stoping

class AEImputer():
    def __init__(self, missing_values = np.nan, n_layers = 3, hidden_dims = None, latent_dim_percentage = 'auto') -> None:
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.latent_dim_percentage = latent_dim_percentage
        self.missing_values = missing_values

    def fit(self, X, max_epochs = 1000, lr = 1e-3, device = 'cpu', batch_size = 32, verbose = False):

        if device == 'cuda' and not torch.cuda.is_available():
           self.device = 'cpu'
           warnings.warn("device = 'cuda' is specified, but no avaliable cuda devices were found. switching to cpu. (torch.cuda.is_available() is False)")
        
        
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = np.array(X.values)
        elif not isinstance(X, np.ndarray):
            raise TypeError("Excpeceted X to be np.ndarray, list or pd.DataFrame.")
        
        if len(X.shape) != 2:
            raise ValueError(f"Excpeceted X to be of shape (batch, features), got {X.shape} instead.")
        
        X = X.astype(np.float32)
        
        X[X == self.missing_values] = np.nan   
        
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
        dataset = TensorDataset(torch.tensor(X[~incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.in_features = X.shape[1]
        
        
        
        # if no hidden_dims is specified, reduce layer dimensionality linearly from in_features to a fraction of in_features defined by latent_dim_percentage
        
        
        if self.latent_dim_percentage == 'auto':
            latent_dim = int(self.in_features**0.75)
        elif not isinstance(self.latent_dim_percentage, float):
            raise TypeError("Expected latent_dim_percentage to be of type float or 'auto'")
        
        if self.hidden_dims == None:          
            self.hidden_dims = list(linear_generator(self.in_features, latent_dim, n_steps = self.n_layers + 1))[1:]
        
        self.model = AutoEncoder(self.in_features, self.n_layers, self.hidden_dims).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)        
        criterion = nn.MSELoss() 
        
        # AutoEncoder training loop
        for epoch in range(max_epochs):
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
            self.tolerance = running_loss/len(dataloader)
        return self
        
    def transform(self, X, batch_size = 32, max_iters = 100, verbose = False):
        
        self.model.eval()
        
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = np.array(X.values)
        elif not isinstance(X, np.ndarray):
            raise TypeError("Excpeceted X to be np.ndarray, list or pd.DataFrame.")
        
        if len(X.shape) != 2:
            raise ValueError(f"Excpeceted X to be of shape (batch, features), got {X.shape} instead.")
        
        X = X.astype(np.float32)
        
        X[X == self.missing_values] = np.nan   
        
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
        dataset = TensorDataset(torch.tensor(X[incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.MSELoss() 
        
        imputed_batches = []
                
        for data in dataloader:
            batch = data[0]
            
            nan_mask = torch.isnan(batch)
            
            # Replace missing values in data with random values;
            batch[nan_mask] = torch.randn(batch[nan_mask].shape) 
            

            for epoch in range(max_iters):
                
                # Input the data to the trained VAE
                _, imputed_batch = self.model(batch) 
                
                # Replace the missing values with the reconstructed values, leaving the observed values unchanged;
                batch[nan_mask] = imputed_batch[nan_mask]  
              
                # Compute the reconstruction error of the observed values;       
                reconstruction_loss = criterion(imputed_batch, batch)     
                
                # If the reconstruction error is below a specified tolerance, end.
                if reconstruction_loss < self.tolerance: 
                    if verbose:
                        print(f"reconstruction_loss crossed the threshold of {self.tolerance} at the {epoch} epoch.")
                    break
                
            imputed_batches.append(batch)

        imputed_batches = torch.vstack(imputed_batches).detach().numpy()
          
        X[incomplete_rows_mask] = imputed_batches
        return X
    
    
    
    def fit_transform(self, X, max_epochs = 1e3, max_iters = 1e2, lr = 1e-3, device = 'cpu', batch_size = 32, verbose = False):
        self.fit(X, max_epochs=max_epochs, lr=lr, device=device, batch_size=batch_size, verbose=verbose)
        return self.transform(X, max_iters=max_iters, batch_size=batch_size, verbose=verbose)
