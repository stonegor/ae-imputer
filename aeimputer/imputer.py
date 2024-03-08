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
# ELBO

class _BaseImputer:
    def __init__(self, missing_values):
        self.missing_values = missing_values

    def fit(self, X, verbose = False):
        raise NotImplementedError("Abstract method")

    def transform(self, X, verbose = False) -> np.ndarray:
        raise NotImplementedError("Abstract method")

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def _format_input(self, X):
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
        
        X[X == self.missing_values] = np.nan  
        
        return X 


class AEImputer(_BaseImputer):
    def __init__(self, missing_values = np.nan, n_layers = 3, hidden_dims = None, latent_dim_percentage = 'auto',max_epochs = 1000, max_impute_iters = 100, lr = 1e-3, device = 'cpu', batch_size = 32):
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.latent_dim_percentage = latent_dim_percentage
        self.max_epochs = max_epochs
        self.max_impute_iters = max_impute_iters
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        super().__init__(missing_values)

        if self.device == 'cuda' and not torch.cuda.is_available():
           self.device = 'cpu'
           warnings.warn("device = 'cuda' is specified, but no avaliable cuda devices were found. switching to cpu. (torch.cuda.is_available() is False)")
        

    def fit(self, X, verbose = False):

        X = self._format_input(X)
        
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
      
        self.in_features = X.shape[1]
        incomplete_rows_mask = np.isnan(X).any(axis=1)

        dataset = TensorDataset(torch.tensor(X[~incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                   
        # if no hidden_dims is specified, reduce layer dimensionality linearly from in_features to a fraction of in_features defined by latent_dim_percentage
        if self.latent_dim_percentage == 'auto':
            latent_dim = int(self.in_features**0.75)
        elif not isinstance(self.latent_dim_percentage, float):
            raise TypeError("Expected latent_dim_percentage to be of type float or 'auto'")
        
        if self.hidden_dims == None:          
            self.hidden_dims = list(linear_generator(self.in_features, latent_dim, n_steps = self.n_layers + 1))[1:]
        
        self.model = AutoEncoder(self.in_features, self.n_layers, self.hidden_dims).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)        
        criterion = nn.MSELoss() 
        
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
            self.tolerance = running_loss/len(dataloader)
        
    def transform(self, X, verbose = False):
        
        X = self._format_input(X)   
        
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
        #TODO: Get nan mask for all of the X and pass it along with X to the dataset; This will allow for efficient starting imputation (mb start not with random, but with median values)
        
        self.model.eval()

        dataset = TensorDataset(torch.tensor(X[incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.MSELoss() 
        
        imputed_batches = []
                
        for data in dataloader:
            batch = data[0]
            
            nan_mask = torch.isnan(batch)
            
            # Replace missing values in data with random values;
            batch[nan_mask] = torch.randn(batch[nan_mask].shape) 
            

            for epoch in range(self.max_impute_iters):
                
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
        
        X[incomplete_rows_mask] = torch.vstack(imputed_batches).detach().numpy()
        return X

