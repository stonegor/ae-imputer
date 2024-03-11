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
from .utils import linear_generator, format_input, kld_loss
from .early_stopper import EarlyStopper

# TODO: 
# Check for complete data, or zero training data
# In the case of zero training data, throw warning and impute missing with mean during training

class _BaseImputer:
    def fit(self, X, verbose = False):
        raise NotImplementedError("Abstract method")

    def transform(self, X, verbose = False) -> np.ndarray:
        raise NotImplementedError("Abstract method")

    def fit_transform(self, X, verbose=False) -> np.ndarray:
        self.fit(X,verbose)
        return self.transform(X,verbose)
    
class AEImputer(_BaseImputer):
    """
    An imputer class that uses an AutoEncoder to impute missing values in a dataset.

    This imputer assumes that the data comes in a tabular format with rows as samples
    and columns as features. It uses an AutoEncoder neural network to learn the underlying
    data distribution and predict the missing values.

    Currently does not support categorical variables.

    Parameters
    ----------
    model_type : 'variational' or 'vanilla', default='variational;
        Type of AutoEncoder architecture to use.

    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of `missing_values` will be imputed.

    n_layers : int, default=3
        The number of layers in the AutoEncoder network.

    hidden_dims : list of int, default=None
        The number of neurons for each hidden layer in the AutoEncoder network. If None, the network
        will generate a linearly decreasing number of neurons per layer until it reaches
        number of in features times `latent_dim_percentage`.

    latent_dim_percentage : float or 'auto', default='auto'
        The percentage of the input features that will be used to calculate the size of the latent space
        in the AutoEncoder network. If 'auto', it will be set to the 0.75 power of the number of features.

    max_epochs : int, default=1000
        The maximum number of epochs to train the AutoEncoder.

    lr : float, default=1e-3
        The learning rate for the optimizer during training.

    patience : int, default=10
        The number of epochs with no improvement after which training will be stopped.

    min_delta : float, default=1e-6
        The minimum change in the monitored quantity to qualify as an improvement.

    max_impute_iters : int, default=15
        The maximum number of iterations for the imputation process within the transform method.

    preimpute_strategy : {'mean', 'median', 'most_frequent', 'noise'}, default='mean'
        The strategy to initialize missing values at model inference steps. 
    
    preimpute_at_train : bool, default = False
        AEImputer uses only complete rows of data during fitting by default.
        If set True the missing values will be imputed with 'preimpute_strategy' before training.
        Advised, if the fraction of missing rows is significant

    device : {'cpu', 'cuda'}, default='cpu'
        The device to run the AutoEncoder model on. If 'cuda' is not available, it will fall back to 'cpu'.

    batch_size : int, default=32
        The size of the batches for processing the data.

    Methods
    -------
    fit(X, verbose=False)
        Fit the imputer on the input data `X`.

    transform(X, verbose=False)
        Impute all missing values in `X`.

    fit_transform(X, verbose=False)
        Fit the imputer on `X`, and then transform `X`.

    Notes
    -----
    This imputer requires that `X` is a numpy array, list or a pandas DataFrame with numerical values only.

    The imputer uses only complete rows during training, if the fraction of rows with missing data is significant,
    it is advised to set 'preimpute_at_train' = True

    It is recommended to normalize input features before training and inference.

    Examples
    --------
    >>> import numpy as np
    >>> from aeimputer import AEImputer
    >>> X = [[1,2,3],[2,np.nan,4],[np.nan,5,6],[np.nan,2,3],[2,3,4],[4,5,6]]
    >>> imputer = AEImputer(n_layers = 5)
    >>> X_imputed = imputer.fit_transform(X)
    """
    def __init__(self,model_type = 'variational', missing_values = np.nan, n_layers = 3, hidden_dims = None, latent_dim_percentage = 'auto',max_epochs = 1000, lr = 1e-3, patience = 10, min_delta = 1e-6, max_impute_iters = 15, preimpute_strategy = 'mean',preimpute_at_train = False, device = 'cpu', batch_size = 32):
        self.model_type = model_type
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.latent_dim_percentage = latent_dim_percentage
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.max_impute_iters = max_impute_iters
        self.preimpute_strategy = preimpute_strategy
        self.preimpute_at_train = preimpute_at_train
        self.device = device
        self.batch_size = batch_size
        self.missing_values = missing_values

        if self.device == 'cuda' and not torch.cuda.is_available():
           self.device = 'cpu'
           warnings.warn("device = 'cuda' is specified, but no avaliable cuda devices were found. switching to cpu. (torch.cuda.is_available() is False)")
        
        self._reconstruction_loss = nn.MSELoss()

    def fit(self, X, verbose = False):

        X = format_input(X, self.missing_values)
        
        self.in_features = X.shape[1]
        incomplete_rows_mask = np.isnan(X).any(axis=1)
        
        if self.preimpute_at_train:
            X = self._preimpute(X)    
        else:
            X = X[~incomplete_rows_mask]

        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                   
        self._model = self._build_model().to(self.device)
        
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)        

        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

        for epoch in range(self.max_epochs):
            running_loss = 0.0
            for data in dataloader:
                batch = data[0].to(self.device)
                optimizer.zero_grad()
                
                loss = self._loss(batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

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
        
        self._model.eval()

        X = self._preimpute(X)

        dataset = TensorDataset(torch.tensor(X[incomplete_rows_mask]), torch.tensor(nan_mask[incomplete_rows_mask]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.MSELoss() 
        
        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

        imputed_batches = []

        for data in dataloader:

            batch, nan_mask = data
            batch = batch.to(self.device)
            nan_mask = nan_mask.to(self.device)
            
            for epoch in range(self.max_impute_iters):
                
                
                imputed_batch = self._model(batch)[-1]
                
                # Replace the missing values with the reconstructed values, leaving the observed values unchanged;
                batch[nan_mask] = imputed_batch[nan_mask]  
                
                
                reconstruction_loss = criterion(imputed_batch, batch)     
                
                if verbose:
                    print(f"Epoch {epoch+1}, Loss: {reconstruction_loss}")

                if early_stopper.early_stop(reconstruction_loss):  
                    if verbose:
                        print(f"Loss converged on {epoch} Epoch.")           
                    break
                
            imputed_batches.append(batch.detach().cpu())
        
        X[incomplete_rows_mask] = torch.vstack(imputed_batches).numpy()
        return X
    def _preimpute(self, X):
        """
        Impute missing values with basic imputation before inference or training.
        """
        nan_mask = np.isnan(X)

        if self.preimpute_strategy == 'noise':
            X[nan_mask] = np.random.randn(*X[nan_mask].shape) 
            
        elif self.preimpute_strategy in ('mean','median','most_frequent'):
            simple_imputer = SimpleImputer(strategy=self.preimpute_strategy)
            X = simple_imputer.fit_transform(X)

        else:
            raise TypeError(f"Expected preimpute_strategy to be in ['noise','mean','median','most_frequent'], got {self.preimpute_strategy} instead")           
        
        return X
    def _build_model(self) -> nn.Module:
        """
        Parse params and build specified autoencoder model
        """

        # if no hidden_dims is specified, reduce layer dimensionality linearly from in_features 
        # to a fraction of in_features defined by latent_dim_percentage

        if self.latent_dim_percentage == 'auto':
            latent_dim = int(self.in_features**0.75)
        elif not isinstance(self.latent_dim_percentage, float):
            raise TypeError("Expected latent_dim_percentage to be of type float or 'auto'")
        
        if self.hidden_dims == None:          
            self.hidden_dims = list(linear_generator(self.in_features, latent_dim, n_steps = self.n_layers + 1))[1:]
        
        if self.model_type == 'vanilla':
           model = AutoEncoder(self.in_features, self.n_layers, self.hidden_dims)
        elif self.model_type == 'variational':
            model = VariationalAutoEncoder(self.in_features, self.n_layers, self.hidden_dims)
        else:
            raise ValueError("Expected model_type to be 'vanilla' or 'variational'")
        
        return model

    def _loss(self, X):
        if self.model_type == 'vanilla':
            _, decoded = self._model(X)
            loss = self._reconstruction_loss(X, decoded)
        if self.model_type == 'variational':
            mean, log_var, decoded = self._model(X)
            loss = self._reconstruction_loss(X, decoded) + kld_loss(mean, log_var)
        return loss
                