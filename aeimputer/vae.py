import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims, latent_dim):
    super(Encoder, self).__init__()
    self.encoder_layers = nn.ModuleList()
    for i in range(n_layers):
      if i == 0:
        input_dim = in_features
      else:
        input_dim = hidden_dims[i-1]
      output_dim = hidden_dims[i]
      linear_layer = nn.Linear(input_dim, output_dim)
      self.encoder_layers.append(linear_layer)
    # The last layer of the encoder outputs two vectors: mean and log variance of the latent distribution
    self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
    self.log_var_layer = nn.Linear(hidden_dims[-1], latent_dim)
  
  def forward(self, x):
    for layer in self.encoder_layers:
      x = layer(x)
      x = F.relu(x)
    mean = self.mean_layer(x)
    log_var = self.log_var_layer(x)
    return mean, log_var


class Decoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims, latent_dim):
    super(Decoder, self).__init__()
    self.decoder_layers = nn.ModuleList()
    for i in reversed(range(n_layers)):
      if i == 0:
        output_dim = in_features
      else:
        output_dim = hidden_dims[i-1]
      input_dim = hidden_dims[i]
      linear_layer = nn.Linear(input_dim, output_dim)
      self.decoder_layers.append(linear_layer)
      
    self.latent_layer = nn.Linear(latent_dim, hidden_dims[-1])
  
  def forward(self, z):
    x = self.latent_layer(z)
    x = F.relu(x)
    for layer in self.decoder_layers:
      x = layer(x)
      x = F.relu(x)
    return x


class VariationalAutoEncoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims, latent_dim):
    super(VariationalAutoEncoder, self).__init__()
    self.encoder = Encoder(in_features, n_layers, hidden_dims, latent_dim)
    self.decoder = Decoder(in_features, n_layers, hidden_dims, latent_dim)
  
  def forward(self, x):
    # Encode the input data into mean and log variance vectors
    mean, log_var = self.encoder(x)
    # Sample a latent vector from the normal distribution defined by mean and log variance
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mean + eps * std
    # Decode the latent vector into a reconstruction of the input data
    x_recon = self.decoder(z)
    return x_recon, mean, log_var