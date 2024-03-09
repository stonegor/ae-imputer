import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims):
    super(Encoder, self).__init__()
    
    self.activation = nn.LeakyReLU()

    self.encoder_layers = nn.ModuleList()
    for i in range(n_layers):
      if i == 0:
        input_dim = in_features
      else:
        input_dim = hidden_dims[i-1]
      output_dim = hidden_dims[i] 
      linear_layer = nn.Linear(input_dim, output_dim)
      self.encoder_layers.append(linear_layer)
  
  def forward(self, x):
    for layer in self.encoder_layers:
      x = layer(x)
      x = self.activation(x)
    return x


class Decoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims):
    super(Decoder, self).__init__()

    self.activation= nn.LeakyReLU()

    self.decoder_layers = nn.ModuleList()
    for i in reversed(range(n_layers)):
      if i == 0:
        output_dim = in_features
      else:
        output_dim = hidden_dims[i-1]
      input_dim = hidden_dims[i]
      linear_layer = nn.Linear(input_dim, output_dim)
      self.decoder_layers.append(linear_layer)
  
  def forward(self, x):
    for layer in self.decoder_layers:
      x = layer(x)
      x = self.activation(x)
    return x


class AutoEncoder(nn.Module):
  def __init__(self, in_features, n_layers, hidden_dims):
    super(AutoEncoder, self).__init__()
       
    self.encoder = Encoder(in_features, n_layers, hidden_dims)
    self.decoder = Decoder(in_features, n_layers, hidden_dims)
  
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded

