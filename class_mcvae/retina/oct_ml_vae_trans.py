import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from IPython import embed
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, output_channels, deep, width, height):
        super(UnFlatten, self).__init__()
        self.output_channels = output_channels
        self.width = width
        self.height = height
        self.deep = deep
    def forward(self, input):
        return input.view(input.size(0), self.output_channels, self.deep,  self.width, self.height)

class VAE_oct(nn.Module):

    def __init__(self, input_dim_oct, latent_dim_oct, n_channels_oct, kernel_size_oct, padding_oct, stride_oct):

       super(VAE_oct, self).__init__()
       self.latent_dim = latent_dim_oct
       self.n_channels = n_channels_oct
       self.kernel_size = kernel_size_oct
       self.padding = padding_oct
       self.stride = stride_oct
       self.input_dim = input_dim_oct[-3:]

       enc_dim = [self.input_dim]
       for j in range(len(self.n_channels)-1):
           enc_dim.append( [int( (enc_dim[j][i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] +1) for i in (0,1,2)])

       enc_nested_module_list = [[
         nn.Conv3d(self.n_channels[i], self.n_channels[i+1], self.kernel_size, self.stride, self.padding),
         nn.BatchNorm3d(self.n_channels[i+1]),
         nn.ReLU() ] for i in range(len(self.n_channels)-1)
       ]

       self.encoder_layers = nn.ModuleList([item for sublist in enc_nested_module_list for item in sublist])
       self.encoder_layers.append(Flatten())
       self.fc1 = nn.Linear(self.n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1] * enc_dim[-1][2], self.latent_dim)
       self.fc2 = nn.Linear(self.n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1] * enc_dim[-1][2], self.latent_dim)

       self.fc3 = nn.Linear(self.latent_dim, self.n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1] * enc_dim[-1][2])

       # These values were calculated ad hoc for the input dimensions (450, 312)
       o_paddings_decoder = []
       dec_dim = [enc_dim[-1]]
       for j in range(len(self.n_channels)-1):
         for padding in [(0,0,0), (0,1,0), (0,0,1), (1,0,0), (1,1,0), (0,1,1), (1,0,1), (1,1,1)]:  # [(0,0), (0,1), (1,0), (1,1)]:
           dim = [int( (dec_dim[j][i]-1) * self.stride[i] - 2 * self.padding[i] + self.kernel_size[i] + padding[i]) for i in (0,1,2)]
           if tuple(dim) == tuple(enc_dim[-j-2]):
             o_paddings_decoder.append(padding)
             dec_dim.append(dim)
             break

       dec_nested_module_list = []
       for i in range(len(self.n_channels)-1, 1, -1):
         dec_nested_module_list.append([
           nn.ConvTranspose3d(self.n_channels[i], self.n_channels[i-1], self.kernel_size, self.stride, self.padding, o_paddings_decoder[-i]),
           nn.BatchNorm3d(self.n_channels[i-1]),
           nn.LeakyReLU(0.2),
         ])


       dec_nested_module_list_2 = [
          nn.ConvTranspose3d(self.n_channels[1], self.n_channels[0], self.kernel_size, self.stride, self.padding, o_paddings_decoder[-1]),
          nn.BatchNorm3d(self.n_channels[0]),
          nn.Sigmoid(),
          ]

       dec_nested_module_list.append(dec_nested_module_list_2)
       self.decoder_layers = ModuleList([UnFlatten(self.n_channels[-1], dec_dim[0][0], dec_dim[0][1], dec_dim[0][2])])
       self.decoder_layers.extend([item for sublist in dec_nested_module_list for item in sublist])


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()) #.to(mu.device)
        esp = esp.type_as(mu)
        z = mu + std * esp
        return z  # .to(device)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encoder(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decoder(self, z):
        h = self.fc3(z)
        for layer in self.decoder_layers:
            h = layer(h)
        return h


    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x1 = self.decoder(z)
        return x1, z, mu, logvar

class classifier_oct(nn.Module):

    def __init__(self, latent_dim_oct, n_classes_oct, nhead_oct, num_encoder_layers_oct):

       super(classifier_oct, self).__init__()
       self.latent_dim = latent_dim_oct
       self.n_classes = n_classes_oct
       self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim_oct, nhead=nhead_oct),
            num_layers=num_encoder_layers_oct)
       self.fc = nn.Linear(latent_dim_oct, n_classes_oct-1)
       self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class vae_classifier_oct(nn.Module):
    def __init__(self, VAE_oct, classifier_oct):
        super(vae_classifier_oct, self).__init__()
        self.vae_oct = VAE_oct
        self.classifier_oct = classifier_oct

    def forward(self, x):
        x1, z, mu, logvar = self.vae_oct(x)
        prediction =self.classifier_oct(z)
        return prediction, x1, z, mu, logvar
