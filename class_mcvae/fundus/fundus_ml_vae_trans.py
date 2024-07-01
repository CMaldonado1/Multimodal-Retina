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
    def __init__(self, output_channels, width, height):
        super(UnFlatten, self).__init__()
        self.output_channels = output_channels
        self.width = width
        self.height = height
    #def forward(self, input, size=2048):
    def forward(self, input):
        return input.view(input.size(0), self.output_channels, self.width, self.height)


class VAE_FUNDUS(nn.Module):

    def __init__(self, input_dim, latent_dim, n_channels, kernel_size, padding, stride):

       super(VAE_FUNDUS, self).__init__()
       self.latent_dim = latent_dim
       self.n_channels = n_channels
       self.kernel_size = kernel_size
       self.padding = padding
       self.stride = stride
       self.input_dim = input_dim[-2:]


       # dimensions of the feature maps       
       enc_dim = [self.input_dim]
       for j in range(len(n_channels)-1):       
           enc_dim.append( [int( (enc_dim[j][i] + 2 * self.padding[i] - self.kernel_size[i]) / self.stride[i] +1) for i in (0,1)])       

       enc_nested_module_list = [[  
         nn.Conv2d(n_channels[i], n_channels[i+1], self.kernel_size, self.stride, self.padding),
         nn.BatchNorm2d(n_channels[i+1]),
         nn.ReLU() ] for i in range(len(n_channels)-1)
       ]

       self.encoder_layers = nn.ModuleList([item for sublist in enc_nested_module_list for item in sublist])
       self.encoder_layers.append(Flatten())
       self.fc1 = nn.Linear(n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1], self.latent_dim)
       self.fc2 = nn.Linear(n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1], self.latent_dim)

       self.fc3 = nn.Linear(self.latent_dim, n_channels[-1] * enc_dim[-1][0] * enc_dim[-1][1])

       # These values were calculated ad hoc for the input dimensions (450, 312)
       o_paddings_decoder = []
       dec_dim = [enc_dim[-1]]
       for j in range(len(n_channels)-1):              
         for padding in [(0,0), (0,1), (1,0), (1,1)]:
           dim = [int( (dec_dim[j][i]-1) * self.stride[i] - 2 * self.padding[i] + self.kernel_size[i] + padding[i]) for i in (0,1)]
           if tuple(dim) == tuple(enc_dim[-j-2]):
             o_paddings_decoder.append(padding)
             dec_dim.append(dim)
             break
       
       dec_nested_module_list = []
       for i in range(len(n_channels)-1, 1, -1):
         dec_nested_module_list.append([
           # nn.ConvTranspose2d(n_channels[i], n_channels[i-1], self.kernel_size, self.stride, self.padding, o_paddings_decoder[-i-1]),
           nn.ConvTranspose2d(n_channels[i], n_channels[i-1], self.kernel_size, self.stride, self.padding, o_paddings_decoder[-i]),
           nn.BatchNorm2d(n_channels[i-1]),
           nn.LeakyReLU(0.2),
         ])
       
    
       dec_nested_module_list_2 = [
          nn.ConvTranspose2d(n_channels[1], n_channels[0], self.kernel_size, self.stride, self.padding, o_paddings_decoder[-1]),
          nn.BatchNorm2d(n_channels[0]),
          nn.Sigmoid(),
          ]

       dec_nested_module_list.append(dec_nested_module_list_2) 
       self.decoder_layers = ModuleList([UnFlatten(n_channels[-1], dec_dim[0][0], dec_dim[0][1])])
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
    
    def sample_single_latent_component(self,latent_vector, component_index, std_dev):
        device = latent_vector.device
        latent_sample = torch.randn(latent_vector.shape, device=device) * std_dev
        latent_vector[:,component_index] = latent_vector[:,component_index] + latent_sample[:, component_index]
        return latent_vector

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x1 = self.decoder(z) #modified_latent)
        return x1, z, mu, logvar

class classifier_FUNDUS(nn.Module):

    def __init__(self, latent_dim, n_classes, nhead, num_encoder_layers):

       super(classifier_FUNDUS, self).__init__()
       self.latent_dim = latent_dim
       self.n_classes = n_classes
       self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead),
            num_layers=num_encoder_layers)
       self.fc = nn.Linear(latent_dim, n_classes-1)
       self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class vae_classifier_fundus(nn.Module):
    def __init__(self, VAE_FUNDUS, classifier_FUNDUS):
        super(vae_classifier_fundus, self).__init__()
        self.vae_fundus = VAE_FUNDUS
        self.classifier_fundus = classifier_FUNDUS

    def forward(self, x):
        x1 = self.vae_fundus(x)
        prediction =self.classifier_fundus(z)
        return x1, z, mu, logvar
