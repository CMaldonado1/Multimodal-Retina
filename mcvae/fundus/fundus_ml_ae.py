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

class AE_fundus(nn.Module):

    def __init__(self, input_dim, latent_dim, n_classes, n_channels, kernel_size, padding, stride):

       super(AE_fundus, self).__init__()
       self.latent_dim = latent_dim
       self.n_classes = n_classes
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


    def bottleneck(self, h):
        mu = self.fc1(h)
        return mu

    def encoder(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        z = self.bottleneck(h)
        return z

    def decoder(self, z):
        h = self.fc3(z)
        for layer in self.decoder_layers:
            h = layer(h)
        return h


    def forward(self, x):
        z = self.encoder(x)
        x1 = self.decoder(z)
        return x1
