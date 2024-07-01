import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F

class LatentCombiner(nn.Module):
    def __init__(self, latent_dim_fundus, latent_dim_oct):
        super(LatentCombiner, self).__init__()
        self.fc = nn.Linear(latent_dim_fundus + latent_dim_oct, latent_dim_fundus + latent_dim_oct)

    def forward(self, mu_fundus, logvar_fundus, mu_oct, logvar_oct):
        z_fundus = self.reparameterize(mu_fundus, logvar_fundus)
        z_oct = self.reparameterize(mu_oct, logvar_oct)
        z_combined = [z_fundus, z_oct]  #torch.cat((z_fundus, z_oct), dim=1)
#        z_combined = F.relu(self.fc(z_combined))
        return z_combined

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class MCVAE_retina(nn.Module):
    def __init__(self, VAE_FUNDUS, VAE_oct):
        super(MCVAE_retina, self).__init__()
        self.vae_fundus =  VAE_FUNDUS
        self.vae_oct =  VAE_oct
        self.latent_combiner = LatentCombiner(2048,128)
    
    def forward(self, x_oct, x_fundus):
        z_f, mu_f, logvar_f = self.vae_fundus.encoder(x_fundus)
        z_o, mu_o, logvar_o = self.vae_oct.encoder(x_oct)
        z_combined = self.latent_combiner(mu_f, logvar_f, mu_o, logvar_o)
        x_recon_f = self.vae_fundus.decoder(z_combined[0])
        x_recon_o = self.vae_oct.decoder(z_combined[1])
        return x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o

class classifier_retina(nn.Module):

    def __init__(self, classifier_FUNDUS, classifier_OCT):

       super(classifier_retina, self).__init__()
       self.classifier_fundus = classifier_FUNDUS
       self.classifier_oct = classifier_OCT

    def forward(self, mu_f, mu_o):
        y_f = self.classifier_fundus(mu_f)
        y_o = self.classifier_oct(mu_o)
        return y_f, y_o

class mcvae_classifier_retina(nn.Module):
    def __init__(self, vae_classifier_fundus, vae_classifier_oct):
        super(mcvae_classifier_retina, self).__init__()
        self.vae_classifier_fundus = vae_classifier_fundus
        self.vae_classifier_oct = vae_classifier_oct

    def forward(self, x_oct, x_fundus):
        y_f, x_recon_f, z_f, mu_f, logvar_f = self.vae_classifier_fundus(x_fundus)
        y_o, x_recon_o, z_o, mu_o, logvar_o = self.vae_classifier_oct(x_oct)
        return y_f, y_o, x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o 
 
