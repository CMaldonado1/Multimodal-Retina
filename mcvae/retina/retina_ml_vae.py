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

class MCVAE(nn.Module):
    def __init__(self, VAE_FUNDUS, VAE_oct):
        super(MCVAE, self).__init__()
#        self.class_vae_fundus = vae_classifier_fundus(VAE_FUNDUS, classifier_FUNDUS)
        self.vae_fundus =  VAE_FUNDUS
#        self.class_fundus = classifier_FUNDUS(latent_dim, n_classes, nhead, num_encoder_layers)
#        self.class_vae_oct = vae_classifier_oct(VAE_oct, classifier_oct)
        self.vae_oct =  VAE_oct
        self.latent_combiner = LatentCombiner(2048,128)
#        self.class_oct = classifier_oct(latent_dim, n_classes, nhead, num_encoder_layers)
    
    def forward(self, x_oct, x_fundus):
        z_f, mu_f, logvar_f = self.vae_fundus.encoder(x_fundus)
        z_o, mu_o, logvar_o = self.vae_oct.encoder(x_oct)
        z_combined = self.latent_combiner(mu_f, logvar_f, mu_o, logvar_o)
        x_recon_f = self.vae_fundus.decoder(z_combined[0])
        x_recon_o = self.vae_oct.decoder(z_combined[1])
        return x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o


        
