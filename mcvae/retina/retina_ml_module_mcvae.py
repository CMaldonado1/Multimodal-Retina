import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed
import pandas as pd
from utils import save_images, _log_z_vectors, get_concat_oct, get_concat_fundus, update_kl, evaluation_log_dict

class mcvae(pl.LightningModule):
    
   def __init__(self, model, params):
      super(mcvae, self).__init__()
      self.model = model
      self.params = params
      self.mse = F.mse_loss

   def forward(self, *input, **kwargs):
       return self.model(*input, **kwargs)

 
   def training_step(self, batch, batch_idx):
     fundus_, oct_, ids  = batch
     oct_ = oct_.unsqueeze(1)
     x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o = self(oct_, fundus_)
     loss_total, loss_mse, loss_kld, loss_oct, loss_fundus, loss_mse_o, loss_mse_f, loss_kld_o, loss_kld_f = self.loss_mcvae(fundus_, oct_, x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o)
     loss_dict = {"loss": loss_total, "loss_mse": loss_mse, "loss_kld":loss_kld, "loss_fundus":loss_fundus, "loss_mse_f":loss_mse_f, "loss_kld_f":loss_kld_f, "loss_oct":loss_oct, "loss_mse_o":loss_mse_o, "loss_kld_o":loss_kld_o}
     if self.current_epoch == 10:
         save_images(batch, batch_idx)
     return loss_dict


   def training_epoch_end(self, outputs):
       evaluation_log_dict(self, outputs, 'train')        

   def validation_step(self, batch, batch_idx):
         fundus_, oct_, ids= batch
         oct_ = oct_.unsqueeze(1)
         x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o = self(oct_, fundus_)
         loss_total, loss_mse, loss_kld, loss_oct, loss_fundus, loss_mse_o, loss_mse_f, loss_kld_o, loss_kld_f = self.loss_mcvae(fundus_, oct_, x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o)
         loss_dict = {"loss": loss_total, "loss_mse": loss_mse, "loss_kld":loss_kld, "loss_fundus":loss_fundus, "loss_mse_f":loss_mse_f, "loss_kld_f":loss_kld_f, "loss_oct":loss_oct, "loss_mse_o":loss_mse_o, "loss_kld_o":loss_kld_o}
         if self.current_epoch == 10:
            save_images(fundus_, oct_, x_recon_f, x_recon_o, self.params.optimizer.batch_size, self.current_epoch, self.logger.run_id, self.logger.experiment.log_image)
         return loss_dict


   def validation_epoch_end(self, outputs):
         evaluation_log_dict(self, outputs, mode='val')

   def test_step(self, batch, batch_idx):
       fundus_, oct_, ids = batch
       x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o = self(oct_, fundus_) 
       if self.current_epoch == 10:
          save_images(fundus_, oct_, x_recon_f, x_recon_o, self.params.optimizer.batch_size, self.current_epoch, self.logger.run_id, self.logger.experiment.log_image)
       return dict(**{"ids":ids, "z":z.cpu()}) 

   def test_epoch_end(self, outputs):
        _log_z_vectors(outputs, "latent_vector_test.xlsx")


   def loss_classifier(self, y_predict, y):
       bce_left = 0.0005 * self.bce(y_predict, y)
       return bce_left

   def loss_vae(self, x, recon_x, mu, logvar, z):
       w=0.00001
       w_kld = update_kl(w, self.current_epoch)
       MSE = self.mse(recon_x, x, reduction='mean')
       KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
       loss = MSE + KLD*w_kld
       return loss, MSE, KLD*w_kld

   def loss_mcvae(self, x_fundus, x_oct, x_recon_f, x_recon_o, z_f, mu_f, logvar_f, z_o, mu_o, logvar_o ):
       loss_oct, loss_mse_o, loss_kld_o = self.loss_vae(x_oct, x_recon_o, mu_o, logvar_o, z_o)
       loss_fundus, loss_mse_f, loss_kld_f = self.loss_vae(x_fundus, x_recon_f, mu_f, logvar_f, z_f)
       return loss_oct + loss_fundus, loss_mse_o + loss_mse_f, loss_kld_o + loss_kld_f, loss_oct, loss_fundus, loss_mse_o, loss_mse_f, loss_kld_o, loss_kld_f


   def configure_optimizers(self):

         algorithm = self.params.optimizer.algorithm
         algorithm = torch.optim.__dict__[algorithm]
         parameters = vars(self.params.optimizer.parameters)
         optimizer = algorithm(self.model.parameters(), **parameters)
         return optimizer
