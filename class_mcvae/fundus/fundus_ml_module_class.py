import pytorch_lightning as pl
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed
import pandas as pd
from utils import update_kl, _log_z_vectors, get_concat_h, evaluate_metrics,  evaluation_log_dict 


class AE(pl.LightningModule):
    
   def __init__(self, model, params):
      super(AE, self).__init__()
      self.model = model
      self.params = params
      self.mse = F.mse_loss
      self.bce = nn.BCELoss()

   def forward(self, input): 
       return self.model(input)
 
   def training_step(self, batch, batch_idx):
     x, _, y = batch
     y_pred, recon_x, mu, logvar, z = self(x)
     MSE, KLD = self.loss_vae(x, recon_x, mu, logvar, z)
     bce = self.loss_classifier(y_pred, y.float())
     loss = MSE + bce
     if self.current_epoch % 10 == 0:
         self.save_images(x, recon_x, mode='train')
     acc, precision, sensitivity, specificity, macro_roc_auc = evaluate_metrics(y_pred, y)
     loss_dict = {"MSE_loss": MSE.detach(), "loss": loss, "KLD": KLD, "BCE":bce, "acc":acc, "precision":precision, "sensitivity":sensitivity, "specificity":specificity, "macro_roc_auc":macro_roc_auc}
     return loss_dict


   def training_epoch_end(self, outputs):
        evaluation_log_dict(self, outputs, mode='train') 

   def validation_step(self, batch, batch_idx):
         x, _, y = batch
         y_pred, recon_x, mu, logvar, z = self(x)
         MSE, KLD = self.loss_vae( x, recon_x, mu, logvar, z)
         bce = self.loss_classifier(y_pred, y.float())
         val_loss = MSE + bce
         if self.current_epoch % 10 == 0:
            self.save_images(x, recon_x, mode='val')
         acc, precision, sensitivity, specificity, macro_roc_auc = evaluate_metrics(y_pred, y)   
         loss_dict = {"loss": val_loss.detach(), "MSE_loss": MSE.detach(), "KLD": KLD, "BCE":bce, "acc":acc, "precision":precision, "sensitivity":sensitivity, "specificity":specificity, "macro_roc_auc":macro_roc_auc}
         return loss_dict


   def validation_epoch_end(self, outputs):
        evaluation_log_dict(self, outputs, mode='val')

   def test_step(self, batch, batch_idx):
       x, ids, y = batch
       y_pred, recon_x, mu, logvar, z = self(x)
       acc, precision, sensitivity, specificity, macro_roc_auc = evaluate_metrics(y_pred, y)
       return dict(**{"ids":ids, "z":z.cpu(), "acc":acc, "precision":precision, "sensitivity":sensitivity, "specificity":specificity, "macro_roc_auc":macro_roc_auc}) 

   def test_epoch_end(self, outputs):
        _log_z_vectors(self, outputs, "latent_vector_test.xlsx")
        evaluation_log_dict(self, outputs, mode='test')

   def _shared_eval_step(self, x,x_recon):
         x1, x2, x3 = get_concat_h(x,x_recon, self.params.optimizer.batch_size)        
         return x1, x2, x3   


   def loss_classifier(self, y_predict, y):
       bce_left = 0.0005 * self.bce(y_predict, y)
       return bce_left

   def loss_vae(self, x, recon_x, mu, logvar, z):
       w=0.00001
       w_kld = update_kl(w, self.current_epoch)
       MSE = self.mse(recon_x, x, reduction='mean')
       KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
       loss = MSE + KLD*w_kld
       return loss, KLD*w_kld

   def save_images(self, x, x_recon, mode):
       if x.size(0) == self.params.optimizer.batch_size:
          comparison1, comparison2, comparison3 = self._shared_eval_step(x, x_recon)
          self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1_{}_{}.png".format(self.current_epoch, mode), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2_{}_{}.png".format(self.current_epoch, mode), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3_{}_{}.png".format(self.current_epoch, mode), run_id=self.logger.run_id)


   def configure_optimizers(self):

         algorithm = self.params.optimizer.algorithm
         algorithm = torch.optim.__dict__[algorithm]
         parameters = vars(self.params.optimizer.parameters)
         optimizer = algorithm(self.model.parameters(), **parameters)
         return optimizer
