import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
#import shap
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix,  confusion_matrix
import torch.nn as nn
from IPython import embed


class VAE(pl.LightningModule):
    
   def __init__(self, model, params):
      super(VAE, self).__init__()
      self.model = model
      self.params = params
      self.mse = F.mse_loss


   def forward(self, input): 
       return self.model(input)
   #def forward(self, input, **kwargs): 
   #    return self.model(input, **kwargs)


#   def on_train_epoch_start(self):
#       self.model.set_mode("training") 
 
   def training_step(self, batch, batch_idx):
     X, _ = batch
     w=0.0002
#     X = X.unsqueeze(1) 
     recon_x, mu, logvar = self(X)
     w_kld = self.update_kl(w, self.current_epoch)
     MSE = self.mse(recon_x, X, reduction='sum')
     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())   
     loss = MSE + KLD*w_kld 
     if self.current_epoch % 10 == 0:
        if X.size(0) == self.params.batch_size:
            comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
            self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
            self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
            self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
     loss_dict = {"MSE_loss": MSE.detach(), "loss": loss}
     self.log_dict(loss_dict)
     return loss_dict


   def training_epoch_end(self, outputs):
        avg_mse_loss = torch.stack([x["MSE_loss"] for x in outputs]).mean()        

        avg_kdl_loss = torch.stack([x["KLD_loss"] for x in outputs]).mean()

        avg_total_loss = torch.stack([x["loss"] for x in outputs]).mean()
          
        self.log_dict(
            {"avg_MSE_loss_training": avg_mse_loss, "avg_total_loss_training":avg_total_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
     

   def validation_step(self, batch, batch_idx):
         x, _ = batch
         w=0.0002
#         embed()
#         x = x.unsqueeze(1)
         recon_x, mu, logvar = self(x)
         w_kld = self.update_kl(w, self.current_epoch)
         MSE = self.mse(recon_x, x, reduction='sum')
         KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
         train_loss = MSE + KLD*w_kld #+ alp
         loss_dict = {"MSE_loss": MSE.detach(),  "Total_loss": train_loss}
         self.log_dict(loss_dict)
         if self.current_epoch % 10 == 0:
            if x.size(0) == self.params.batch_size:
               comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
               self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
               self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
               self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
         return loss_dict


   def validation_epoch_end(self, outputs):
         avg_mse_loss = torch.stack([x["MSE_loss"] for x in outputs]).mean()

         avg_kdl_loss = torch.stack([x["KLD_loss"] for x in outputs]).mean()

         avg_total_loss = torch.stack([x["Total_loss"] for x in outputs]).mean()

         self.log_dict(
            {"avg_MSE_loss_validation": avg_mse_loss, "avg_total_loss_validation":avg_total_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
          )

   def test_step(self, batch, batch_idx):
       x, _ = batch
       if x.size(0) == self.params.batch_size:
          comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
          self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1test{}.png".format(self.current_epoch), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2test{}.png".format(self.current_epoch), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3test{}.png".format(self.current_epoch), run_id=self.logger.run_id)


   def _shared_eval_step(self, batch, batch_idx):
         x, _ = batch
         x_recon, _, _ = self(x)
         
         x1, x2, x3 = self.get_concat_h(x,x_recon)        
         return x1, x2, x3   


   def update_kl(self, w, t):
    t_kl = w*(1.1)**(t-1)
    max_kl = 0.002
    if t_kl >= max_kl:
       w = max_kl
    else:
       w = t_kl
    return w


   def get_concat_h(self, x, x_recon):
       transform = T.ToPILImage()
       if x.size(0) == self.params.batch_size:
          x1= x[2, 0]   #,0][0]
          x2 = x[1, 64]   #,0][3]
          x3 = x[3, -3] #,0][5]
          x_recon1= x_recon[2,0]  #.item()    #[1,0].item() #[1] 
          x_recon2= x_recon[1,64] #[3]
          x_recon3= x_recon[3, -3] #,0][5]
          x1_1 = transform(x1)
          x_recon1_1 = transform(x_recon1)
          x2_2 = transform(x2)
          x_recon2_2 = transform(x_recon2)
          x3_3 = transform(x3)
          x_recon3_3 = transform(x_recon3) 
          dst1 = Image.new('RGB', (x1_1.width + x_recon1_1.width, x1_1.height))
          dst1.paste(x1_1, (0, 0))
          dst1.paste(x_recon1_1, (x1_1.width, 0))
          dst2 = Image.new('RGB', (x2_2.width + x_recon2_2.width, x2_2.height))
          dst2.paste(x2_2, (0, 0))
          dst2.paste(x_recon2_2, (x2_2.width, 0))
          dst3 = Image.new('RGB', (x3_3.width + x_recon3_3.width, x3_3.height))
          dst3.paste(x3_3, (0, 0))
          dst3.paste(x_recon3_3, (x3_3.width, 0))
          return dst1, dst2, dst3


#ef test_step(self, batch, batch_idx):
#         embed()
#         acc, precision, f1score, recallscore, t_n, f_p, f_n, t_p = self._shared_eval_step(batch, batch_idx)
#         save_imagen = save_image(comparison.cpu(), localhome/scclmg/MMVAE/experiments/mentiritas/oct_test.png)
#         metrics = {"Accuracy": acc, "Precision": precision, "f1_score": f1score, "recall_score": recallscore, "true_negatives": t_n, "false_positives": f_p, "false_negatives": f_n, "true_positives":t_p } # , "confusion_matrix": confusionmatrix}
#         self.log_dict(metrics)
#         return metrics #, save_imagen
         

#   def test_epoch_end(self, outputs):
#         save_imagen = save_imagen
 #        avg_loss = torch.stack([x["avg_total_loss"] for x in outputs]).mean()
#         embed()
#         avg_acc =  np.array([x["Accuracy"] for x in outputs]).mean()
#         avg_precision = np.array([x["Precision"] for x in outputs]).mean()
#         avg_f1_score = np.array([x["f1_score"] for x in outputs]).mean()
#         avg_recall_score = np.array([x["recall_score"] for x in outputs]).mean()
#         avg_t_n = np.array([x["true_negatives"] for x in outputs]).mean()
#         avg_f_p = np.array([x["false_positives"] for x in outputs]).mean()
#         avg_f_n = np.array([x["false_negatives"] for x in outputs]).mean()
#         avg_t_p = np.array([x["true_positives"] for x in outputs]).mean()
#         avg_confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).mean()
#         self.log_dict({"avg_acc": avg_acc, "avg_precision": avg_precision, "avg_f1_score": avg_f1_score, "avg_recall_score":avg_recall_score, "avg_t_n": avg_t_n, "avg_f_p":avg_f_p, "avg_f_n":avg_f_n, "avg_t_p": avg_t_p }) #, "avg_confusion_matrix":avg_confusion_matrix})


#   def _shared_eval_step(self, batch, batch_idx):
#         x, y = batch
#         x_recon, mu, logvar, y_predict = self(batch)
#         y_predict = torch.round(y_predict).cpu()
# #        y = y.cpu()
# #        embed() 
#         acc = accuracy_score(y, y_predict)
#         precision = precision_score(y, y_predict)
#         f1score = f1_score(y, y_predict)
#         recallscore = recall_score(y, y_predict)
#         confusionmatrix = confusion_matrix(y, y_predict)        
#         t_n, f_p, f_n, t_p = confusionmatrix.ravel() 
#         confusionmatrix = multilabel_confusion_matrix(y, y_predict)
#         comparison = torch.cat(x,x_recon)
#         embed()
#         save_image(comparison.cpu(), localhome/scclmg/MMVAE/experiments/mentiritas/oct_test.png  )
 #        return acc, precision, f1score, recallscore, t_n, f_p, f_n, t_p #, confusionmatrix #, comparison


   def configure_optimizers(self):

         algorithm = self.params.optimizer.algorithm
         algorithm = torch.optim.__dict__[algorithm]
         parameters = vars(self.params.optimizer.parameters)
         optimizer = algorithm(self.model.parameters(), **parameters)
         return optimizer
