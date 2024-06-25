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


class AE(pl.LightningModule):
    
   def __init__(self, model, params):
      super(AE, self).__init__()
      self.model = model
      self.params = params
      self.mse = F.mse_loss


   def forward(self, input): 
       return self.model(input)
 
   def training_step(self, batch, batch_idx):
     x, _ = batch
     x = x.unsqueeze(dim=1)
     recon_x, z = self(x)
     loss = self.mse(recon_x, x, reduction='mean')
     if self.current_epoch % 10 == 0:
        if x.size(0) == self.params.optimizer.batch_size:
            comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
            self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
            self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
            self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3_train{}.png".format(self.current_epoch), run_id=self.logger.run_id)
     loss_dict = {"loss": loss}
     return loss_dict


   def training_epoch_end(self, outputs):
        avg_mse_loss = torch.stack([x["loss"] for x in outputs]).mean()        
          
        self.log_dict(
            {"avg_total_loss_training":avg_mse_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
     

   def validation_step(self, batch, batch_idx):
         x, _ = batch
         x = x.unsqueeze(dim=1)
         recon_x, z = self(x)
         MSE = self.mse(recon_x, x, reduction='mean')
         val_loss = MSE 
         loss_dict = {"Total_loss": val_loss.detach()}
         if self.current_epoch % 10 == 0:
            if x.size(0) == self.params.optimizer.batch_size:
               comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
               self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
               self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
               self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3_val{}.png".format(self.current_epoch), run_id=self.logger.run_id)
         return loss_dict


   def validation_epoch_end(self, outputs):
         avg_total_loss = torch.stack([x["Total_loss"] for x in outputs]).mean()

         self.log_dict(
            {"avg_total_loss_validation":avg_total_loss },
            on_epoch=True,
            prog_bar=True,
            logger=True,
          )

   def test_step(self, batch, batch_idx):
       x, ids = batch
       x = x.unsqueeze(dim=1)
       recon_x, z = self(x) 
       if x.size(0) == self.params.optimizer.batch_size:
          comparison1, comparison2, comparison3 = self._shared_eval_step(batch, batch_idx)
          self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1test{}.png".format(self.current_epoch), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2test{}.png".format(self.current_epoch), run_id=self.logger.run_id)
          self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3test{}.png".format(self.current_epoch), run_id=self.logger.run_id)
       return dict(**{"ids":ids, "z":z.cpu()}) 

   def test_epoch_end(self, outputs):
        self._log_z_vectors(outputs, "latent_vector_test.xlsx")

   def _shared_eval_step(self, batch, batch_idx):
         x, _ = batch
         x = x.unsqueeze(dim=1)
         x_recon, _ = self(x)
         
         x1, x2, x3 = self.get_concat_h(x,x_recon)        
         return x1, x2, x3   

   def get_concat_h(self, x, x_recon):
       transform = T.ToPILImage()
       if x.size(0) == self.params.optimizer.batch_size:
          x1= x[2,0, 0]   #,0][0]
          x2 = x[1,0, 64]   #,0][3]
          x3 = x[3,0, -3] #,0][5]
          x_recon1= x_recon[2,0,0]  #.item()    #[1,0].item() #[1] 
          x_recon2= x_recon[1,0,64] #[3]
          x_recon3= x_recon[3,0, -3] #,0][5]
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

   def _collect_ids(self, outputs, filename=None):
      ids = [x["ids"] for x in outputs]
      ids = [id for sublist in ids for id in sublist]
      ids_ = pd.DataFrame(data=np.array([ids], dtype=object).T, columns=["IDs"])
      return ids_

   def _log_z_vectors(self, outputs, filename):
        zl = torch.concat([x["z"] for x in outputs])
        zl_columns = [f"z{i:03d}" for i in range(zl.shape[1])] # z001, z002, z003, ...
        zl_df = pd.DataFrame(np.array(zl), columns=zl_columns)
        ids = self._collect_ids(outputs)
        ids = pd.DataFrame(np.array(ids), columns=["IDs"])
        zl_df = pd.concat([ids, zl_df], axis=1)

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            zl_df.to_excel(writer, sheet_name="zl", index=False)

        self.logger.experiment.log_artifact(
            local_path = filename,
            artifact_path = "output", run_id=self.logger.run_id
        )


   def configure_optimizers(self):

         algorithm = self.params.optimizer.algorithm
         algorithm = torch.optim.__dict__[algorithm]
         parameters = vars(self.params.optimizer.parameters)
         optimizer = algorithm(self.model.parameters(), **parameters)
         return optimizer
