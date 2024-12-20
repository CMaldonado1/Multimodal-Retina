import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix,  confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score, classification_report, precision_recall_fscore_support, average_precision_score


def save_images(x_fundus, x_oct, x_recon_f, x_recon_o, batch_size, current_epoch, run_id, logger_experiment):
          comparison1, comparison2 = get_concat_oct(x_oct, x_recon_o, batch_size)
          comparison1_f, comparison2_f = get_concat_fundus(x_fundus, x_recon_f, batch_size)
          logger_experiment(image=comparison1, artifact_file="reconstruction_1_oct_{}.png".format(current_epoch), run_id=run_id)
          logger_experiment(image=comparison2, artifact_file="reconstruction_2_oct_{}.png".format(current_epoch), run_id=run_id)
          logger_experiment(image=comparison1_f, artifact_file="reconstruction_1_fundus{}.png".format(current_epoch), run_id=run_id)
          logger_experiment(image=comparison2_f, artifact_file="reconstruction_2_fundus{}.png".format(current_epoch), run_id=run_id)

def get_concat_oct(x, x_recon, batch_size):
    transform = T.ToPILImage()
    if x.size(0) == batch_size:  #self.params.optimizer.batch_size:
          x1= x[2,0, 0]   #,0][0]
          x2 = x[1,0, 64]   #,0][3]
          x_recon1= x_recon[2,0,0]  #.item()    #[1,0].item() #[1]
          x_recon2= x_recon[1,0,64] #[3]
          x1_1 = transform(x1)
          x_recon1_1 = transform(x_recon1)
          x2_2 = transform(x2)
          x_recon2_2 = transform(x_recon2)
          dst1 = Image.new('RGB', (x1_1.width + x_recon1_1.width, x1_1.height))
          dst1.paste(x1_1, (0, 0))
          dst1.paste(x_recon1_1, (x1_1.width, 0))
          dst2 = Image.new('RGB', (x2_2.width + x_recon2_2.width, x2_2.height))
          dst2.paste(x2_2, (0, 0))
          dst2.paste(x_recon2_2, (x2_2.width, 0))
          return dst1, dst2


def get_concat_fundus(x, x_recon, batch_size):
       transform = T.ToPILImage()
       if x.size(0) == batch_size:
          x1= x[2]   #,0][0]
          x2 = x[1]   #,0][3]
          x_recon1= x_recon[2]  #.item()    #[1,0].item() #[1]
          x_recon2= x_recon[1] #[3]
          x1_1 = transform(x1)
          x_recon1_1 = transform(x_recon1)
          x2_2 = transform(x2)
          x_recon2_2 = transform(x_recon2)
          dst1 = Image.new('RGB', (x1_1.width + x_recon1_1.width, x1_1.height))
          dst1.paste(x1_1, (0, 0))
          dst1.paste(x_recon1_1, (x1_1.width, 0))
          dst2 = Image.new('RGB', (x2_2.width + x_recon2_2.width, x2_2.height))
          dst2.paste(x2_2, (0, 0))
          dst2.paste(x_recon2_2, (x2_2.width, 0))
          return dst1, dst2

def _collect_ids(outputs, filename=None):
      ids = [x["ids"] for x in outputs]
      ids = [id for sublist in ids for id in sublist]
      ids_ = pd.DataFrame(data=np.array([ids], dtype=object).T, columns=["IDs"])
      return ids_

def _log_z_vectors(self, outputs, filename):
        zl = torch.concat([x["z"] for x in outputs])
        zl_columns = [f"z{i:03d}" for i in range(zl.shape[1])] # z001, z002, z003, ...
        zl_df = pd.DataFrame(np.array(zl), columns=zl_columns)
        ids = _collect_ids(outputs)
        ids = pd.DataFrame(np.array(ids), columns=["IDs"])
        zl_df = pd.concat([ids, zl_df], axis=1)

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            zl_df.to_excel(writer, sheet_name="zl", index=False)

        self.logger.experiment.log_artifact(
            local_path = filename,
            artifact_path = "output", run_id=self.logger.run_id
        )

def update_kl(w, t):
    t_kl = w*(1.1)**(t-1)
    max_kl = 0.002
    if t_kl >= max_kl:
       w = max_kl
    else:
       w = t_kl
    return w

def evaluate_metrics(y_pred, y_true):
    y_predict = y_pred.cpu().detach().numpy().round()
    y_true = y_true.cpu().detach().numpy()
    
    acc = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, zero_division=0)
    sensitivity = recall_score(y_true, y_predict, zero_division=0)
    
    # Calculate specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        specificity = 0  # In case of an error, e.g., when there's only one class present

    # Calculate ROC AUC if there is more than one class
    if len(np.unique(y_true)) > 1:
        macro_roc_auc = roc_auc_score(y_true, y_pred.cpu().detach().numpy())
    else:
        macro_roc_auc = None

    return acc, precision, sensitivity, specificity, macro_roc_auc

def evaluation_log_dict_metrics(self, outputs, mode):
    def calculate_avg(metric_name):
        return np.array([x[metric_name] for x in outputs]).mean()
    
    def calculate_avg_ignore_none(metric_name):
        values = [x[metric_name] for x in outputs if x[metric_name] is not None]
        return np.mean(values) if values else None

    avg_acc = calculate_avg("acc")
    avg_precision = calculate_avg("precision")
    avg_sensitivity = calculate_avg("sensitivity")
    avg_specificity = calculate_avg("specificity")
    avg_roc_auc = calculate_avg_ignore_none("macro_roc_auc")
    
    log_dict = {
        f"avg_acc_{mode}": avg_acc,
        f"avg_precision_{mode}": avg_precision,
        f"avg_sensitivity_{mode}": avg_sensitivity,
        f"avg_specificity_{mode}": avg_specificity,
    }

    if avg_roc_auc is not None:
        log_dict[f"avg_roc_auc_{mode}"] = avg_roc_auc

    if mode != "test":
        avg_mse_loss = torch.stack([x["MSE_loss"] for x in outputs]).mean()
        avg_kdl_loss = torch.stack([x["KLD"] for x in outputs]).mean()
        avg_bce_loss = torch.stack([x["BCE"] for x in outputs]).mean()
        avg_total_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        log_dict.update({
            f"avg_MSE_loss_{mode}": avg_mse_loss,
            f"avg_total_loss_{mode}": avg_total_loss,
            f"avg_kdl_loss_{mode}": avg_kdl_loss,
            f"avg_bce_loss_{mode}": avg_bce_loss,
        })

    self.log_dict(
        log_dict,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
#        loss_dict = {"loss": loss_total, "loss_mse": loss_mse, "loss_kld":loss_kld, "loss_fundus":loss_fundus, "loss_mse_f":loss_mse_f, "loss_kld_f":loss_kld_f, "loss_oct":loss_oct, "loss_mse_o":loss_mse_o, "loss_kld_o":loss_kld_o}


def evaluation_log_dict(self, outputs, mode):
    log_dict = {}
    if mode != "test":
        avg_mse_loss_oct = torch.stack([x["loss_mse_o"] for x in outputs]).mean()
        avg_mse_loss_fundus = torch.stack([x["loss_mse_f"] for x in outputs]).mean()
        avg_mse_loss_total = torch.stack([x["loss_mse"] for x in outputs]).mean()
        avg_kdl_loss_oct = torch.stack([x["loss_kld_o"] for x in outputs]).mean()
        avg_kdl_loss_fundus = torch.stack([x["loss_kld_f"] for x in outputs]).mean()
        avg_kdl_loss_total = torch.stack([x["loss_kld"] for x in outputs]).mean()
        avg_total_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_total_loss_oct = torch.stack([x["loss_oct"] for x in outputs]).mean()
        avg_total_loss_fundus = torch.stack([x["loss_fundus"] for x in outputs]).mean()

        log_dict.update({
            f"avg_MSE_loss_oct_{mode}": avg_mse_loss_oct,
            f"avg_MSE_loss_fundus_{mode}": avg_mse_loss_fundus,
            f"avg_MSE_loss_{mode}": avg_mse_loss_total,
            f"avg_total_loss_{mode}": avg_total_loss,
            f"avg_kdl_loss_{mode}": avg_kdl_loss_total,
            f"avg_kdl_loss_fundus_{mode}": avg_kdl_loss_fundus,
            f"avg_kdl_loss_oct_{mode}": avg_kdl_loss_oct,
            f"avg_total_loss_oct_{mode}": avg_total_loss_oct,
            f"avg_total_loss_fundus_{mode}": avg_total_loss_fundus,
        })

    self.log_dict(
        log_dict,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )

