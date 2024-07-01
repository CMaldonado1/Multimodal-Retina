import torchvision.transforms as T
import shap
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix,  confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score, classification_report, precision_recall_fscore_support, average_precision_score


def save_images(x, x_recon, batch_size):
       if x.size(0) == self.params.optimizer.batch_size:
          comparison1 = self._shared_eval_step(x, x_recon)
          self.logger.experiment.log_image(image=comparison1, artifact_file="reconstruction1{}.png".format(self.current_epoch), run_id=self.logger.run_id)
#          self.logger.experiment.log_image(image=comparison2, artifact_file="reconstruction2{}.png".format(self.current_epoch), run_id=self.logger.run_id)
#          self.logger.experiment.log_image(image=comparison3, artifact_file="reconstruction3{}.png".format(self.current_epoch), run_id=self.logger.run_id)


def get_concat_h(x, x_recon, batch_size):
       transform = T.ToPILImage()
       if x.size(0) == batch_size:
          x1= x[0]   #,0][0]
#          x2 = x[1]   #,0][3]
#          x3 = x[3] #,0][5]
          x_recon1= x_recon[0]  #.item()    #[1,0].item() #[1]
#          x_recon2= x_recon[1] #[3]
#          x_recon3= x_recon[3] #,0][5]
          x1_1 = transform(x1)
          x_recon1_1 = transform(x_recon1)
#          x2_2 = transform(x2)
#          x_recon2_2 = transform(x_recon2)
#          x3_3 = transform(x3)
#          x_recon3_3 = transform(x_recon3)
          dst1 = Image.new('RGB', (x1_1.width + x_recon1_1.width, x1_1.height))
          dst1.paste(x1_1, (0, 0))
          dst1.paste(x_recon1_1, (x1_1.width, 0))
#          dst2 = Image.new('RGB', (x2_2.width + x_recon2_2.width, x2_2.height))
#          dst2.paste(x2_2, (0, 0))
#          dst2.paste(x_recon2_2, (x2_2.width, 0))
#          dst3 = Image.new('RGB', (x3_3.width + x_recon3_3.width, x3_3.height))
#          dst3.paste(x3_3, (0, 0))
#          dst3.paste(x_recon3_3, (x3_3.width, 0))
          return dst1 #, dst2, dst3

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

def __shap__(self, outputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_shapes = [x["z"].shape for x in outputs]
    tensors_z = [x["z"] for x in outputs]
    tensors_array = np.concatenate(tensors_z, axis=0)
    mean_values = np.mean(tensors_array, axis=0)
    mean_across_tensors = torch.tensor(mean_values)
    mean_across_tensors = mean_across_tensors.unsqueeze(0).to(device)
    explainer = shap.DeepExplainer(self.model.classifier_fundus, mean_across_tensors)
    shap_values = explainer.shap_values(mean_across_tensors, check_additivity=False)
    np.save('shap_values.npy', shap_values)
    if isinstance(shap_values, list):
       for i, val in enumerate(shap_values):
           print(f"Shape of shap_values[{i}]: {val.shape}")
    else:
        print(f"Shape of shap_values: {shap_values.shape}")
    shap_df = pd.DataFrame(shap_values[0])
    shap_df.to_csv('shap_values.csv', index=False)


def evaluation_log_dict(self, outputs, mode):
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


def visualize_output(modified_image, oct_imag_left_list, ids):
    pil_image = transforms.ToPILImage()(oct_imag_left_list[0])
    pil_image1 = transforms.ToPILImage()(modified_image[0])
    pil_image.save('imagen_original.jpg')
    pil_image1.save('imagen_modificada.jpg')

    img1 = cv2.imread('imagen_original.jpg')
    img2 = cv2.imread('imagen_modificada.jpg')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(gray1)

    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=50, qualityLevel=0.3, minDistance=7)
    prevPts = corners.reshape(-1, 1, 2)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    nextPts, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, prevPts, None, **lk_params)

    prevPts = prevPts[status == 1]
    nextPts = nextPts[status == 1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for pt1, pt2 in zip(prevPts, nextPts):
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        circle = plt.Circle((x2, y2), radius=3, color='r') #fill=False, edgecolor='yellow')
        axes[0].add_patch(circle)    

    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for pt1, pt2 in zip(prevPts, nextPts):
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        circle = plt.Circle((x2, y2), radius=3, color='r') #fill=False, edgecolor='yellow')
        axes[1].add_patch(circle)

    axes[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), alpha=0.3)
    axes[1].set_title('Modified Image')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Optical Flow')
    axes[2].axis('off')
    plt.savefig(f'modified_image_with_circles_{ids}.png')

    plt.tight_layout()
    plt.show()
