import sys, os
import yaml
import logging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import numpy as np
import torch
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger
import argparse
from subprocess import check_output
from argparse import Namespace
from oct_ml_vae_trans import VAE_oct, vae_classifier_oct, classifier_oct
from pl_datamodule_oct import oct_DM
from load_config import load_config
from oct_ml_module_class import AE
from IPython import embed
from pytorch_lightning.profiler import PyTorchProfiler
from torch.profiler import profile, record_function, ProfilerActivity

def get_eye_args(config):
    net = config.network_architecture
    convs = net.convolution.parameters
    class_ = config.classifier
    vae_args = {
                 "input_dim": config.input_dim,
                 "latent_dim": net.latent_dim,
                 "n_channels": net.convolution.parameters.channels,
                 "kernel_size": net.convolution.parameters.kernel_size,
                 "padding":net.convolution.parameters.padding,
                 "stride":net.convolution.parameters.stride
                  }
    class_args = {
               "latent_dim": net.latent_dim,
               "n_classes": class_.n_classes,
               "nhead": class_.nhead,
               "num_encoder_layers": class_.num_encoder_layers 
                 }
    return vae_args, class_args

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def get_datamodule(config):
#        #TOFIX: make this more general
    dm = oct_DM(config.dir_imgs_ae, config.ids_set_class, img_size=config.input_dim[-3:], batch_size=config.optimizer.batch_size) 
    return dm

def ml_model1_trainer(config):
     dm_ae = get_datamodule(config)
     ae_oct_args = get_eye_args(config)
     ae_oct = VAE_oct(**ae_oct_args)
     ckpt= torch.load("/cmaldonado/multimodal_retina/Multimodal-Retina/mlruns/2/7fb3cb89c1774154be729a1289178f5f/checkpoints/epoch=34-step=78224.ckpt", map_location= torch.device('cuda:0'))
     _model_pretrained_weights = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()} 
     model1 = AE(ae_oct, config)
     model1.model.load_state_dict(_model_pretrained_weights)
     trainer_ae = pl.Trainer(accelerator="gpu", devices=1, callbacks=[EarlyStopping(monitor="avg_total_loss_validation", mode="min", patience=10)], max_epochs=0)
     return dm_ae, model1, trainer_ae


def ml_model2_trainer(config):
     dm_vae = get_datamodule(config)
     vae_oct_args, class_oct_args = get_eye_args(config)
     oct_vae = VAE_oct(**vae_oct_args)
     oct_class = classifier_oct(**class_oct_args)
     classifiers = vae_classifier_oct(oct_vae, oct_class)
     ckpt= torch.load(config.pretrained_model, map_location= torch.device('cuda:0'))
     _model_pretrained_weights = {k.replace("model.", "vae_oct."): v for k, v in ckpt['state_dict'].items()} #if "fc"  in k}
     model2 =  AE(classifiers, config)
     model2.model.load_state_dict(_model_pretrained_weights, strict=False)
     early_stopping_callback = [EarlyStopping(monitor="avg_total_loss_val", mode='min',  patience=15)]  #, ModelSummary(max_depth=-1)]
     trainer = pl.Trainer(accelerator="gpu", callbacks=early_stopping_callback, devices=1, num_nodes=1, max_epochs=-1)  
     return dm_vae, model2, trainer #, trainer_right


def get_mlflow_parameters(config):

    mlflow_parameters = {
            "platform": check_output(["hostname"]).strip().decode(),
            "w_kl": config.w_kld,
            "w_bce": config.w_bce,
            "latent_dim": config.network_architecture.latent_dim,
            "n_channels": config.network_architecture.convolution.parameters.channels,
            "batch_size": config.optimizer.batch_size
    }
    return mlflow_parameters


def main(config):
      # dm_vae, dm_classification,  model1, model2, trainer_vae, trainer = ml_model_trainer(config)
#      dm_classification, model2, trainer = ml_model2_trainer(config)  
      if config.log_to_mlflow:
          mlflow.pytorch.autolog()
              
          if config.pretrained_model is None:
              exp1 = config.mlflow.pretraining.experiment_name
              exp1 = exp1 if exp1 is not None else "default"
              mlf_logger = MLFlowLogger(experiment_name=exp1, tracking_uri="file:./mlruns")

              try:
                  exp_id = mlflow.create_experiment(exp1)
              except:
                  # If the experiment already exists, we can just retrieve its ID
                  exp_id = mlflow.get_experiment_by_name(exp1).experiment_id

              with mlflow.start_run(run_id=mlf_logger.run_id, experiment_id=exp_id, run_name=config.mlflow.pretraining.run_name) as run:
                  # experiment_id, run_id = run.experiment_id, run.run_id
                   dm_ae, model1, trainer_ae = ml_model1_trainer(config)
                   trainer_ae.logger = mlf_logger
                   for k, v in get_mlflow_parameters(config).items():
                     mlflow.log_param(k, v)
                   trainer_ae.fit(model1, datamodule=dm_ae) #, logger=mlf_logger)            
                   print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))          
          else: 
           exp2 = config.mlflow.classifier_training.experiment_name
           exp2 = exp2 if exp2 is not None else "default"
           mlf_logger_2 = MLFlowLogger(experiment_name=exp2, tracking_uri="file:./mlruns")

           try:
            exp_id = mlflow.create_experiment(exp2)
           except:
            # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(exp2).experiment_id

           with mlflow.start_run(experiment_id=exp_id, run_id=mlf_logger_2.run_id, run_name=config.mlflow.classifier_training.run_name ) as run: 
            for k, v in get_mlflow_parameters(config).items():
                     mlflow.log_param(k, v)
            try: 
                mlflow.log_param("base_model", run_id)
            except:
                mlflow.log_param("base_model", config.pretrained_model)

            dm_vae, model2, trainer = ml_model2_trainer(config)
            trainer.logger = mlf_logger_2
            trainer.fit(model2, datamodule=dm_vae)
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
      else:
         #trainer_ae.fit(model1, datamodule=dm_ae)  
         trainer.fit(model2, datamodule=dm_vae)
         
      trainer.test(model2, datamodule=dm_vae)# , trainer=trainer)
#      trainer.predict(model2, datamodule=dm_classification)
#      trainer_ae.test(model1, datamodule=dm_ae)

if __name__ == '__main__':
 
    import argparse

 
    parser = argparse.ArgumentParser(description='AE oct')
    parser.add_argument('--config', default = 'config_oct.yaml')
    parser.add_argument('--latent_dim', type=int, default=None,
                    help='latent dimensionality (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                    help='batch size for data (default: 1)')
    parser.add_argument('--epochs', type=int, default=None, metavar='E',
                    help='number of epochs to train (default: 5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--dir_imgs', type=str, default = None)
    parser.add_argument('--ids_set', type=str, default = None)
    parser.add_argument('--lr', type=float, default=None,
                    help='the learning rate')
    parser.add_argument('--w_kld', type=float, default=None,
                    help='the weight of the KL term.')
    parser.add_argument('--n_classes', type=int, default=None,
                    help='num of classes')
    parser.add_argument('--weight_decay', type=float, default=None,
                    help='the weight decay')
#    parser.add_argument("--log_to_mlflow", default=False, action="store_true",
#                    help="Set this flag if you want to log the run's data to MLflow.",)
    parser.add_argument("--disable_mlflow_logging", default=False, action="store_true",
        help="Set this flag if you don't want to log the run's data to MLflow.",)

    args = parser.parse_args()
    # Configurar el logger
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ### Load configuration
    if not os.path.exists(args.config):
        logger.error("Config not found" + args.config)

    config = load_config(args.config, args)
    config.log_to_mlflow = not args.disable_mlflow_logging
    main(config)

   




