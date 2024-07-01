# Multimodal-Retina

## Predicting-CVD-using-OCT-fundus-images
This repository contain the code for the paper "Integrating Deep Learning with Fundus and Optical Coherence Tomography for Cardiovascular Disease Prediction"

Docker Image
The Docker file used for this work can be found at the following link:

[Docker image](https://hub.docker.com/layers/scclmgadmin/oct/oct/images/sha256-af579dc2cab9c7504937fdea208c683a61603126fbab4ccf641a4c2bef71b043?context=repo)

Overview
El modelo propuesto en este trabajo esta divido en dos partes, la primera es el pretraining del mcvae y la segunda el el task-aware MCVAE. Los scripts del primer stage, se encuentran en la carpeta mcvae, mientras que los correspondientes del segundo stage estan al folder class_mcvae. 

### MCVAE (Pretraining)
#### Retina: El python script principal donde se entrena ambas modalidades estan se llama main.py.
Para modificar/crear los hyperparmeter search, basta con modificar el config.yaml dependiendo de lo que se tiene que agregar.

#### Fundus: El python script para ejecutar el pretraining del VAE usando solamente Fundus, se llama fundus_main.py

#### OCT: El python script para ejecutar el pretraining del VAE usando solamente OCT, se llama oct_main.py

### Task-Aware MCVAE 
#### Retina: El python script principal donde se entrena ambas modalidades estan se llama main.py.
Para modificar/crear los hyperparmeter search, basta con modificar el config.yaml dependiendo de lo que se tiene que agregar.

#### Fundus: El python script para ejecutar el pretraining del VAE usando solamente Fundus, se llama fundus_main.py

#### OCT: El python script para ejecutar el pretraining del VAE usando solamente OCT, se llama oct_main.py
