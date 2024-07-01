# Multimodal-Retina

## Predicting Cardiovascular Disease using OCT and Fundus Images

This repository contains the code for the paper **"Integrating Deep Learning with Fundus and Optical Coherence Tomography for Cardiovascular Disease Prediction"**.

### Docker Image

The Docker file used for this work can be found at the following link:

[Docker image](https://hub.docker.com/layers/scclmgadmin/oct/oct/images/sha256-af579dc2cab9c7504937fdea208c683a61603126fbab4ccf641a4c2bef71b043?context=repo)

### Conda Environment

For running the code in Conda environments instead of Docker images, use the provided `environment.yml` file.

### Overview

The proposed model in this work is divided into two parts: the pretraining of the MCVAE (Multi-Channel Variational Autoencoder) and the task-aware MCVAE. The scripts for the first stage are located in the `mcvae` folder, while those for the second stage are in the `class_mcvae` folder.

#### MCVAE (Pretraining)

1. **Retina**: The main Python script for training both modalities is called `main.py`.

2. **Fundus**: The Python script for executing the pretraining of the VAE using only Fundus images is called `fundus_main.py`.

3. **OCT**: The Python script for executing the pretraining of the VAE using only OCT images is called `oct_main.py`.

To modify or create hyperparameter searches, simply edit the `config.yaml` file according to the required additions.

#### Task-Aware MCVAE

1. **Retina**: The main Python script for training both modalities is called `main.py`.

2. **Fundus**: The Python script for executing the pretraining of the VAE using only Fundus images is called `fundus_main.py`.

3. **OCT**: The Python script for executing the pretraining of the VAE using only OCT images is called `oct_main.py`.

To modify or create hyperparameter searches, simply edit the `config.yaml` file according to the required additions.

### Data

The retina data used in this study is from the Field-ID-UKBB: 21012.

[21012-UKBB](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21012)

### Usage

#### Setting Up the Environment

1. **Using Docker**:
   - Pull the Docker image:
     ```sh
     docker pull scclmgadmin/oct:latest
     ```
   - Run the Docker container:
     ```sh
     docker run -it scclmgadmin/oct:latest
     ```

2. **Using Conda**:
   - Create the Conda environment:
     ```sh
     conda env create -f environment.yml
     ```
   - Activate the environment:
     ```sh
     conda activate your_environment_name
     ```

#### Running the Scripts

1. **Pretraining MCVAE**:
   - For Retina (both modalities):
     ```sh
     python mcvae/main.py
     ```
   - For Fundus:
     ```sh
     python mcvae/fundus_main.py
     ```
   - For OCT:
     ```sh
     python mcvae/oct_main.py
     ```

2. **Task-Aware MCVAE**:
   - For Retina (both modalities):
     ```sh
     python class_mcvae/main.py
     ```
   - For Fundus:
     ```sh
     python class_mcvae/fundus_main.py
     ```
   - For OCT:
     ```sh
     python class_mcvae/oct_main.py
     ```

### Configuration

All hyperparameters and configurations can be adjusted in the `config.yaml` file. Make sure to update the file according to your specific needs before running the scripts.


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
