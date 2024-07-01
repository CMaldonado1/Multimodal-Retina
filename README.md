# Multimodal-Retina

## Predicting-CVD-using-OCT-fundus-images
This repository contain the code for the paper "Integrating Deep Learning with Fundus and Optical Coherence Tomography for Cardiovascular Disease Prediction"

Docker Image
The Docker file used for this work can be found at the following link:

Docker image

Overview
In the src/ folder, you'll find the scripts needed to train the Variational Autoencoder, specifically with this command:

python oct_main_ml_fc2.py

You can modify the arguments to conduct a hyperparameter search grid.

An example of a bash script to run the hyperparameter script is also included: hyperparameter_grid_search.sh. In this bash script provides an example of running 4 jobs in parallel simultaneously for all grid searches; the feasibility of this will depend on the capacity of the HPC you are using. In the jupyter-notebooks/ folder, you'll find the notebooks for the seven classifiers tasks. Make sure to include the Excel files that will contain the Z's and metadata information.

In the jupyter-notebooks are the code of the Random Forest classifiers (the seven specifies in the manuscript)
