# Project_VAE
Variational Autoencoder (VAE) codes for the paer "J. Chem. Inf. Model. 2025, 65, 23, 12846–12860"
https://doi.org/10.1021/acs.jcim.5c02323

## THE CODE HAS BEEN DIVIDED INTO 5 CHUNKS ##

Step1_Hyperparameter_Tuning.py: Loading of the trajectories and perfoming hyperparameter tuning.

Step2_Model_Training.py: Building the model with the best hyperparameters and splitting the data into train and test sets, also saves the best model as .pth file.

Step3_Validation.py: Extracting coordinates from the test set via the decoder of the best model and comparing the rmsd against its actual structures, relative to the first frame.

Step4_Clustering_and_Point_Generation.py: Clustering of the total latent space and generating novel points.

Step5_Frame_Generation.py: Decoding these points into new pdb structures for subsequent MD runs.
