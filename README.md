# PiNNwall

This project is about using PiNN to predict the charge response kernal (CPK) and use it in the Metwalwall MD simulation. PiNNWall is based on the work presented in this paper (bibtex here for citation). Here is a link to the Jupyter notebook Tutorial.

# Preparation 
The PiNNWall is based on PiNN (link) and predicts the Hessian Matrix for the MetalWall (link), so to run PiNNWall, one needs to have PiNN installed first.

Before prediction, one needs to have the input files of MetalWall, namely the data.inpt and runtime.inpt.

Then, clone this repo to get the scripts and the ML-models that will be used in predicting the CPK.

# Synopsis
python pinnwall.py [-p <MODEL_DIR>] [-i <WORKING_DIR>] [-m [<methodename>]] [-o <filename>]

Options:

-p  <MODEL_DIR>(./trained_models)
path to the trained pinn model

-i <WORKING_DIR>(./)
path to the input files of MW

-m [<methodename>](eem)
list of model used to compute the CRK

-o <filename>(pinnwall.out)
log of pinnwall, default to inputs_dir 

# Known Issues

