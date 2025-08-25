""""PiNNwall runscript 

This script is used to run PiNNwall through the command line. 
It defines a description, extra information to be printed when the help flag is used, and arguments.
Then it executes the main function from pinnwall_functions.py which runs PiNNwall.

To run PiNNwall use the following command:

python pinnwall [-p <MODEL_DIR>] [-i <WORKING_DIR>] [-m <method_name>] [-o <filename>]

Options:

`--path_pinn_model, -p  <MODEL_DIR> (default=./trained_models)`
path to the trained pinn model

`--inputs_dir, -i <WORKING_DIR> (default=./)`
path to the input files of MW

`--models, -m <method_name> (default=eem)`
List of model used to compute the CRK to construct the Hessian Matrix employed by Metalwalls. To pass multiple model types, i.e. `-m eem local etainv acks2`

`--output_log, -o <filename> (default=pinnwall.out)`
log of pinnwall, default to *inputs_dir*
"""

from pinnwall_functions import *
import argparse

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PiNNwall integrates a machine learned charge response kernel (CRK) predicted by PiNN with the Metalwalls MD simulation software.', epilog='For more information on the use of PiNNwall see: https://github.com/Teoroo-CMC/PiNNwall')
    parser.add_argument('--path_pinn_model','-p',default='./trained_models',type=str,help='path to the trained pinn model')
    parser.add_argument('--inputs_dir','-i',default='./',type=str,help='path to the input files of MW')
    parser.add_argument('--models','-m', nargs='+', default=['eem'],type=str,help='modeltype used to compute the CRK')
    parser.add_argument('--output_log','-o',default='pinnwall.out',type=str,help='log of pinnwall')
    args = parser.parse_args()
    main(args)
