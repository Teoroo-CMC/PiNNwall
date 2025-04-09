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
