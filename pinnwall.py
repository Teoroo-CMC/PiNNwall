from pinnwall_functions import *
import argparse

if  __name__ == '__main__':
    parser = argparse.ArgumentParser('My argument parser')
    parser.add_argument('--prefix_pinn_model','-p',default='./trained_models',type=str,help='path to the trained pinn model')
    parser.add_argument('--inputs_dir','-i',default='./',type=str,help='path to the input files of MW')
    parser.add_argument('--models','-m', nargs='+', default=['eem', 'acks2'],type=str,help='list of model used to compute the CRK')
    parser.add_argument('--output_log','-o',default='pinnwall.out',type=str,help='log of pinnwall')
    # parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help='Display help information')
    args = parser.parse_args()
    main(args)
