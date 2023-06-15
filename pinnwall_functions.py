from pinn.io import sparse_batch
from glob import glob
import os, re, warnings
from ase.data import atomic_numbers
from pinn.io.base import list_loader
from scipy.io import FortranFile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pol_models_ewald import *
from pol_utils_ewald import *
# import csv
# import numpy
import time
import math

n_elec_check = 10 # this is fixed in the code, because this is the value that MW uses 
                  # to check the position of the electrode atoms

ds_spec = {
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'ptensor': {'dtype': tf.float32, 'shape': [3, 3]},
    'cell':   {'dtype': tf.float32, 'shape': [3, 3]},
    'coord_check': {'dtype':  tf.float64, 'shape': [None, 3]}}

@list_loader(ds_spec=ds_spec, pbc=True)
def load_data_inpt(fname):
    """Load the data from the Metalwalls configuration file data.inpt

    Args:
       fname: data.inpt file
       
    Returns:
        coord: the coordinate of the electrode atoms
        ptensor:
        cell: cell parameters, for an orthorombic box
        coord_check: an array containing the position of the first ten electrode atoms
        
    Issues:
        For now, the runtime.inpt fiel is not read. It should be added in the future to get the tolerance and cutoff to
        compute the Ewald parameters, check the finite field type as well as the Gaussian width parameters
    """
    
    with open(fname) as run:
        for (linenum, line) in enumerate(run):
            if (line.lstrip()).startswith("num_electrode_atoms"):
                nelec = int(line.split()[1])
            if (line.lstrip()).startswith("num_atoms"):
                nat = int(line.split()[1])
            if (line.lstrip()).startswith("# box"):
                line2 = run.readline()
                cellx = float(line2.split()[0])
                celly = float(line2.split()[1])
                cellz = float(line2.split()[2])
            if (line.lstrip()).startswith("# coordinates"):
                nheader = linenum + 2

    nions=nat-nelec

    atname,x,y,z = np.loadtxt(fname, skiprows=nions+nheader, max_rows=nelec, unpack=True, dtype='U')

    x = np.asfarray(x) * 0.52917721092
    y = np.asfarray(y) * 0.52917721092
    z = np.asfarray(z) * 0.52917721092
    
    elems = [atomic_number(a) for a in atname]

    
    coord = np.column_stack((x,y,z))
    
    coord_check = coord [:n_elec_check,:] / 0.52917721092

    pol = np.zeros((3,3))

    
    cellx = cellx * 0.52917721092
    celly = celly * 0.52917721092
    cellz = cellz * 0.52917721092
    cell = [[cellx,0,0],[0,celly,0],[0,0,cellz]]

    applied_D = [False,False,False]
    
    return {'coord': coord, 'elems':elems, 'ptensor': pol, 'cell': cell, 'coord_check': coord_check}

def atomic_number(atomic_name):
    """Returns the atomic number list associated with the electrode atoms

    Args:
       atomic_name: list of strings corresponding to the atom names in the data.inpt file.
       
    Returns:
        atomic_number: a list of atomic numbers corresponding to the electrode atoms
       
    Issue:
        At the moment, I use the first string of the atomic name to get the atomic element because 1 electrode = 1 species and 1 species = 1 name, but I don't find that satisfying
    """
    
    if atomic_name.startswith("C"):
        atnumber = 6
    if atomic_name.startswith("N"):
        atnumber = 7
    if atomic_name.startswith("O"):
        atnumber = 8
    if atomic_name.startswith("H"):
        atnumber = 1
    if atomic_name.startswith("S"):
        atnumber = 16

    return atnumber

def load_runtime_inpt(fname):
    """Reads in the runtime file if the electric displacement is applied and in which direction

    Args:
       fname: path and name of the runtime.inpt file
       
    Returns:
        applied_D: a list of logical variable saying if the electric displacement is applied in the x, y and z directions respectively
       
    Issue:
        Under construction, will go back to it after the main issues are fixed. Potential improvement: use the python interface of Metalwalls?
    """
    field_param = ''
    with open(fname) as run:
        for (linenum, line) in enumerate(run):
            # if (line.lstrip()).startswith("external_field"):
            #     field_param = np.loadtxt(fname, skiprows=linenum+1, unpack=True, dtype='U')
            if (line.lstrip()).startswith("num_pbc"):
                pbc = int(line.split()[1])
            if (line.lstrip()).startswith("coulomb_rtol"):
                rtol = float(line.split()[1])
            if (line.lstrip()).startswith("coulomb_rcut"):
                rcut = float(line.split()[1])
            if (line.lstrip()).startswith("coulomb_ktol"):
                ktol = float(line.split()[1])
#    print(field_param)
#    field_type = 'E'
#    field_direction = [0,0,0]
#    for i in range(len(field_param[:,0])):
#        if field_param[i,0] == 'type':
#            field_type=field_param[i,1]
#        if field_param[i,0] == 'direction':
#            field_direction = field_param[i,1:3]
    
#    applied_D = [False,False,False]
#    for i in range(3):
#        if field_type=='D' and not field_direction[i]==0:
#            applied_D[i]=True
        
    return pbc,rcut,rtol,ktol

def get_Ewald_parameters(box,rcut,rtol,ktol):
    
    """Computes the Ewald summation parameters from the simulation cell parameters

    Args:
       box: list of strings corresponding to the atom names in the data.inpt file.
       
    Returns:
        eta: Gaussian width used for the Ewald summation
        rcut: cutoff for electrostatic interactions
        kmax: maximum number of k vectors
       
    Issue:
        At the moment, the cutoff is chosen as half of the smallest box dimension, and not taken from the Metalwalls input.
        The maximum number of k vectors is the same in all the directions, even though the box is not cubic.
        The Gaussian width is derived from a given tolerance, this should be read from the Metalwalls input
    """
    
    L = box        # box dimensions
    
    V = L[0,0]*L[1,1]*L[2,2]
    acc = math.sqrt(-math.log(rtol)) # Desired accuracy
    c = np.cbrt(1/(V))
    eta = (1/(math.sqrt(2*math.pi)*c))/10.
    rcut = acc*math.sqrt(2)*eta
    kcut = math.sqrt(2)*acc/eta
    kmax = np.math.ceil(kcut*np.amax(L)/(2*math.pi))

    rcut =np.amin([L[0,0],L[1,1],L[2,2]])/2.
    eta = rcut / math.sqrt(-math.log(rcut*ktol)*2)
    kmax = np.math.ceil(math.sqrt(-math.log(ktol)*2)*rcut/(eta*math.pi))
    
    return eta, kmax

def check_existence(path_to_files,data_file,runtime_file,model_list,path_to_models):

    """ Check the existence of the different inputs: data.inpt file, models used

    Args:
       path_to_files: path to input (data.inpt and runtime.inpt) and output (pinnwall.out and hessian_matrix.inpt)
       data_file: data.inpt file
       runtime_file: runtime.inpt file
       model_list: list of model used to compute the CRK
       path_to_models: path to the directory where the ML models are stored
       
    Returns:
        If a required file does not exist, an error message is written in pinnwall.out and the execution stops
        If a model in model_list is not supported by PiNN, an error message is written in pinnwall.out and the execution stops
    """
    
    default_list = ['acks2','eem','etainv','local']
    
    if not (os.path.isfile(data_file)):
        fname = path_to_files + '/pinnwall.out'
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(path_to_files))
        output.write("ERROR : data.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
        
    if not (os.path.isfile(runtime_file)):
        fname = path_to_files + '/pinnwall.out'
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(path_to_files))
        output.write("ERROR : runtime.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
    
    for model in model_list:
        if not (model in default_list):
            fname = path_to_files + '/pinnwall.out'
            output = open(fname, 'w')
            output.write("PiNNWALL started\n\n")
            output.write("Working directory :\n")
            output.write("{0:50s}\n".format(path_to_files))
            output.write("ERROR : the following model is not supported by PiNN\n")
            output.write("{0:8s} ".format(model))
            output.write("List of models supported by PiNN:\n")
            for CDFT_method in model_list:
                output.write("{0:8s} ".format(CDFT_method))
                output.write("\n")
            output.write("exit\n")
            exit()
            
def main(args):
    path_to_models = args.prefix_pinn_model
    path_to_files = args.inputs_dir
    model_list = args.models
    
    if os.path.isabs(args.output_log):
        # Output is a full path, use it as-is
        output_fname = args.output_log
    else:
        # Output is just a filename, write in the current path
        current_path = os.getcwd()
        output_fname = os.path.join(current_path, args.output_log)
        
    data_file = os.path.join(path_to_files,'data.inpt')
    runtime_file = os.path.join(path_to_files,'runtime.inpt')
    filelist = glob(data_file)
    check_existence(path_to_files,data_file,runtime_file,model_list,path_to_models)
    dataset = lambda: load_data_inpt(filelist)
    box = np.float64(next(dataset().as_numpy_iterator())['cell']) * 1.88973
    pbc,rcut,rtol,ktol = load_runtime_inpt(runtime_file)
    eta, kmax = get_Ewald_parameters(box,rcut,rtol,ktol)

    output = open(output_fname, 'w')
    output.write("PiNNWALL started\n\n")
    output.write("Working directory :\n")
    output.write("{0:50s}\n".format(path_to_files))
    output.write("Models used : ")

    for CDFT_method in model_list:
        output.write("{0:8s} ".format(CDFT_method))
    output.write("\n")
    output.write("Ewald cutoff {0:8.3f}\n".format(rcut * 0.52917721092))
    output.write("Eta {0:8.3f}\n".format(eta))
    output.write("Maximum number of k points {0:d}\n\n".format(kmax))
    for CDFT_method in model_list:
        tmodel0 = time.time()
        output.write("Start model {0:8s}\n".format(CDFT_method))
        model_choice = os.path.join(path_to_models, '*'+CDFT_method+'*')
        # model_choice = path_to_models + '/*' + CDFT_method + '*'
        #
        avg_chi = []
        for m in glob(model_choice):
            model = get_model(m)
            params = model.params.copy()
            params['model']['params'].update(ewald_rc=rcut, ewald_kmax=kmax, ewald_eta=eta)
            model = get_model(params)
            pred = [out for out in 
                    model.predict(lambda: dataset().apply(sparse_batch(1)))]

            for c, prediction in enumerate(pred):
                mat_chi = prediction['chi']
                avg_chi.append(mat_chi)

        average_chi = np.float64(np.average(avg_chi, axis=0))
        
        # Future improvement: Move the writing of the file in its own function?
        coord_check = np.float64(next(dataset().as_numpy_iterator())['coord_check'])
        
        fname = path_to_files + '/hessian_matrix_' + CDFT_method + '.inpt'
        # If the model list contains only one element, the matrix file is named data.inpt, otherwhise the model is specified in the file name
        if len(model_list) == 1:
            fname = path_to_files + '/hessian_matrix.inpt'
        # The matrix file read by Metalwalls is a binary file
        output.write("Generate CRK file :\n")
        output.write("{0:50s}\n".format(fname))
        f = FortranFile(fname, 'w')
        f.write_record(n_elec_check)
        f.write_record(coord_check.T)
        f.write_record(-average_chi.T)
        f.close()
        tmodel1 = time.time()
        output.write("End model {0:8s}\n\n".format(CDFT_method))

    output.write("\n End of PiNNWALL")
    output.close()