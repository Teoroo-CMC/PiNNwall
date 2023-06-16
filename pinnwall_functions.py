from pinn.io import sparse_batch
from glob import glob
import os
from ase.data import atomic_numbers
from pinn.io.base import list_loader
from scipy.io import FortranFile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pol_models_ewald import *
from pol_utils_ewald import *
import time
import math
import re

# This is a fixed value that MW uses to check the position of the electrode atoms
n_elec_check = 10 
bohr_to_angstrom = 0.52917721092

# Judge if a float is zero
is_float_zero = lambda value, tolerance=1e-6: abs(value) < tolerance

# Define a datastructure for reading the data.inpt file
### CAN I USE FLOAT64 FOR ALL THE FLOATS? 
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
    
    # 1 a.u. = 0.52917721092 Angstrom
    ### outputs all are converted to Angstron 
    conversion_factor = 0.52917721092
    
    with open(fname) as run:
        for (linenum, line) in enumerate(run):
            if (line.lstrip()).startswith("num_electrode_atoms"):
                nelec = int(line.split()[1])
            if (line.lstrip()).startswith("num_atoms"):
                nat = int(line.split()[1])
            if (line.lstrip()).startswith("# box"):
                line2 = run.readline()
                # Need to use double precision here, otherwise meaningless to use float64 in main function
                cellx = np.float64(line2.split()[0]) * conversion_factor
                celly = np.float64(line2.split()[1]) * conversion_factor
                cellz = np.float64(line2.split()[2]) * conversion_factor
            if (line.lstrip()).startswith("# coordinates"):
                nheader = linenum + 2

    nions=nat-nelec
    
    atname,x,y,z = np.loadtxt(fname, skiprows=nions+nheader, max_rows=nelec, unpack=True, dtype='U')

    # Convert the coordinates from Bohr to Angstrom, this is becasue the prediction uses Angstrom as unit
    x = np.asfarray(x) * conversion_factor
    y = np.asfarray(y) * conversion_factor
    z = np.asfarray(z) * conversion_factor
    
    elems = [atomic_number(a) for a in atname]

    coord = np.column_stack((x,y,z))
    
    coord_check = coord [:n_elec_check,:] 

    pol = np.zeros((3,3))
     
    cell = [[cellx,0,0],[0,celly,0],[0,0,cellz]] 
    
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

def extract_external_field_info(fname):
    """ Example usage:
    fname = './runtime.inpt'
    info = extract_external_field_info(fname)
    if info:
        print(f'Type: {info["Type"]}')
        print(f'Direction: {info["Direction"]}')
        print(f'Amplitude: {info["Amplitude"]}')
    else:
        print('External field information not found.')
    """
    # "Read the text file and extract the relevant lines
    with open(fname, 'r') as file:
        lines = file.readlines()

    # Find the index of the line containing "external_field"
    external_field_index = None
    for i, line in enumerate(lines):
        if 'external_field' in line:
            external_field_index = i
            break

    # Extract information starting from the line after "external_field"
    if external_field_index is not None:
        # Define the regular expressions to match the desired patterns
        type_regex = r'type\s+([A-Za-z]+)'
        direction_regex = r'direction\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)'
        amplitude_regex = r'amplitude\s+([\d.-]+)'

        # Initialize variables to store the extracted values
        type_value = None
        direction_values = None
        amplitude_value = None

        # Iterate over the lines after "external_field" and search for the patterns using regex
        for line in lines[external_field_index + 1:]:
            if 'type' in line:
                type_match = re.search(type_regex, line)
                if type_match:
                    type_value = type_match.group(1)
            elif 'direction' in line:
                direction_match = re.search(direction_regex, line)
                if direction_match:
                    direction_values = [float(direction_match.group(1)), float(direction_match.group(2)), float(direction_match.group(3))]
            elif 'amplitude' in line:
                amplitude_match = re.search(amplitude_regex, line)
                if amplitude_match:
                    amplitude_value = float(amplitude_match.group(1))

        # Return the extracted values
        return {
            'Type': type_value,
            'Direction': direction_values,
            'Amplitude': amplitude_value
        }
    else:
        return None

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
    
    external_field_info = extract_external_field_info(fname)   
        
    return pbc,rcut,rtol,ktol, external_field_info

def get_Ewald_parameters(box,rcut,rtol, ktol):
    
    """Computes the Ewald summation parameters from the simulation cell parameters

    Args:
       box: box dimensions
       rcut: cutoff for short-range electrostatic interactions
       ktol: tolerance for the Ewald summation
       
    Returns:
        alpha: Gaussian_width used for the Ewald summation
        kmax: maximum number of k vectors
       
    Issue:
        At the moment, the maximum number of k vectors is the same in all the directions, even though the box is not cubic.
        The Gaussian width is derived based on Metalwall input rcut and ktol.
    """
    
    L = box        # box dimensions

    alpha = rcut / math.sqrt(-math.log(rcut*rtol))
    one_over_alpha = 1.0/alpha
    
    rkmax = math.sqrt(-math.log(ktol)*4.0*one_over_alpha*one_over_alpha)

    kmax_x = np.math.floor(rkmax*L[0,0]/(math.pi*2.0))
    kmax_y = np.math.floor(rkmax*L[1,1]/(math.pi*2.0))
    kmax_z = np.math.floor(rkmax*L[2,2]/(math.pi*2.0))

    ### kmax should be different for different dimensions but here we use the same value
    kmax = np.math.ceil(max(kmax_x,kmax_y,kmax_z))

    return alpha, kmax, kmax_x, kmax_y, kmax_z, rkmax

def check_existence(fname,data_file,runtime_file,model_list):

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
        # fname = path_to_files + '/pinnwall.out'
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(fname))
        output.write("ERROR : data.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
        
    if not (os.path.isfile(runtime_file)):
        # fname = path_to_files + '/pinnwall.out'
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(fname))
        output.write("ERROR : runtime.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
    
    for model in model_list:
        if not (model in default_list):
            # fname = path_to_files + '/pinnwall.out'
            output = open(fname, 'w')
            output.write("PiNNWALL started\n\n")
            output.write("Working directory :\n")
            output.write("{0:50s}\n".format(fname))
            output.write("ERROR : the following model is not supported by PiNN\n")
            output.write("{0:8s} ".format(model))
            output.write("List of models supported by PiNN:\n")
            for CDFT_method in model_list:
                output.write("{0:8s} ".format(CDFT_method))
                output.write("\n")
            output.write("exit\n")
            exit()
            
def main(args):
    # Parse arguments
    path_to_models = args.prefix_pinn_model
    path_to_files = args.inputs_dir
    model_list = args.models
    
    if os.path.isabs(args.output_log):
        # Output is a full path, use it as-is
        output_fname = args.output_log
    else:
        # Output is just a filename, write in the working path
        output_fname = os.path.join(path_to_files, args.output_log)
    
    # read data.inpt and get Ewarld parameters from runtime.inpt 
    data_file = os.path.join(path_to_files,'data.inpt')
    runtime_file = os.path.join(path_to_files,'runtime.inpt')
    filelist = glob(data_file)
    
    check_existence(output_fname,data_file,runtime_file,model_list)
    
    dataset = lambda: load_data_inpt(filelist)
    box = np.float64(next(dataset().as_numpy_iterator())['cell']) / bohr_to_angstrom
    box_volume = np.linalg.det(box)
    n_elec = len(next(dataset().as_numpy_iterator())['elems'])
    elec_xyz = np.float64(next(dataset().as_numpy_iterator())['coord'])[-n_elec:,:] / bohr_to_angstrom
    
    pbc,rcut,rtol,ktol,external_field_info = load_runtime_inpt(runtime_file)
    eta, kmax, kmax_x, kmax_y, kmax_z, rkmax = get_Ewald_parameters(box,rcut,rtol,ktol)
    
    # write output log file
    output = open(output_fname, 'w')
    output.write("PiNNWALL started\n\n")
    output.write("Working directory :\n")
    output.write("{0:50s}\n".format(path_to_files))
    output.write("Models used : ")

    for CDFT_method in model_list:
        output.write("{0:8s} ".format(CDFT_method))
    output.write("\n")
    output.write("Ewald cutoff (a.u.) {0:8.3f}\n".format(rcut))
    output.write("Alpha (a.u.^-1) {0:8.3f} \n".format(1/eta))
    output.write("Number of k points in X, Y, Z {0:d}\t{1:d}\t{2:d} \n\n".format(kmax_x, kmax_y, kmax_z))
    
    # check the type of simulation
    if external_field_info:
        external_field_type = external_field_info["Type"]
        external_field_direction = external_field_info["Direction"]
        external_field_amplitude = external_field_info["Amplitude"]
        output.write("External field type {0:8s}\n".format(external_field_type))
        output.write("External field direction {0:8.3f}\t{1:8.3f}\t{2:8.3f}\n".format \
                     (external_field_direction[0],external_field_direction[1],external_field_direction[2]))
        output.write("External field amplitude {0:8.3f}\n".format(external_field_amplitude))
    else:
        external_field_type = 'Constant Potential'
        output.write("External field type {0:8s}\n".format(external_field_type))
    
    ### for constant D = 0 and eem model. For other model, maybe not a good way to add this correction? 
    ### Or better do it to the ita_e in pinn_chi, but pinn_chi doesn't know if it is constant D = 0 or not
    if external_field_type == 'D' and 'eem' in model_list:
        if is_float_zero(external_field_amplitude):
            # compute potential felt by an electrode atom due to 
            # D = 0 external field (equals to dipole-correction)
            mat_constD = np.zeros((n_elec,n_elec),  dtype=np.float64)
            print("Constant-D = 0, no external field")
            for i in range(n_elec):
                for j in range(n_elec):
                    for ixyz in range(3):
                        if not is_float_zero(external_field_direction[ixyz]):
                            mat_constD[i,j] += 4*math.pi / box_volume * elec_xyz[i, ixyz] * elec_xyz[j, ixyz]
        else:
            output.write("ERROR : None-zero constant-D is not supported \n")
            raise SystemExit("Execution ended with error")

    # compute the average chi for each model
    for CDFT_method in model_list:
        tmodel0 = time.time()
        output.write("Start model {0:8s}\n".format(CDFT_method))
        model_choice = os.path.join(path_to_models, '*'+CDFT_method+'*')
        
        avg_chi = []
        for m in glob(model_choice):
            model = get_model(m)
            params = model.params.copy()
            params['model']['params'].update(ewald_rc=rcut, ewald_kmax=[kmax_x,kmax_y,kmax_z], ewald_eta=eta)
            model = get_model(params)
            pred = [out for out in 
                    model.predict(lambda: dataset().apply(sparse_batch(1)))]

            for c, prediction in enumerate(pred):
                mat_chi = prediction['chi']
                avg_chi.append(mat_chi)

        average_chi = np.float64(np.average(avg_chi, axis=0))
        
        ### - to match with MW hessian matrix format
        ### BUT WHY WE DO IT HERE? GUESS WE BETTER DO IT IN PINN_CHI 
        inv_matrix = -average_chi
        # matrix = np.linalg.inv(inv_matrix)
        # print(matrix)
        ### only when D = 0 and using eem model can we do dipole correction 
        ### also better do it in pinn_chi becasue it is more efficient 
        ### and can avoid matrix inversion
        if external_field_type == 'D' and CDFT_method=='eem':
            matrix = np.linalg.inv(inv_matrix)
            inv_matrix = np.linalg.inv(matrix + mat_constD)
                
        # write hessian matrix file
        # Future improvement: Move the writing of the file in its own function?
        coord_check = np.float64(next(dataset().as_numpy_iterator())['coord_check']) / bohr_to_angstrom
        
        fname = path_to_files + '/hessian_matrix_' + CDFT_method + '.inpt'
        # If the model list contains only one element, the matrix file is named data.inpt, otherwhise the model is specified in the file name
        if len(model_list) == 1:
            fname = path_to_files + '/hessian_matrix.inpt'
        
        # The matrix file read by Metalwalls is a binary file
        output.write("Generate CRK file :\n")
        output.write("{0:50s}\n".format(fname))
        # Note that data in multidimensional arrays is written in
        # row-major order --- to make them read correctly by Fortran
        # programs, you need to transpose the arrays yourself when
        # writing them.
        f = FortranFile(fname, 'w')
        f.write_record(n_elec_check)
        f.write_record(coord_check.T)
        f.write_record(inv_matrix.T)
        f.close()
        
        tmodel1 = time.time()
        output.write("End model {0:8s}\n\n".format(CDFT_method))

    output.write("\n End of PiNNWALL")
    output.close()