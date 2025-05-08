""""PiNNwall functions and helper functions

This script contains all the helper functions used by PiNNwall as well as the main function.
It also defines the environment variables used to run PiNNwall, 
e.g. whether it runs on the CPU or GPU.

This can be changed by adjusting the 'os.environ['CUDA_VISIBLE_DEVICES']' variable.

See the individual functions for further documentation, where the most important 
one is the main function found at the end of the file.
"""

from pinn.io import sparse_batch
from glob import glob
import os
#from ase.data import atomic_numbers
from pinn.io.base import list_loader
from scipy.io import FortranFile
from pol_models_ewald import *
from pol_utils_ewald import *
import time
import math
import re

# Sets the environment to use the first GPU device for prediction, can be set to '' to run on CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# This is a fixed value that MW uses to check the position of the electrode atoms
n_elec_check = 10 
bohr_to_angstrom = 0.52917721092

# Judge if a float is zero
is_float_zero = lambda value, tolerance=1e-6: abs(value) < tolerance

# Define a datastructure for reading the data.inpt file 
ds_spec = {
    'elems': {'dtype':  tf.int32,   'shape': [None]},
    'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
    'ptensor': {'dtype': tf.float32, 'shape': [3, 3]},
    'cell':   {'dtype': tf.float32, 'shape': [3, 3]},
    'coord_check': {'dtype':  tf.float64, 'shape': [None, 3]}}

# Decorator to ensure compatibility with PiNN, defines the data to be a periodic structure
@list_loader(ds_spec=ds_spec, pbc=True)
def load_data_inpt(fname):
    """Load the data about the electrode from the Metalwalls configuration file data.inpt

    Args:
       fname: data.inpt file
       
    Returns:
        coord: the coordinate of the electrode atoms
        ptensor:
        cell: cell parameters, for an orthorombic box
        coord_check: an array containing the position of the first ten electrode atoms
        
    Issues:
        For now, the runtime.inpt file is not read. It should be added in the future to get the tolerance and cutoff to
        compute the Ewald parameters, check the finite field type as well as the Gaussian width parameters
    """
    
    # 1 a.u. = 0.52917721092 Angstrom
    # All outputs are converted to Angstron 
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
    # Convert the coordinates from Bohr to Angstrom, since prediction uses Angstrom as unit
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
        At the moment, the first string of the atomic name to get the atomic element is used because 1 electrode = 1 species and 1 species = 1 name
        This could cause issues in the future, if expanded to different elements
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
    """Extracts the external field information from runtime.inpt

    Args:
       fname: runtime.inpt file

    Returns:
        dict:
            Type: Whether applied field is E or D type
            Direction: Direction of applied field
            Amplitude: Applied field strength
        None:
            returns None is external field block is not present in runtime.inpt 
        
    Example usage:

    fname = './runtime.inpt'
    info = extract_external_field_info(fname)
    if info:
        print(f'Type: {info["Type"]}')
        print(f'Direction: {info["Direction"]}')
        print(f'Amplitude: {info["Amplitude"]}')
    else:
        print('External field information not found.')
    """
    # Read the text file and extract the relevant lines
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
        # Return None if external field block is not defined
        return None

def load_runtime_inpt(fname):
    """Reads in the runtime file if the electric displacement is applied and in which direction

    Args:
       fname: path and name of the runtime.inpt file
       
    Returns:
        applied_D: a list of logical variable saying if the electric displacement is applied in the x, y and z directions respectively
       
    Issue:
        Under construction, will go back to it after the main issues are fixed.
    """
    with open(fname) as run:
        for (linenum, line) in enumerate(run):
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
       rtol: tolerance for the Ewald summation
       ktol: tolerance for the Ewald summation
       
    Returns:
        alpha: Gaussian_width used for the Ewald summation
        kmax: maximum number of k vectors in three directions used for the Ewald summation
        kmax_x: maximum number of k vectors in x-direction
        kmax_y: maximum number of k vectors in y-direction
        kmax_z: maximum number of k vectors in z-direction
        rkmax: computed rkmax
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
    """Check the existence of the different inputs: data.inpt file, models used

    Args:
       fname: output filename 
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
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(fname))
        output.write("ERROR : data.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
        
    if not (os.path.isfile(runtime_file)):
        output = open(fname, 'w')
        output.write("PiNNWALL started\n\n")
        output.write("Working directory :\n")
        output.write("{0:50s}\n".format(fname))
        output.write("ERROR : runtime.inpt file does not exist in working directory\n")
        output.write("exit\n")
        raise SystemExit("Execution ended with error")
    
    for model in model_list:
        if not (model in default_list):
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

def get_electrode_species_name(fname):
    """Returns the names of the electrode species

    Args:
       fname: data.inpt file

    Returns:
        atname: list of names of the electrode species
    """
    with open(fname) as run:
        for (linenum, line) in enumerate(run):
            if (line.lstrip()).startswith("num_electrode_atoms"):
                nelec = int(line.split()[1])
            if (line.lstrip()).startswith("num_atoms"):
                nat = int(line.split()[1])
            if (line.lstrip()).startswith("# box"):
                line2 = run.readline()
            if (line.lstrip()).startswith("# coordinates"):
                nheader = linenum + 2

    nions=nat-nelec
    
    atname,x,y,z = np.loadtxt(fname, skiprows=nions+nheader, max_rows=nelec, unpack=True, dtype='U')
    return list(set(atname))

def update_gaussian_width(fname,fout, species_names, eta_dict):
    """Returns the atomic name associated with the given atomic number

    Args:
       fname: runtime.inpt file to be read
       fout: name of updated runtime.inpt file to be written
       species_names: names of electrode species
       eta_dict: dictionary of eta parameters used by PiNN to predict CRK

    Returns:
        writes out updated runtime.inpt file with updated Gaussian widths
    """
    with open(fname, 'r') as file:
        data = file.readlines()
    updated_data = []
    data_iter = iter(data)
    is_target_block = False
    for line in data_iter:
        line_strip = line.strip()
        if line_strip.startswith('species_type'):
            name_line = next(data_iter)  # Read the next line to check the name
            name = name_line.split()[1]  # Extract the name from the line
            if name in species_names:
                is_target_block = True
                updated_data.extend([line, name_line])
            else:
                is_target_block = False
                updated_data.extend([line, name_line])
        elif is_target_block:
            if line_strip.startswith('charge gaussian'):
                line_split = line_strip.split()
                line_split[2] = str(eta_dict[atomic_number(name[0])])
                updated_data.append('\t\t' + ' '.join(line_split))
                updated_data.append('\n')  # Add an empty line after the updated line
            else:
                updated_data.append(line)
        else:
            updated_data.append(line)
    
    # Write the updated data to the file
    with open(fout, 'w') as file:
        file.writelines(updated_data)
         
def main(args):
    """Main PiNNwall function, reads in electrode structure and predicts CRK and creates updated runtime.inpt

    Args:
       path_pinn_model: path to the train models to be used for prediction
       inputs_dir: path to the Metalwalls input files
       models: list of model types to be used, e.g. ['eem'] or ['eem', 'acks2'] 
       output_log: filename of output log file that will be written

    Returns:
        hessian_matrix.inpt: written file containing predicted charge response kernel in Metalwalls format
        hessian_matrix.out: written file containing human-readable predicted charge response kernel
        runtime_<method_name>.inpt: a modified version of the provided runtime.inpt file, that containing updated Gaussian width and electrostatics parameters
        pinnwall.out: text file containing the parameters used for this run
    """
    # Parse arguments
    path_to_models = args.path_pinn_model
    path_to_files = args.inputs_dir
    model_list = args.models
    
    if os.path.isabs(args.output_log):
        # Output is a full path, use it as-is
        output_fname = args.output_log
    else:
        # Output is just a filename, write in the working path
        output_fname = os.path.join(path_to_files, args.output_log)
    
    # read data.inpt and get Ewald parameters from runtime.inpt 
    data_file = os.path.join(path_to_files,'data.inpt')
    runtime_file = os.path.join(path_to_files,'runtime.inpt')
    filelist = glob(data_file)
    
    check_existence(output_fname,data_file,runtime_file,model_list)
    
    dataset = lambda: load_data_inpt(filelist)
    box = np.float64(next(dataset().as_numpy_iterator())['cell']) / bohr_to_angstrom
    box_volume = np.linalg.det(box)
    
    electrode_species = get_electrode_species_name(data_file)
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
    
    # check whether simulation is run under field or external potential
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

    # compute the average chi for each model
    for CDFT_method in model_list:
        output.write("Start model {0:8s}\n".format(CDFT_method))
        model_choice = os.path.join(path_to_models, '*'+CDFT_method+'*')
        
        avg_chi = []
        avg_sigma_e = []
        for m in glob(model_choice):
            print(m)
            model = get_model(m)
            params = model.params.copy()
            params['model']['params'].update(ewald_rc=rcut, ewald_kmax=[kmax_x,kmax_y,kmax_z], ewald_eta=eta)
            model = get_model(params)
            pred = [out for out in 
                    model.predict(lambda: dataset().apply(sparse_batch(1)))]
            atom_types = params['network']['params']['atom_types']
            for c, prediction in enumerate(pred):
                mat_chi = prediction['chi']
                avg_chi.append(mat_chi)
                if CDFT_method == 'eem' or CDFT_method == 'acks2':
                    sigma_e = prediction['sigma_e']
                    avg_sigma_e.append(sigma_e)

        average_chi = np.float64(np.average(avg_chi, axis=0))
        if CDFT_method == 'eem' or CDFT_method == 'acks2':
            average_sigma_e = np.float64(np.average(avg_sigma_e, axis=0)) # here the Unit is Angstrom
            # convert to MW's electrode gaussian parameter, which is in 1/bohr and need to scale by 1/sqrt(2)
            eta_e = 1/(np.sqrt(2)*average_sigma_e/bohr_to_angstrom)          
            eta_e = {k: v for k, v in zip(atom_types, eta_e)}
            
            # update gaussian widths in the runtime file
            update_gaussian_width(runtime_file,os.path.join(path_to_files,'runtime_'+CDFT_method+'.inpt'),
                                  electrode_species, eta_e)
   
        # To match with MW hessian matrix format
        inv_matrix = -average_chi
                
        # write hessian matrix file
        # Future improvement: Move the writing of the file in its own function?
        coord_check = np.float64(next(dataset().as_numpy_iterator())['coord_check']) / bohr_to_angstrom
        
        # Check if the number of rows is less than n_ele_check
        ### I am not sure about how MW handles this, so I just print a warning here
        if coord_check.shape[0] < n_elec_check:
            output.write("WARNING : The number of rows in coord_check is less than 10 \n")
            output.write("          This may cause an error in Metalwalls \n")
            output.write("          Please check the coord_check file \n")
            raise SystemExit("Execution ended with error")

        
        # If the model list contains only one model type, the matrix file is named data.inpt, otherwise the model is specified in the filename
        if len(model_list) == 1:
            fname = os.path.join(path_to_files, 'hessian_matrix.inpt')

        else:
            fname = os.path.join(path_to_files, 'hessian_matrix_' + CDFT_method + '.inpt')
        
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
        
        output.write("End model {0:8s}\n\n".format(CDFT_method))

    output.write("\n End of PiNNWALL")
    output.close()
