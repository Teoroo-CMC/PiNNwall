from ase.data import atomic_numbers, atomic_names
import re

atom_names = ['C', 'C1', 'Ca', 'CA','Cablue', 'Cb', 'CH', 'C_b', 'C_alpha']

def generate_elem_num(name):
    potential_atom = re.split('[^A-Za-z]',name)[0]

    try:
        atomic_numbers[potential_atom[:2].capitalize()]

    except:
        element = atomic_numbers[potential_atom[0]]

    else:
        element = atomic_numbers[potential_atom[:2].capitalize()]

    return element
    
for name in atom_names:
    print(generate_elem_num(name))
