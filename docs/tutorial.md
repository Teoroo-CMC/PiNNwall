# Tutorial

**NOTE:** This tutorial assumes that you are able to run Metalwalls, and know how to create Metalwalls input files. For more information on this see the [Metalwalls repository](https://gitlab.com/ampere2/metalwalls), and the [Metalwalls wiki](https://gitlab.com/ampere2/metalwalls/-/wikis/home).

In this tutorial we will use PiNNwall to predict a machine learned charge response kernel which we will then use in Metalwalls to run MD simulations of this system. The system we will be looking at in this tutorial is that of a hydroxylated carbon electrode which was investigated in the following work[^1]. The Metalwalls input files used for this calculation can be found in the *examples/hydroxylated_graphene* folder in this repository.

## Metalwalls files

To get started, we will first need to have created input files for Metalwalls, namely a *data.inpt* file which contains the structure of the system, and a *runtime.inpt* which contains the simulation settings and parameters.

The *data.inpt* file should also include base charges for the electrode. Depending on the electrode structure, these could directly be taken from the PiNNwall papers[^1] [^2], predicted using PiNN, taken from force-field parameters or computed using a population analysis method. These should be placed at the end of the *data.inpt* file, using the `# base_charges :` header, e.g.

```bash
# base_charges :
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
C1        0.0000000000000
...
```

These should be ordered like the electrode atoms in the *data.inpt* file.

You can double-check that Metalwalls is able to read these, by checking if *base charges      are provided: yes* is printed in the *run.out* file.

## Preparation

### Installing PiNN and PiNNwall

The second crucial aspect is to have PiNN installed, and to have PiNNwall downloaded.

PiNN can be installed using pip or run via docker following instructions from the GitHub repository[^3].
The most straightforward way is to use docker, as this requires no further installation but simply be executed right away.

Then PiNNwall needs to be downloaded. PiNNwall consists of two scripts, *pinnwall.py* which is the main script used for execution, and *pinnwall_functions.py* which contains the PiNNwall source code. The other crucial part of PiNNwall is the *trained_models* folder, which contains the machine learning models that will be used during prediction of the charge response kernel. There are four different model types included in the *trained_models* folder. Feel free to have a look at the paper where they are introduced[^4].

### Creating the directory

Now, as a final step before running PiNNwall make sure that you create a working directory where you want to run PiNNwall. Place the Metalwalls input files, for which you would like to predict a machine learned charge response kernel, in this folder. This is both the *data.inpt* and the *runtime.inpt* files.

## Executing PiNNwall

Now it is finally time to execute PiNNwall. This can be done by writing the following command in the commandline:

```python
python pinnwall [-p <MODEL_DIR>] [-i <WORKING_DIR>] [-m [<method_name>]] [-o <filename>]
```

Here,

`-p  <MODEL_DIR> (./trained_models)`
is the path to the trained PiNN models you would like to use in prediction.

`-i <WORKING_DIR> (./)`
is the path to the Metalwalls input files, which will be used to read the electrode structure, and to update the parameters.

`-m <method_name> (eem)`
The type of models that should be used to compute the CRK to construct the Hessian Matrix which will be used by Metalwalls. To pass multiple model types, i.e. `-m eem local etainv acks2`

`-o <filename> (pinnwall.out)`
The name of the log of pinnwall containing the parameters used for this run, default to *inputs_dir/pinnwall.out*.

By default, PiNNwall will be executed by the first GPU. You can also run PiNNwall on the CPU by changing `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` to `os.environ['CUDA_VISIBLE_DEVICES'] = ''`.

If you would like to execute PiNNwall using the docker image, make sure to append `singularity exec pinn.sif` before this command, e.g.

```bash
singularity exec pinn.sif python pinnwall -p ./trained_models -i ./ -m eem
```

Executing produces:

- *hessian_matrix.inpt* - the predicted charge response kernel file to be used by Metalwalls
- *hessian_matrix.out* - a human-readable version of the charge response kernel file
- *runtime_<method_name>.inpt* - an updated version of the provided *runtime.inpt* file, which contains parameters that are consistent with those used when predicting the Hessian matrix
- *pinnwall.out* - a text file containing the parameters used for this run

## Running Metalwalls

Now we can start our simulation using the *hessian_matrix.inpt* file generated by PiNNwall. Place this file in the directory where you wish to run Metalwalls, along with the *data.inpt* file, and the *runtime.inpt* file produced by PiNNwall. The updated *runtime.inpt* file contains updated parameters to ensure consistency between the parameters used when predicting the Hessian matrix, and those used by Metalwalls. Now we run Metalwalls as you would do normally. The Metalwalls simulation should now proceed as usual.

To be sure that Metalwalls actually uses the *hessian_matrix.inpt* file generated by PiNNwall, and does not compute its own it might be useful to run a `diff` command on both files.

[^1]: Dufils, T., Knijff, L., Shao, Y., & Zhang, C. (2023). PiNNwall: Heterogeneous electrode models from integrating machine learning and atomistic simulation. *Journal of Chemical Theory and Computation, 19*(15), 5199-5209.
[^2]: Li, J., Knijff, L., Zhang, Z. Y., Andersson, L., & Zhang, C. (2025). PiNN: Equivariant Neural Network Suite for Modeling Electrochemical Systems. *Journal of Chemical Theory and Computation, 21*(3), 1382-1395.
[^3]: [PiNN GitHub repository](https://github.com/Teoroo-CMC/PiNN)
[^4]: Shao, Y., Andersson, L., Knijff, L., & Zhang, C. (2022). Finite-field coupling via learning the charge response kernel. *Electronic Structure, 4*(1), 014012.
