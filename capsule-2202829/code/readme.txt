Read me

TopNetTree is a software package which calculate the binding affinity change upon mutation. The package is mainly based on matlab javaplex, R-TDA and python scikit-learning/Keras package. Training set of the model is AB-Bind 645 set.

1. Install
TopNetTree requires matlab and VMD pre installed on the root folder.

For python package:
pip install scikit-learn
pip install keras

For R-tda package:
install.packages(pkgs = "TDA")

Other softwares required for the package are contained in the data/software folder. Large model parameters also saved in data folder. 

2. Examples
Input of the package contains: protein complex structure, chain id, wildtype, mutatnt type, residue id and feature option
For feature option, 0 means all features included in the model (which may requires more time), 1 means only topological features included in the model (which is much faster)

The script will return a *.ddg file contains the predicted binding affinity change upon mutation in kcal/mol.

An example of 1AK4 D A 488 G would be:

cp example_structure/1AK4.pdb .
./run_topnettree.sh 1AK4.pdb D A G 488 0

The software will return an output file "1AK4_D_A_488_G.ddg" contains the energy change.

	
