# Script to run the topnettree prediction model
# Usage: ./run_topnettree.sh [*.pdb] [chainid] [wildtype] [mutanttype] [resid] [feature_option]
# Output: *.ddg contains the binding affinity change upon mutation

# !/bin/bash

# Input of model
filename=$1
chainid=$2
wildtype=$3
mutanttype=$4
resid=$5
option=$6
pdbid=${filename%.pdb}

# cp structure file into src folder
mkdir src/feature
mkdir src/input_structure
cp $filename src/input_structure
cd src

# run the model
if [ $option -eq 0 ]
then
    echo "Using topo and aux features"
    python Gen_feature_topo.py $pdbid $chainid $wildtype $mutanttype $resid
    python Gen_feature_aux.py $pdbid $chainid $wildtype $mutanttype $resid    
    folder=./feature/${pdbid}_${chainid}_${wildtype}_${resid}_${mutanttype}
    echo $folder
    cp $folder/Feature_cnn.npy .
    cp $folder/X_h12.npy .
    cp $folder/X_aux.npy .

    python predict.py $option
else
    echo "Using only topo features"
    python Gen_feature_topo.py $pdbid $chainid $wildtype $mutanttype $resid
    folder=./feature/${pdbid}_${chainid}_${wildtype}_${resid}_${mutanttype}
    echo $folder
    cp $folder/Feature_cnn.npy .
    cp $folder/X_h12.npy .

    python predict.py $option
fi


# clean the intermeidate files
rm -r feature
rm -r input_structure
rm -r *.npy

mv result.ddg ../${pdbid}_${chainid}_${wildtype}_${resid}_${mutanttype}.ddg


