#! /bin/bash
# Usage:
# ./run.sh $pdbid $chainid $resid $wildname $mutname $cutoff 


python gen_mut_structure.py $1 $2 $3 $4 $5 $6
python Gen_pointclouds.py $1 $2 $3

cp Construct_jplex_binding.m binding_PH
cp java.opts binding_PH
cd binding_PH 
matlab -nodisplay -r "run Construct_jplex_binding.m"
cd ..

cp Construct_jplex_mutation.m mutation_PH
cp java.opts mutation_PH
cd mutation_PH
matlab -nodisplay -r "run Construct_jplex_mutation.m"
cd ..

python Construct_Feature_tda.py
python gen_cnn_feature.py
python gen_feature_12.py
