import numpy as np
import sys,os


pdbid = sys.argv[1]
chainid = sys.argv[2]
wildname = sys.argv[3]
resname = sys.argv[4]
resid = sys.argv[5]


PH = '7.0'
foldername = './feature/'+ pdbid +'_'+chainid+'_'+wildname+'_'+resid+'_'+resname


os.system('mkdir '+ foldername)
os.system('cp ./gen_feature_aux/PrepareStructure.py '+ foldername)
os.system('cp ./gen_feature_aux/Construct_Feature.py '+ foldername)
os.system('cp ./gen_feature_aux/mibpb5 '+ foldername)
os.system('cp ./gen_feature_aux/MS_Intersection '+ foldername)
os.system('cp ./gen_feature_aux/Run_MIBPB.py '+ foldername)
os.system('cp ./gen_feature_aux/MutationModule.py '+ foldername)
os.chdir(foldername)
os.system('python PrepareStructure.py '+pdbid+' '+ chainid+' '+ resid+ ' ' + wildname +' '+resname)
os.system('python ./Run_MIBPB.py '+ PH +' '+ chainid+' '+resid+' '+ wildname +' '+resname)
os.system('python ./Construct_Feature.py '+ PH +' '+ chainid+' '+resid+' '+ wildname +' '+resname)




