import MutationModule as Mut
import os
import sys
import pickle
import numpy as np

pH = float(sys.argv[1])
chain = sys.argv[2]
resid = sys.argv[3]
res_org = sys.argv[4]
res_mut = sys.argv[5]

DirName = '.'
Name = 'prot_'+chain+'_wild_'+str(resid)+'_'+str(pH)
ResNum = int(resid)
wild = Mut.Protein(DirName+'/', DirName+'/', Name[0:11], Name[0:11]+'_'+str(pH), Name[0:11], Name[0:11], Name[5], ResNum, pH)
wild.setup_basic()
wild.construct_index_list()
wild.setup_pairwise_interaction()
wild.construct_feature()
wild.get_pka_info()
wild.construct_environment_feature()

DirName = '.'
Name = 'prot_'+chain+'_'+res_org+resid+res_mut+'_mut'
mute = Mut.Protein(DirName+'/', DirName+'/', Name, Name, Name, Name, Name[5], ResNum, pH)
mute.setup_basic()
mute.construct_index_list()
mute.setup_pairwise_interaction()
mute.construct_feature()
mute.get_pka_info()

Feature = Mut.construct_mutation_feature(wild, mute)

OutFile = open('X.npy', 'w')
np.save(OutFile, Feature)
OutFile.close()

OutFile = open('wild.pkl', 'w')
pickle.dump(wild, OutFile)
OutFile.close()

OutFile = open('mute.pkl', 'w')
pickle.dump(mute, OutFile)
OutFile.close()
