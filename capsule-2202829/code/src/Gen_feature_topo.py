import numpy as np
import sys,os
import csv

def main():
    
    pdbid = sys.argv[1]
    chainid = sys.argv[2]
    wildname = sys.argv[3]
    mutname = sys.argv[4]
    resid = sys.argv[5]


    foldername = './feature/'+ pdbid +'_'+chainid+'_'+wildname+'_'+resid+'_'+mutname
    os.system('mkdir '+ foldername)
    os.system('cp ./gen_feature_topo/*.py '+ foldername)
    os.system('cp ./gen_feature_topo/scap '+ foldername)
    os.system('cp ./gen_feature_topo/profix '+ foldername)
    os.system('cp ./gen_feature_topo/java.opts '+ foldername)
    os.system('cp ./gen_feature_topo/jackal.dir '+ foldername)
    os.system('cp ./gen_feature_topo/*.m '+ foldername)
    os.system('cp ./gen_feature_topo/PH_Alpha.R '+ foldername)
    os.system('cp ./gen_feature_topo/run.sh '+ foldername)

    #Run script to generate feature
    os.chdir(foldername)
    cutoff = 12.0
    os.system('./run.sh '+pdbid+' '+ chainid+' '+ resid+ ' ' + wildname +' '+mutname+' '+str(cutoff))

main()
