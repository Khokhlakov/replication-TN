import sys,os
import numpy as np

elelist = ['C','N','O']

#Run tda pacakge for alpha complex

def run_tda():

    dir1 = './mutation_PH/'
    dir2 = './binding_PH/'
	
    prefix = dir1
    for e1 in elelist:
        for e2 in elelist:
            wildpts = prefix + 'wild_mutation_'+e1+'_'+e2+'.csv'
            wildout = prefix + 'wild_mutation_'+e1+'_'+e2+'.tda'
            os.system("Rscript PH_Alpha.R "+ wildpts + " tmp_output")
            os.system("sed '1d' tmp_output > "+wildout)
			
            mutpts = prefix + 'mut_mutation_'+e1+'_'+e2+'.csv'
            mutout = prefix + 'mut_mutation_'+e1+'_'+e2+'.tda'
            os.system("Rscript PH_Alpha.R "+ mutpts + " tmp_output")
            os.system("sed '1d' tmp_output > "+mutout)

    prefix = dir2
    for e1 in elelist:
        for e2 in elelist:
            wildpts = prefix + 'wild_binding_'+e1+'_'+e2+'.csv'
            wildout = prefix + 'wild_binding_'+e1+'_'+e2+'.tda'
            os.system("Rscript PH_Alpha.R "+ wildpts + " tmp_output")
            os.system("sed '1d' tmp_output > "+wildout)

            mutpts = prefix + 'mut_binding_'+e1+'_'+e2+'.csv'
            mutout = prefix + 'mut_binding_'+e1+'_'+e2+'.tda'
            os.system("Rscript PH_Alpha.R "+ mutpts + " tmp_output")
            os.system("sed '1d' tmp_output > "+mutout)



run_tda()

