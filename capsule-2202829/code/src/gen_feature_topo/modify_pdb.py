import os 
import sys 
import numpy as np

def modifypdb(name1,name2):
        pdbfile = open(name1)
	outfile = open(name2,"w")
        lines = pdbfile.read().splitlines()
        PRO = []
        for line in lines:
                if line[0:4] == 'ATOM':
			outfile.write(line+'\n')

	outfile.close()
                        
        return 

modifypdb(sys.argv[1],sys.argv[2])

