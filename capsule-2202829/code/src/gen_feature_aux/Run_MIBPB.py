import os
import sys

pH = float(sys.argv[1])
chain = sys.argv[2]
resid = sys.argv[3]
res_org = sys.argv[4]
res_mut = sys.argv[5]

pdb2pqr_path = '../../../../data/software/pdb2pqr-linux-bin64-2.1.1/'
pdb_path = './'
pdb_out_path = './'

pdbfile = 'prot_'+chain+'_wild.pdb'
pqrfile = 'prot_'+chain+'_wild_'+str(pH)+'.pqr'
os.system(pdb2pqr_path+'pdb2pqr --ff=amber --ph-calc-method=propka --with-ph='+str(pH)+' '+pdb_path+pdbfile+' '+pdb_out_path+pqrfile)
os.system('cp '+pqrfile+' wild.pqr')

pdbfile = 'prot_'+chain+'_'+res_org+resid+res_mut+'_mut.pdb'
pqrfile = 'prot_'+chain+'_'+res_org+resid+res_mut+'_mut.pqr'
os.system(pdb2pqr_path+'pdb2pqr --ff=charmm --ph-calc-method=propka --with-ph='+str(pH)+' '+pdb_path+pdbfile+' '+pdb_out_path+pqrfile)
os.system('cp '+pqrfile+' mute.pqr')

os.system('./mibpb5 wild h=0.5')
name = 'prot_'+chain+'_wild_'+str(pH)
os.system('mv partition_area.txt '+name+'.arealist')
os.system('mv area_volume.dat '+name+'.areavolume')
os.system('mv wild.eng '+name+'.eng')
os.system('mv wild.englist '+name+'.englist')

os.system('./mibpb5 mute h=0.5')
name = 'prot_'+chain+'_'+res_org+resid+res_mut+'_mut'
os.system('mv partition_area.txt '+name+'.arealist')
os.system('mv area_volume.dat '+name+'.areavolume')
os.system('mv mute.eng '+name+'.eng')
os.system('mv mute.englist '+name+'.englist')
