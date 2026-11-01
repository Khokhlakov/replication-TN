import os
import sys

pdbid = sys.argv[1]
chainid = sys.argv[2]
resid = sys.argv[3]
wildname = sys.argv[4]
resname = sys.argv[5]


os.system('mkdir vmd_workdir')
command = 'cp ../../input_structure/' + pdbid +'.pdb ./vmd_workdir/prot.pdb'
os.system(command)
#os.system('cp prot.pdb ./vmd_workdir/')
tclfile = open('vmd_workdir/vmd.tcl', 'w')
tclfile.write("mol new {vmd_workdir/prot.pdb} type {pdb} first 0 last 0 step 1 waitfor 1\n")
tclfile.write('set prot [atomselect top "protein and chain '+chainid+'"]\n')
tclfile.write('$prot writepdb vmd_workdir/protonly.pdb\n')
tclfile.write('exit')
tclfile.close()
os.system('vmd -dispdev text -e vmd_workdir/vmd.tcl')
os.system('mv vmd_workdir/protonly.pdb vmd_workdir/prot_'+chainid+'_wild_tmp.pdb')
os.system('cp vmd_workdir/prot_'+chainid+'_wild_tmp.pdb ./')
os.system('rm vmd_workdir/*.pdb vmd_workdir/vmd.tcl')

#os.system('cp ../../jackal.dir ./')
os.system('./profix -fix 0 prot_'+chainid+'_wild_tmp.pdb')
os.system('mv prot_'+chainid+'_wild_tmp_fix.pdb prot_'+chainid+'_wild.pdb')

scapfile = open('./tmp_scap.list', 'w')
scapfile.write(chainid+','+resid+','+resname)
scapfile.close()
os.system('./scap -ini 20 -min 4 ./prot_'+chainid+'_wild.pdb ./tmp_scap.list')
os.system('mv prot_'+chainid+'_wild_scap.pdb prot_'+chainid+'_'+wildname+resid+resname+'_mut.pdb')

# os.system('cp prot_'+chainid+'_'+wildname+resid+resname+'_mut.pdb mute.pdb')
# os.system('cp prot_'+chainid+'_wild.pdb wild.pdb')
