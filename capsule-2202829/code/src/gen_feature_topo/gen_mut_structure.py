# cut_structure.py
# Use vmd tcl scripts to cull original PDB files

import os
import sys 


def input_para():
    pdbid = sys.argv[1]
    chainid = sys.argv[2]
    resid = sys.argv[3]
    wildname = sys.argv[4]
    mutname = sys.argv[5]
    cutoff = sys.argv[6]
    return pdbid,chainid,resid,wildname,mutname,cutoff

# Generate Mutant chain using scap

def Gen_mut(workdir,pdbid,chainid,resid,mutname):
    scapfile = open('./tmp_scap.list', 'w')
    scapfile.write(chainid+','+resid+','+mutname)
    scapfile.close()
    filename = workdir+'/'+ pdbid+'.pdb'
    os.system('./scap -ini 20 -min 4 '+ filename +' ./tmp_scap.list')
    os.system('mv '+pdbid+'_scap.pdb '+ workdir+'/'+pdbid+'_mut.pdb')
    os.system('rm ./tmp_scap.list')
    return

# fix pdb structure

def fix_structure(filepath):
    os.system('./profix -fix 0 '+'/'+filepath+'/complex.pdb')
    os.system('mv complex_fix.pdb complex.pdb')
    os.system('mv complex.pdb '+filepath+'/complex.pdb')
    return

# tcl scripts to define binding site and mutation site with cutoff values

def def_site(filepath,name,chainid,resid,cutoff):
    tclname = filepath +'/def_site.tcl'
    filename = filepath +'/'+ name +'.pdb'
    b_name = name +'_b.pdb'
    m_name = name +'_m.pdb'
    tclfile = open(tclname,'w')
    tclfile.write("mol new {" + filename +"} type {pdb} first 0 last 0 step 1 waitfor 1\n")
    tclfile.write('set prot [atomselect top "within '+ str(cutoff) + ' of chain '+chainid+'"]\n')
    tclfile.write('$prot writepdb '+ b_name +'\n')
    tclfile.write('set prot2 [atomselect top "within '+ str(cutoff) + ' of resid '+resid+' and chain '+chainid+'"]\n')
    tclfile.write('$prot2 writepdb '+ m_name +'\n')
    tclfile.write('exit')
    tclfile.close()
    os.system('vmd -dispdev text -e '+ tclname)
    os.system('mv *.pdb '+ filepath)
    return

# Main function
def main():
    Datadir = '../../input_structure/'
    workdir = 'pdbfile'
    os.system('rm -r pdbfile')
    if not os.path.exists(workdir):
        os.system('mkdir '+ workdir)
        pdbid,chainid,resid,wildname,mutname,cutoff = input_para()
        command = 'cp '+ Datadir + pdbid +'.pdb '+ './'+ workdir 
        os.system(command)
    else:
        return
    os.system("python modify_pdb.py ./" +workdir + '/'+pdbid + '.pdb '+ workdir+'/complex.pdb')
    fix_structure(workdir)
    os.system("mv "+workdir+'/complex.pdb ' + workdir +'/'+pdbid+'.pdb ')
    Gen_mut(workdir,pdbid,chainid,resid,mutname)
    def_site(workdir,pdbid,chainid,resid,cutoff)
    def_site(workdir,pdbid+'_mut',chainid,resid,cutoff)
    return

if __name__ == "__main__":
    main()

