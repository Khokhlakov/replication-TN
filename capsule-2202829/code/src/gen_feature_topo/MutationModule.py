# This file contains classes and functions to construct features
import os
import numpy as np
import scipy as sp
import pickle
import operator
from scipy.spatial import cKDTree
from Bio.PDB.Polypeptide import three_to_one

def atmtyp_to_ele( st ):
    if len(st.strip()) == 1:
        return st.strip()
    elif st[0] == 'H':
        return 'H'
    elif st == "CA":
        return "CA"
    elif st == "CL":
        return "CL"
    elif st == "BR":
        return "BR"
    else:
        print st, 'Not in dictionary'
        return

def ele_to_index( st ):
    if st == 'C':
        return 0
    elif st == 'N':
        return 1
    elif st == 'O':
        return 2
    elif st == 'S':
        return 3
    elif st == 'H':
        return 4

def FRIKernel( (st, tau, pwr, ri, rj, dis) ):
    if st == 'Lorentz':
        y = 1.0/(1.0 + np.power(dis/(tau*(ri+rj)),pwr))
    elif st == 'Exp':
        y = np.exp(-np.power(dis/(tau*(ri+rj)), pwr))
    else:
        print "The input FRI kernel type is not defined."
    return y

class Protein:

    def __init__(self, InDataDir, OutDataDir, Name, PhysicsName, PDBName, SeqName, Chain, MutResID, pH):
        self.InDataDir = InDataDir
        self.OutDataDir = OutDataDir
        self.Name = Name
        self.PhysicsName = PhysicsName
        self.PDBName = PDBName
        self.SeqName = SeqName
        self.Chain = Chain
        self.MutResID = int(MutResID)
        self.pH = float(pH)
        self.AtomPos = []
        self.AtomType = []
        self.AtomVerboseType = []
        self.Charge = []
        self.ResName = []
        self.ResID = []
        self.Radii = []
        self.Area = []
        self.SolvEng = []
        self.Sequence = []
        self.MutSiteInSeq = -1 # index of the mut site in self.Sequence
        self.TotalArea = 0.0
        self.TotalVolume = 0.0
        self.TotalSolvEng = 0.0

    def setup_basic(self):
        # Info from pqr file
        PQRFile = open(self.InDataDir+self.PhysicsName+'.pqr')
        lines = PQRFile.read().splitlines()
        for line in lines:
            if line[0:4] == 'ATOM':
                self.AtomPos.append( np.array([float(line[26:38]), float(line[38:46]), float(line[46:54])]) )
                self.AtomType.append( atmtyp_to_ele(line[12:14]) )
                self.AtomVerboseType.append( line[11:17] )
                self.Charge.append( float(line[54:62]) )
                resname = line[17:20]
                if resname == 'HSE': resname = 'HIS'
                self.ResName.append( resname )
                self.ResID.append( int(line[20:26]) )
                self.Radii.append( float(line[62:69]) )
        PQRFile.close()
        # Info from arealist file
        AreaListFile = open(self.InDataDir+self.PhysicsName+'.arealist')
        lines = AreaListFile.read().splitlines()
        for line in lines:
            a,b = line.split()
            self.Area.append( float(b) )
        AreaListFile.close()
        # Info from englist file
        EngListFile = open(self.InDataDir+self.PhysicsName+'.englist')
        lines = EngListFile.read().splitlines()
        for line in lines:
            self.SolvEng.append( float(line) )
        EngListFile.close()
        # Info from areavolume file
        AreaVolumeFile = open(self.InDataDir+self.PhysicsName+'.areavolume')
        self.TotalArea = float(AreaVolumeFile.readline())
        self.TotalVolume = float(AreaVolumeFile.readline())
        AreaVolumeFile.close()
        # Info from eng file
        EngFile = open(self.InDataDir+self.PhysicsName+'.eng')
        EngFile.readline()
        self.TotalSolvEng = float(EngFile.readline())
        EngFile.close()
        # Setup sequence
        CurResID = -1000
        for i in range(0, len(self.ResID)):
            if self.ResID[i] > CurResID:
                CurResID = self.ResID[i]
                self.Sequence.append( three_to_one(self.ResName[i]) )
                if CurResID == self.MutResID:
                    self.MutSiteInSeq = len(self.Sequence)-1

    FRIDefault = [('Lorentz',0.5,5),
                  ('Lorentz',1.0,5),
                  ('Lorentz',2.0,5),
                  ('Exp',1.0,15),
                  ('Exp',2.0,15)
    ]
    def setup_pairwise_interaction(self, FRI = FRIDefault):
        self.FRI = FRI
        self.AtomNum = len(self.SolvEng)
        self.CLB = np.zeros([self.AtomNum, 5], float)
        self.VDW = np.zeros([self.AtomNum, 5], float)
        self.RIG = np.zeros([self.AtomNum, len(FRI), 5], float)
        self.CLBoppo = np.zeros([self.AtomNum, 5], float)
        self.VDWoppo = np.zeros([self.AtomNum, 5], float)
        self.RIGoppo = np.zeros([self.AtomNum, len(FRI), 5], float)
        AtomPosNP = np.asarray(self.AtomPos, float)
        t = cKDTree(AtomPosNP)
        NbLong = cKDTree.query_pairs(t,40.0)
        NbShort = cKDTree.query_pairs(t,10.0)
        # Short cutoff for VDW
        for (i,j) in NbShort:
            ri = self.Radii[i]; rj = self.Radii[j];
            posi = self.AtomPos[i]; posj = self.AtomPos[j];
            dis = np.linalg.norm(posi-posj)
            resi = self.ResID[i]; resj = self.ResID[j];
            ei = self.AtomType[i].replace(" ",""); ej = self.AtomType[j].replace(" ","");
            vdw = np.power((ri+rj)*(ri+rj)/dis, 12) - 2.0*np.power((ri+rj)*(ri+rj)/dis, 6)
            self.VDW[i, ele_to_index(ej)] += vdw
            self.VDW[j, ele_to_index(ei)] += vdw
            if bool(resi == self.MutResID) ^ bool(resj == self.MutResID):
                self.VDWoppo[i, ele_to_index(ej)] += vdw
                self.VDWoppo[j, ele_to_index(ei)] += vdw
        # Long cutoff for CLB and RIG
        for (i,j) in NbLong:
            ri = self.Radii[i]; rj = self.Radii[j];
            qi = self.Charge[i]; qj = self.Charge[j];
            posi = self.AtomPos[i]; posj = self.AtomPos[j];
            dis = np.linalg.norm(posi-posj)
            resi = self.ResID[i]; resj = self.ResID[j];
            ei = self.AtomType[i].replace(" ",""); ej = self.AtomType[j].replace(" ","");
            clb = qi*qj/dis
            rig = np.zeros([len(FRI)], float)
            ArgList = []
            for tp in FRI:
                ArgList.append( tp+(ri,rj,dis) )
            rig = map(FRIKernel, ArgList)
            rig = np.asarray(rig, float)
            self.RIG[i,:,ele_to_index(ej)] += rig[:]
            self.RIG[j,:,ele_to_index(ei)] += rig[:]
            self.CLB[i,ele_to_index(ej)] += clb
            self.CLB[j,ele_to_index(ei)] += clb
            if bool(resi == self.MutResID) ^ bool(resj == self.MutResID):
                self.RIGoppo[i,:,ele_to_index(ej)] += rig[:]
                self.RIGoppo[j,:,ele_to_index(ei)] += rig[:]
                self.CLBoppo[i,ele_to_index(ej)] += clb
                self.CLBoppo[j,ele_to_index(ei)] += clb

    def construct_index_list(self, CutNear = 10.0):
        """Lists that contains atom index
            first index: 0 mutsite, 1 other near, 2 all
            seconde index: 0 C, 1 N, 2 O, 3 S, 4 H, 5 heavy, 6 all
        """
        heavy = ['C','N','O','S']
        self.IndexList = [ [ [] for i in range(7) ] for i in range(3) ]
        IndexMutSite = []
        for i in range(0, len(self.ResID)):
            if self.ResID[i] == self.MutResID:
                IndexMutSite.append(i)
        NearRes = []
        IndexNear = []
        for i in range(0, len(self.ResID)):
            dis = 100000.0
            for j in range(0, len(IndexMutSite)):
                tmpdis = np.linalg.norm(self.AtomPos[i] - self.AtomPos[IndexMutSite[j]])
                if tmpdis < dis: dis = tmpdis
            if (dis < CutNear and
               self.ResID[i] not in NearRes and
               self.ResID[i] != self.MutResID):
               NearRes.append( self.ResID[i] )
        for i in range(0, len(self.ResID)):
            if self.ResID[i] in NearRes:
                IndexNear.append(i)
        for i in IndexMutSite:
            self.IndexList[0][ele_to_index(self.AtomType[i])].append(i)
            if self.AtomType[i] in heavy:
                self.IndexList[0][5].append(i)
            self.IndexList[0][6].append(i)
        for i in IndexNear:
            self.IndexList[1][ele_to_index(self.AtomType[i])].append(i)
            if self.AtomType[i] in heavy:
                self.IndexList[1][5].append(i)
            self.IndexList[1][6].append(i)
        for i in range(0, len(self.AtomType)):
            self.IndexList[2][ele_to_index(self.AtomType[i])].append(i)
            if self.AtomType[i] in heavy:
                self.IndexList[2][5].append(i)
            self.IndexList[2][6].append(i)
        for i in range(0,3):
            for j in range(0,7):
                self.IndexList[i][j] = np.asarray(self.IndexList[i][j], int)

    def construct_feature(self, CutNear = 10.0, OutputName = False, SolvEng = True,
                          Area = True, Charge = True, CLB = True, VDW = True,
                          RIG = True, CLBoppo = False, VDWoppo = False,
                          RIGoppo = False, Global = True, Other = True):
        self.Feature = []

        if OutputName:
            self.FeatureName = []
            LocName = ['Site', 'Near', 'All']
            TypeName = ['C', 'N', 'O', 'S', 'H', 'Heavy', 'All']
            MethodName = ['Sum', 'Abs']

        if SolvEng:
            SolvEng = np.asarray(self.SolvEng, float)
            for i in range(0,3):
                for j in range(0,7):
                    self.Feature.append(np.sum(SolvEng[self.IndexList[i][j]]))
            if OutputName:
                for i in range(0,3):
                    for j in range(0,7):
                        self.FeatureName.append('SolvEng_'+LocName[i]+'_Sum_'+TypeName[j])

        if Area:
            Area = np.asarray(self.Area, float)
            for i in range(0,3):
                for j in range(0,7):
                    self.Feature.append(np.sum(Area[self.IndexList[i][j]]))
            if OutputName:
                for i in range(0,3):
                    for j in range(0,7):
                        self.FeatureName.append('Area_'+LocName[i]+'_Sum_'+TypeName[j])

        if Charge:
            Charge = np.asarray(self.Charge, float)
            for k in range(0,2):
                for i in range(0,3):
                    for j in range(0,7):
                        if k == 0:
                            self.Feature.append(np.sum(Charge[self.IndexList[i][j]]))
                        elif k == 1:
                            self.Feature.append(np.sum(np.abs(Charge[self.IndexList[i][j]])))
            if OutputName:
                for k in range(0,2):
                    for i in range(0,3):
                        for j in range(0,7):
                            self.FeatureName.append('Charge_'+LocName[i]+'_'+MethodName[k]+'_'+TypeName[j])

        IndexArray = [np.array([0], int),
                      np.array([1], int),
                      np.array([2], int),
                      np.array([3], int),
                      np.array([4], int),
                      np.array([0,1,2,3], int),
                      np.array([0,1,2,3,4], int)
        ]

        if RIG:
            for f in range(0, len(self.FRI)):
                for i in range(0,3):
                    # for j in [0,1,2,3,5,6]: # including H
                    #     for k in [5,6]:
                    for j in [0,1,2,3,5]: # Only heavy atoms
                        for k in [5]:
                            self.Feature.append(np.sum(self.RIG[self.IndexList[i][j],:,:][:,np.asarray([f], int),:][:,:,IndexArray[k]]))
            if OutputName:
                for f in range(0, len(self.FRI)):
                    for i in range(0,3):
                        # for j in [0,1,2,3,5,6]: # including H
                        #     for k in [5,6]:
                        for j in [0,1,2,3,5]: # Only heavy atoms
                            for k in [5]:
                                self.FeatureName.append('Rigid_'+self.FRI[f][0]+'_'+str(self.FRI[f][1])+'_'+str(self.FRI[f][2])+'_'+LocName[i]+'_Sum_'+TypeName[j]+'_'+TypeName[k])

        if VDW:
            for i in range(0,3):
                for j in [0,1,2,3,5]:
                    for k in [5]: # ele-all
                    # for k in [0,1,2,3,5]: # detailed ele-ele
                        self.Feature.append(np.sum(self.VDW[self.IndexList[i][j],:][:,IndexArray[k]]))
            if OutputName:
                for i in range(0,3):
                    for j in [0,1,2,3,5]:
                        for k in [5]: # ele-all
                        # for k in [0,1,2,3,5]: # detailed ele-ele
                            self.FeatureName.append('Vdw_'+LocName[i]+'_Sum_'+TypeName[j]+'_'+TypeName[k])

        if CLB:
            for m in range(0,2):
                for i in range(0,3):
                    for j in [0,1,2,3,5]:
                        for k in [6]: # ele-all
                        # for k in [0,1,2,3,5,6]: # detailed ele-ele
                            if m == 0:
                                self.Feature.append(np.sum(self.CLB[self.IndexList[i][j]][:,IndexArray[k]]))
                            elif m == 1:
                                self.Feature.append(np.sum(np.abs(self.CLB[self.IndexList[i][j]][:,IndexArray[k]])))
            if OutputName:
                for m in range(0,2):
                    for i in range(0,3):
                        for j in [0,1,2,3,5]:
                            for k in [6]: # ele-all
                            # for k in [0,1,2,3,5,6]: # detailed ele-ele
                                self.FeatureName.append('Clb_'+LocName[i]+'_'+MethodName[m]+'_'+TypeName[j]+'_'+TypeName[k])

        if RIGoppo:
            for f in range(0, len(self.FRI)):
                for i in range(0,3):
                    for j in [0,1,2,3,5,6]:
                        for k in [5,6]:
                            self.Feature.append(np.sum(self.RIGoppo[self.IndexList[i][j],:,:][:,np.asarray([f], int),:][:,:,IndexArray[k]]))
            if OutputName:
                for f in range(0, len(self.FRI)):
                    for i in range(0,3):
                        for j in [0,1,2,3,5,6]:
                            for k in [5,6]:
                                self.FeatureName.append('RigidOppo_'+self.FRI[f][0]+'_'+str(self.FRI[f][1])+'_'+str(self.FRI[f][2])+'_'+LocName[i]+'_Sum_'+TypeName[j]+'_'+TypeName[k])

        if VDWoppo:
            for i in range(0,3):
                for j in [0,1,2,3,5]:
                    for k in [0,1,2,3,5]:
                        self.Feature.append(np.sum(self.VDWoppo[self.IndexList[i][j],:][:,IndexArray[k]]))
            if OutputName:
                for i in range(0,3):
                    for j in [0,1,2,3,5]:
                        for k in [0,1,2,3,5]:
                            self.FeatureName.append('VdwOppo_'+LocName[i]+'_Sum_'+TypeName[j]+'_'+TypeName[k])

        if CLBoppo:
            for m in range(0,2):
                for i in range(0,3):
                    for j in [0,1,2,3,5]:
                        for k in [0,1,2,3,5,6]:
                            if m == 0:
                                self.Feature.append(np.sum(self.CLBoppo[self.IndexList[i][j]][:,IndexArray[k]]))
                            elif m == 1:
                                self.Feature.append(np.sum(np.abs(self.CLBoppo[self.IndexList[i][j]][:,IndexArray[k]])))
            if OutputName:
                for m in range(0,2):
                    for i in range(0,3):
                        for j in [0,1,2,3,5]:
                            for k in [0,1,2,3,5,6]:
                                self.FeatureName.append('ClbOppo_'+LocName[i]+'_'+MethodName[m]+'_'+TypeName[j]+'_'+TypeName[k])

        if Global:
            self.Feature.append(self.TotalSolvEng)
            self.Feature.append(self.TotalArea)
            self.Feature.append(self.TotalVolume)
            if OutputName:
                self.FeatureName.append('Total_SolvEng')
                self.FeatureName.append('Total_Area')
                self.FeatureName.append('Total_Volume')

        if Other:
            def AAcharge(AA):
                if AA in ['D','E']:
                    return -1.
                elif AA in ['R','H','K']:
                    return 1.
                else:
                    return 0.
            Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
            PolarAll = ['S','T','N','Q','R','H','K','D','E']
            PolarUncharged = ['S','T','N','Q']
            PolarPosCharged = ['R','H','K']
            PolarNegCharged = ['D','E']
            SpecialCase = ['C','U','G','P']
            AAvolume = {'A':88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, 'E':138.4, 'Q':143.8, 'G':60.1, 'H':153.2, 'I':166.7, 'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, 'S':89., 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140. }
            AAhydropathy = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D': -3.5, 'C': 2.5, 'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5, 'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
            AAarea = {'A':115.,'R':225.,'D':150.,'N':160.,'C':135.,'E':190.,'Q':180.,'G':75.,'H':195.,'I':175.,'L':170.,'K':200.,'M':185.,'F':210.,'P':145.,'S':115.,'T':140.,'W':255.,'Y':230.,'V':155.}
            AAweight = {'A':89.094,'R':174.203,'N':132.119,'D':133.104,'C':121.154,'E':147.131,'Q':146.146,'G':75.067,'H':155.156,'I':131.175,'L':131.175,'K':146.189,'M':149.208,'F':165.192,'P':115.132,'S':105.093,'T':119.12,'W':204.228,'Y':181.191,'V':117.148}
            Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]
            AA = self.Sequence[self.MutSiteInSeq]
            for Group in Groups:
                if AA in Group:
                    self.Feature.append(1.0)
                else:
                    self.Feature.append(0.0)
            self.Feature.append(AAvolume[AA])
            self.Feature.append(AAhydropathy[AA])
            self.Feature.append(AAarea[AA])
            self.Feature.append(AAweight[AA])
            self.Feature.append(AAcharge(AA))
            if OutputName:
                self.FeatureName.extend(['AAtype-Hydrophobic', 'AAtype-Polar', 'AAtype-PolarUncharged', 'AAtype-PolarPos', 'AAtype-PolarNeg', 'AAtype-Special'])
                self.FeatureName.extend(['AA-Volume', 'AA-Hydropathy', 'AA-Area', 'AA-Weight', 'AA-Charge'])

    def get_chimera_info(self, Clash = True):
        os.system('chimera --nogui --script "chimera.py '+str(self.MutResID)+' '+self.InDataDir+self.PDBName+'.pdb '+self.OutDataDir+self.PDBName+'"')
        self.FeatureChimera = []
        HBondFile = open(self.OutDataDir+self.PDBName+'.hbond')
        lines = HBondFile.read().splitlines()
        HBond = np.zeros([3], float)
        for line in lines:
            if len(line) > 0:
                if line[0] == ':':
                    a = line[0:12]; b = line[12:24]; c = line[24:36];
                    if a[0] == ':':
                        donor, dummy1, dummy2 = a[1:].partition('.')
                        donor_assigned = True
                    else:
                        donor_assigned = False
                    if b[0] == ':':
                        accepter, dummy1, dummy2 = b[1:].partition('.')
                        acceptor_assigned = True
                    else:
                        acceptor_assigned = False
                    if c[0] == ':':
                        H, dummy1, dummy2 = c[1:].partition('.')
                        H_assigned = True
                    else:
                        H_assigned = False
                    if donor_assigned:
                        if int(donor) == self.MutResID: HBond[0] += 1.
                    if acceptor_assigned:
                        if int(accepter) == self.MutResID: HBond[1] += 1.
                    if H_assigned:
                        if int(H) == self.MutResID: HBond[2] += 1.
        self.FeatureChimera.extend(HBond.tolist())
        if Clash: # Not needed for wild type proteins
            Clashes = np.zeros([2], float) # all/heavy
            MaxOverlap = 0.
            MinInvDistance = 0.
            ClashFile = open(self.OutDataDir+self.PDBName+'.clash')
            lines = ClashFile.read().splitlines()
            line = lines[6]
            a,b = line.split()
            if int(a) > 0:
                Clashes[0] = float(a)
                for line in lines[8:8+int(a)]:
                    a,b,c,d = line.split()
                    if a.find('H') == -1 and b.find('H') == -1:
                        Clashes[1] += 1.
                        if float(c) > MaxOverlap:
                            MaxOverlap = float(c)
                        if 1./float(d) > MinInvDistance:
                            MinInvDistance = 1./float(d)
            self.FeatureChimera.extend(Clashes.tolist())
            self.FeatureChimera.append(MaxOverlap)
            self.FeatureChimera.append(MinInvDistance)

    def get_pka_info(self):
        # run PROPKA
        #pdb2pqr_path = '/home/cang/Dropbox/mutation/package/pdb2pqr-linux-bin64-2.1.0/'
        #pdb_path = '/home/cang/Dropbox/mutation/dev/mut_v8/Data_Mem/'+self.PDBName+'/'+self.PDBName+'.pdb'
        #pdb_out_path = '/home/cang/Dropbox/mutation/dev/mut_v8/Data_Mem/'+self.PDBName+'/'
        #pqrfile = self.PhysicsName+'.pqr'
        #os.system(pdb2pqr_path+'pdb2pqr --ff=amber --ph-calc-method=propka --with-ph='+str(self.pH)+' '+pdb_path+' '+pdb_out_path+pqrfile)

        pKaFile = open(self.OutDataDir+self.PhysicsName+'.propka')
        lines = pKaFile.read().splitlines()
        self.pKaSite = 0.0
        self.pKa = []
        self.pKaName = []
        self.pKaCt = 0.0
        self.pKaNt = 0.0
        for line in lines:
            if len(line) < 24: continue
            if line[23] == '%':
                if line[0:2] == 'C-': self.pKaCt = float(line[11:16])
                if line[0:2] == 'N+': self.pKaNt = float(line[11:16])
                resid = int(line[4:7])
                if resid != self.MutResID:
                    self.pKa.append(float(line[11:16]))
                    self.pKaName.append(line[0:3])
                else:
                    self.pKaSite = float(line[11:16])

    def construct_PH_feature(self):
        from dionysus import Filtration, StaticPersistence, data_dim_cmp, vertex_cmp, \
                             fill_alpha3D_complex, fill_alpha2D_complex, points_file
        from    math            import sqrt

        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])
        PDBFile = open(self.InDataDir+self.PDBName+'.pdb')
        lines = PDBFile.read().splitlines()
        pt = []
        for line in lines:
            if line[0:4] == 'ATOM' and line[13:14] != 'H':
                pos = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                pt.append(pos)
        pro_out_file = open(self.OutDataDir+self.PDBName+'.PH', 'w')
        #Filtration
        f = Filtration()
        fill_alpha3D_complex(pt, f)
        f.sort(data_dim_cmp)
        p = StaticPersistence(f)
        p.pair_simplices()
        smap = p.make_simplex_map(f)
        for i in p:
            if i.sign():
                b = smap[i]
                if i.unpaired():
                    pro_out_file.write(str(b.dimension())+' '+str(sqrt(b.data[0]))+' '+'inf\n')
                    continue
                d = smap[i.pair()]
                pro_out_file.write(str(b.dimension())+' '+str(sqrt(b.data[0]))+' '+str(sqrt(d.data[0]))+'\n')
        pro_out_file.close()

        self.FeaturePH = []
        PHFile = open(self.OutDataDir+self.PDBName+'.PH')
        lines = PHFile.read().splitlines()
        bars = np.zeros(len(lines), dtype=dt)
        cnt = 0
        for line in lines:
            dim, b, d = line.split()
            bars[cnt]['dim'] = int(dim)
            bars[cnt]['birth'] = float(b)
            bars[cnt]['death'] = float(d)
            cnt += 1
        b0 = 0.; b1 = 0.; b2 = 0.;
        for bar in bars:
            if bar['death'] < 100000 and bar['dim'] == 0:
                b0 += bar['death'] - bar['birth']
            elif bar['death'] < 100000 and bar['dim'] == 1:
                b1 += bar['death'] - bar['birth']
            elif bar['death'] < 100000 and bar['dim'] == 2:
                b2 += bar['death'] - bar['birth']
        self.FeaturePH.extend([b0,b1,b2])

    def construct_unfolded_feature(self):
        import opengm
        from scipy import special

        # Gaussian chain electrostatics
        Charged = ['D', 'E', 'R', 'K']
        Ionizable = ['D', 'E', 'Y', 'C', 'K', 'R', 'H']
        ChargeDic = {}
        ChargeDic['D'] = -1.0
        ChargeDic['E'] = -1.0
        ChargeDic['R'] = 1.0
        ChargeDic['K'] = 1.0
        IntrinsicPKA = {'D':4.0, 'E':4.4, 'Y':9.6, 'C':8.3, 'K':10.4, 'R':12.0, 'H':6.3}
        X0 = {'D':1.0, 'E':1.0, 'Y':1.0, 'C':1.0, 'K':0.0, 'R':0.0, 'H':0.0}
        s = 5.0
        b = 7.5
        epsilon = 78.5
        seq = self.Sequence
        pH = self.pH
        eng = 0.0
        for j in range(0, len(seq)-1):
            for k in range(j+1, len(seq)):
                Aj = seq[j]; Ak = seq[k];
                if Aj in Charged and Ak in Charged:
                    l = float(k-j)
                    d = b*np.sqrt(l) + s
                    Ionic = 0.15
                    kappa = np.sqrt(Ionic)/3.04
                    x = kappa*d/np.sqrt(6)
                    W = ChargeDic[Aj]*ChargeDic[Ak]*332.0*np.sqrt(6.0/sp.pi)*(1.0 - np.sqrt(np.pi)*x*np.exp(x*x)*special.erfc(x))/(epsilon*d)
                    eng += W
        NodeNum = 0
        for j in range(0, len(seq)):
            if seq[j] in Ionizable:
                NodeNum += 1
        IntrinsicInfo = np.zeros([NodeNum,2], float)
        numberOfStates = np.ones([NodeNum], int)
        numberOfStates[:] = numberOfStates[:]*2
        gm = opengm.graphicalModel(numberOfStates, operator="multiplier")
        cnt = 0
        index = {}
        for j in range(0, len(seq)):
            A = seq[j]
            if A in Ionizable:
                index[j] = cnt
                FactorPot = np.ones([2], dtype=float)
                FactorPot[0] = np.exp(-2.3*0.59*(pH-IntrinsicPKA[A])*1.0)
                FactorPot[1] = 1.0
                fid = gm.addFunction(FactorPot)
                gm.addFactor(fid, cnt)
                cnt += 1
        for j in range(0, len(seq)-1):
            for k in range(j+1, len(seq)):
                Aj = seq[j]; Ak = seq[k];
                if Aj in Ionizable and Ak in Ionizable:
                    r1 = index[j]; r2 = index[k];
                    FactorPot = np.ones([2, 2], dtype=float)
                    l = float(k-j)
                    d = b*np.sqrt(l) + s
                    Ionic = 0.15
                    kappa = np.sqrt(Ionic)/3.04
                    x = kappa*d/np.sqrt(6)
                    W = 332.0*np.sqrt(6.0/sp.pi)*(1.0 - np.sqrt(np.pi)*x*np.exp(x*x)*special.erfc(x))/(epsilon*d)
                    FactorPot[0,0] = np.exp(-W*(1.0-X0[Aj])*(1.0-X0[Ak]))
                    FactorPot[1,0] = np.exp(-W*(0.0-X0[Aj])*(1.0-X0[Ak]))
                    FactorPot[0,1] = np.exp(-W*(1.0-X0[Aj])*(0.0-X0[Ak]))
                    FactorPot[1,1] = np.exp(-W*(0.0-X0[Aj])*(0.0-X0[Ak]))
                    fid = gm.addFunction(FactorPot)
                    gm.addFactor(fid, np.array([r1,r2], dtype=opengm.label_type))
        inference = opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(damping=0.01,steps=1000,convergenceBound=0.000001),
                                                       accumulator="integrator")
        inference.infer()
        Qu = 0.0
        cnt = 0
        QDic = {}
        for j in range(0, len(seq)):
            A = seq[j]
            if A in Ionizable:
                Belief = inference.marginals(cnt)
                Qu += Belief[0][0]/np.sum(Belief) - X0[A]
                QDic[j]=(Belief[0][0]/np.sum(Belief) - X0[A])
                cnt += 1
        # Use inferenced charges to compute interaction energy instead
        InferEng = 0.0
        for j in range(0, len(seq)-1):
            for k in range(j+1, len(seq)):
                Aj = seq[j]; Ak = seq[k];
                if Aj in Ionizable and Ak in Ionizable:
                    r1 = index[j]; r2 = index[k];
                    l = float(k-j)
                    d = b*np.sqrt(l) + s
                    Ionic = 0.15
                    kappa = np.sqrt(Ionic)/3.04
                    x = kappa*d/np.sqrt(6)
                    W = 332.0*np.sqrt(6.0/sp.pi)*(1.0 - np.sqrt(np.pi)*x*np.exp(x*x)*special.erfc(x))/(epsilon*d)
                    qj = QDic[j]; qk = QDic[k];
                    InferEng += W*qj*qk

        # Sequence neighbor statistics
        nbh = [1,2,3,4,5,6,7,8,9,10]
        Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
        PolarAll = ['S','T','N','Q','R','H','K','D','E']
        PolarUncharged = ['S','T','N','Q']
        PolarPosCharged = ['R','H','K']
        PolarNegCharged = ['D','E']
        SpecialCase = ['C','U','G','P']
        AAvolume = {'A':88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, 'E':138.4, 'Q':143.8, 'G':60.1, 'H':153.2, 'I':166.7, 'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, 'S':89., 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140. }
        AAhydropathy = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D': -3.5, 'C': 2.5, 'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5, 'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
        Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]
        loc = self.MutSiteInSeq
        feature_i = []
        for Group in Groups:
            for nb in nbh:
                cnt = 0.
                gpcnt = 0.
                for j in range(0, len(seq)):
                    if abs(j-loc) <= nb and j != loc:
                        cnt += 1.
                        if seq[j] in Group:
                            gpcnt += 1.
                feature_i.extend([gpcnt,gpcnt/cnt])

        for nb in nbh:
            NeighborVol = []
            NeighborHyd = []
            NeighborHydVol = []
            for j in range(0, len(seq)):
                if abs(j-loc) <= nb and j != loc:
                    if not seq[j] in AAvolume.keys() or not seq[j] in AAhydropathy.keys():
                        print seq[j], 'not in dic'
                        exit()
                    NeighborVol.append(np.float64(AAvolume[seq[j]]))
                    NeighborHyd.append(np.float64(AAhydropathy[seq[j]]))
                    if seq[j] in Hydro:
                        NeighborHydVol.append(np.float64(AAvolume[seq[j]]))
            NeighborVol = np.asarray(NeighborVol, np.float64)
            NeighborHyd = np.asarray(NeighborHyd, np.float64)
            NeighborHydVol = np.asarray(NeighborHydVol, np.float64)
            feature_i.extend([np.sum(NeighborVol)/max(float(len(NeighborVol)),1.0),np.var(NeighborVol, dtype=np.float64)])
            feature_i.extend([np.sum(NeighborHyd)/max(float(len(NeighborHyd)),1.0),np.var(NeighborHyd, dtype=np.float64)])
            if len(NeighborHydVol) > 0:
                HydVolVar = np.var(NeighborHydVol, dtype=np.float64)
            else:
                HydVolVar = 0.0
            feature_i.extend([np.sum(NeighborHydVol)/max(float(len(NeighborHydVol)),1.0),HydVolVar])
        for Group in Groups:
            if seq[loc] in Group:
                feature_i.append(1.0)
            else:
                feature_i.append(0.0)
        feature_i.append(AAvolume[seq[loc]])
        feature_i.append(AAhydropathy[seq[loc]])
        self.FeatureUnfolded = [Qu, eng, InferEng] + feature_i

    def construct_environment_feature(self):
        def AAcharge(AA):
            if AA in ['D','E']:
                return -1.
            elif AA in ['R','H','K']:
                return 1.
            else:
                return 0.
        self.FeatureEnvironment = []
        NearSeq = []
        CurResID = -1000
        for i in self.IndexList[1][6]:
            ResID = self.ResID[i]
            if ResID != CurResID:
                CurResID = ResID
                NearSeq.append(three_to_one(self.ResName[i]))
        Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
        PolarAll = ['S','T','N','Q','R','H','K','D','E']
        PolarUncharged = ['S','T','N','Q']
        PolarPosCharged = ['R','H','K']
        PolarNegCharged = ['D','E']
        SpecialCase = ['C','U','G','P']
        AAvolume = {'A':88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, 'E':138.4, 'Q':143.8, 'G':60.1, 'H':153.2, 'I':166.7, 'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, 'S':89., 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140. }
        AAhydropathy = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D': -3.5, 'C': 2.5, 'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5, 'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
        AAarea = {'A':115.,'R':225.,'D':150.,'N':160.,'C':135.,'E':190.,'Q':180.,'G':75.,'H':195.,'I':175.,'L':170.,'K':200.,'M':185.,'F':210.,'P':145.,'S':115.,'T':140.,'W':255.,'Y':230.,'V':155.}
        AAweight = {'A':89.094,'R':174.203,'N':132.119,'D':133.104,'C':121.154,'E':147.131,'Q':146.146,'G':75.067,'H':155.156,'I':131.175,'L':131.175,'K':146.189,'M':149.208,'F':165.192,'P':115.132,'S':105.093,'T':119.12,'W':204.228,'Y':181.191,'V':117.148}
        Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]
        for Group in Groups:
            cnt = 0.
            for AA in NearSeq:
                if AA in Group:
                    cnt += 1.
            self.FeatureEnvironment.append(cnt)
            self.FeatureEnvironment.append(cnt/max(1., float(len(NearSeq))))
        Vol = []; Hyd = []; Area = []; Wgt = []; Chg = [];
        for AA in NearSeq:
            Vol.append(AAvolume[AA])
            Hyd.append(AAhydropathy[AA])
            Area.append(AAarea[AA])
            Wgt.append(AAweight[AA])
            Chg.append(AAcharge(AA))
        Vol = np.asarray(Vol)
        Hyd = np.asarray(Hyd)
        Area = np.asarray(Area)
        Wgt = np.asarray(Wgt)

        if len(NearSeq) == 0:
            self.FeatureEnviroment.extend([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        else:
            self.FeatureEnvironment.extend([np.sum(Vol), np.sum(Vol)/float(len(NearSeq)), np.var(Vol)])
            self.FeatureEnvironment.extend([np.sum(Hyd), np.sum(Hyd)/float(len(NearSeq)), np.var(Hyd)])
            self.FeatureEnvironment.extend([np.sum(Area), np.sum(Area)/float(len(NearSeq)), np.var(Area)])
            self.FeatureEnvironment.extend([np.sum(Wgt), np.sum(Wgt)/float(len(NearSeq)), np.var(Wgt)])
        self.FeatureEnvironment.append(sum(Chg))
        print len(self.FeatureEnvironment), self.FeatureEnvironment

    def run_blast(self, Long = True, Local = True):
        SeqLong = self.Sequence
        loc = self.MutSiteInSeq
        SeqLocal = []
        Neighbor = 7
        l = max(0,loc-7); r = min(len(SeqLong)-1, loc+7);
        self.MutSiteInLocalSeq = loc - l
        for i in range(l,r+1):
            SeqLocal.append(SeqLong[i])
        outfile = open(self.OutDataDir+self.Name+'.fasta', 'w')
        cnt = 0
        for r in SeqLong:
            cnt += 1
            outfile.write(r)
            if cnt%80 == 0: outfile.write('\n')
        outfile.close()
        outfile = open(self.OutDataDir+self.Name+'_local.fasta', 'w')
        cnt = 0
        for r in SeqLocal:
            cnt += 1
            outfile.write(r)
            if cnt%80 == 0: outfile.write('\n')
        outfile.close()
        if (not os.path.exists(self.OutDataDir+self.Name+'.pssm')) and Long:
            os.system('psiblast -query '+self.OutDataDir+self.Name+'.fasta'+ \
            ' -db nr -num_iterations 3 -evalue 5000 -out '+ \
            self.OutDataDir+self.Name+'.out -out_ascii_pssm '+self.OutDataDir+self.Name+'.pssm')
        if (not os.path.exists(self.OutDataDir+self.Name+'_local.pssm')) and Local:
            os.system('psiblast -query '+self.OutDataDir+self.Name+'_local.fasta'+ \
            ' -db nr -num_iterations 3 -evalue 5000 -out '+ \
            self.OutDataDir+self.Name+'_local.out -out_ascii_pssm '+self.OutDataDir+self.Name+'_local.pssm')

    def construct_seq_feature(self, Long = False, Local = True):
        # from Bio.PDB import *
        from Bio.PDB.PDBParser import PDBParser
        from Bio.PDB.DSSP import DSSP
        # from Bio.PDB.Polypeptide import *
        def ss2index( st ):
            if st == 'H':
                return 1
            elif st == 'E':
                return 2
            elif st == 'G':
                return 3
            elif st == 'S':
                return 4
            elif st == 'B':
                return 5
            elif st == 'T':
                return 6
            elif st == 'I':
                return 7
            elif st == '-':
                return 0

        self.FeatureSeq = []
        # Structure based DSSP
        chainid = self.Chain
        parser = PDBParser()
        structure = parser.get_structure('prot', self.OutDataDir+self.PDBName+'.pdb')
        model = structure[0]
        dssp = DSSP(model, self.OutDataDir+self.PDBName+'.pdb', './dssp')
        ssindex = ss2index(dssp[(chainid, (' ', self.MutResID, ' '))][2])
        self.FeatureSeq.append(ssindex)
        # Sequence based Spider3
        if (not os.path.exists(self.OutDataDir+self.SeqName+'.spd3')) and Long:
            os.system('../../package/SPIDER2_local/misc/pred_pssm.py '+self.OutDataDir+self.SeqName+'.pssm')
            os.system('mv ./'+self.Name+'.spd3 '+self.OutDataDir+self.SeqName+'.spd3')
        if (not os.path.exists(self.OutDataDir+self.SeqName+'_local.spd3')) and Local:
            os.system('../../package/SPIDER2_local/misc/pred_pssm.py '+self.OutDataDir+self.SeqName+'_local.pssm')
            os.system('mv ./'+self.Name+'_local.spd3 '+self.OutDataDir+self.SeqName+'_local.spd3')

        psi = 0.0; phi = 0.0; pe = 0.0; pc = 0.0; ph = 0.0;
        if Long:
            spdfile = open(self.OutDataDir+self.SeqName+'.spd3')
            ind = self.MutSiteInSeq
        if Local:
            spdfile = open(self.OutDataDir+self.SeqName+'_local.spd3')
            ind = self.MutSiteInLocalSeq
        lines = spdfile.read().splitlines()
        line = lines[ind+1]
        d0, d1, d2, d3, phi, psi, d4, d5, pc, pe, ph = line.split()
        spdfile.close()
        self.FeatureSeq.extend([float(phi),float(psi),float(pc),float(pe),float(ph)])

def construct_mutation_feature(ProtWild, ProtMut):
    Feature = []
    # Regular features
    Feature.extend(ProtWild.Feature)
    Feature.extend(ProtMut.Feature)
    Feature.extend(map(operator.sub, ProtMut.Feature, ProtWild.Feature))
    print "Regular", len(Feature)

    # Chimera with HBonds and clashes
    #Feature.extend(ProtWild.FeatureChimera)
    #Feature.extend(ProtMut.FeatureChimera)
    #Feature.extend(map(operator.sub, ProtMut.FeatureChimera[0:3], ProtWild.FeatureChimera[0:3]))
    #print "Chimera", len(Feature)

    # pKa features
    pKaIndex = {'ASP':0, 'GLU':1, 'ARG':2, 'LYS':3, 'HIS':4, 'CYS':5, 'TYR':6}
    pKaGroup = ['ASP', 'CLU', 'ARG', 'LYS', 'HIS', 'CYS', 'TYR']
    mutpKa = np.asarray(ProtMut.pKa, float)
    wildpKa = np.asarray(ProtWild.pKa, float)
    wildpKaname = ProtWild.pKaName
    defer = mutpKa - wildpKa
    absmax = np.max(np.abs(defer))
    abssum = np.sum(np.abs(defer))
    maxpos = np.max(defer)
    maxneg = np.min(defer)
    netchange = np.sum(defer)
    DetailShiftAbs = np.zeros([7], float)
    DetailShiftNet = np.zeros([7], float)
    for j in range(0, len(wildpKa)):
        if wildpKaname[j] in pKaGroup:
            DetailShiftAbs[pKaIndex[wildpKaname[j]]] += np.abs(mutpKa[j]-wildpKa[j])
            DetailShiftNet[pKaIndex[wildpKaname[j]]] += mutpKa[j]-wildpKa[j]
    mutC = ProtMut.pKaCt; mutN = ProtMut.pKaNt;
    wildC = ProtWild.pKaCt; wildN = ProtWild.pKaNt;
    mutsitepKa = ProtMut.pKaSite; wildsitepKa = ProtWild.pKaSite;
    Feature.extend([absmax,abssum,maxpos,maxneg,netchange,wildsitepKa,mutsitepKa,mutsitepKa-wildsitepKa,wildC,mutC,mutC-wildC,wildN,mutN,mutN-wildN])
    Feature.extend(DetailShiftNet.tolist())
    Feature.extend(DetailShiftAbs.tolist())
    print "pKa", len(Feature)

    # PH features
    # Feature.extend(ProtWild.FeaturePH)
    # Feature.extend(ProtMut.FeaturePH)
    # Feature.extend(map(operator.sub, ProtMut.FeaturePH, ProtWild.FeaturePH))
    # print "PH", len(Feature)

    # Unfolded features
    #Feature.extend(ProtWild.FeatureUnfolded)
    #Feature.extend(ProtMut.FeatureUnfolded)
    #Feature.extend(map(operator.sub, ProtMut.FeatureUnfolded, ProtWild.FeatureUnfolded))
    #print "Unfolded", len(Feature)

    # Environment features
    Feature.extend(ProtWild.FeatureEnvironment)
    print "Environment", len(Feature)

    # PSSM features
    #AAind = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, \
    #         'I':10, 'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, \
    #         'W':18, 'Y':19, 'V':20}
    #res_org = ProtWild.Sequence[ProtWild.MutSiteInSeq]
    #res_mut = ProtMut.Sequence[ProtMut.MutSiteInSeq]
    #res_num = ProtMut.MutResID
    #pssmfile = open(ProtWild.OutDataDir+ProtWild.Name+'.pssm', 'r')
    #seqlen = len(ProtWild.Sequence)
    #pssm_score1 = np.zeros([seqlen,20])
    #pssm_score2 = np.zeros([seqlen,20])
    #pssm_score3 = np.zeros([seqlen,2])
    #lines = pssmfile.read().splitlines()
    #j = 0
    #for line in lines[3:3+seqlen]:
    #    tmp_vec = line.split()
    #    pssm_score1[j,:] = tmp_vec[2:22]
    #    pssm_score2[j,:] = tmp_vec[22:42]
    #    pssm_score3[j,:] = tmp_vec[42:]
    #    j += 1
    #pssmfile.close()
    #Feature.append(pssm_score1[ProtWild.MutSiteInSeq, AAind[res_mut]-1])
    #Feature.append(pssm_score1[ProtWild.MutSiteInSeq, AAind[res_org]-1])
    #Feature.append(pssm_score1[ProtWild.MutSiteInSeq, AAind[res_mut]-1] - pssm_score1[ProtWild.MutSiteInSeq, AAind[res_org]-1])
    #Feature.append(np.sum(pssm_score1[ProtWild.MutSiteInSeq, :]))
    #Feature.append(pssm_score2[ProtWild.MutSiteInSeq, AAind[res_mut]-1])
    #Feature.append(pssm_score2[ProtWild.MutSiteInSeq, AAind[res_org]-1])
    #Feature.append(pssm_score2[ProtWild.MutSiteInSeq, AAind[res_mut]-1] - pssm_score2[ProtWild.MutSiteInSeq, AAind[res_org]-1])
    #Feature.append(np.sum(pssm_score2[ProtWild.MutSiteInSeq, :]))
    #Feature.extend(pssm_score3[ProtWild.MutSiteInSeq].tolist())
    #print "PSSM", len(Feature)

    # SS features
    #Feature.extend(ProtWild.FeatureSeq)
    #Feature.extend(ProtMut.FeatureSeq)
    #Feature.extend(map(operator.sub, ProtMut.FeatureSeq, ProtWild.FeatureSeq))
    #print "SS", len(Feature)

    return Feature
