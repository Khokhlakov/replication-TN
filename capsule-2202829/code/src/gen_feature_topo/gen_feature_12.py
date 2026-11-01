import os,sys
import numpy as np

np.set_printoptions(threshold='nan')

# Constant Settings
elelist = ['C','N','O']
eleid ={'C':0,'N':1,'O':2}
Cut = 12.0
small = 0.01

# Generate features for the Alpha complex persistent homology (betti1 & betti2)
def feature_cpx_tda(prefix):
	dt = np.dtype([('num',str),('dim', int), ('birth', float), ('death', float)])
	Feature_b1 = np.zeros([3,3,7,3])
	Feature_b2 = np.zeros([3,3,7,3])
	dir = prefix + '_PH/'
	for e1 in elelist:
		for e2 in elelist:
			wildname = dir+ 'wild_'+prefix+'_'+ e1 +'_'+e2+'.tda'
			mutname = dir+ 'mut_'+prefix+'_'+ e1 +'_'+e2+'.tda'
			for j in range(0,2):
				if j == 0: PHFile = open('./'+wildname)
				if j == 1: PHFile = open('./'+mutname)
				lines = PHFile.read().splitlines()
				tmpbars = np.zeros(len(lines),dtype = dt)
				count = 0

				# Exclude the bars with persistence less than small
				for line in lines:
					a, dim, b, d = line.split()
					if float(d) - float(b) < small: continue
					tmpbars[count]['dim'] = int(dim)
					tmpbars[count]['birth'] = float(b)
					tmpbars[count]['death'] = float(d)
					count += 1
				bars = tmpbars[0:count]

				Ip = eleid[e1]
				Is = eleid[e2]

				# Calculate betti1 features
				if len(bars[bars['dim']== 1]['death']) > 0:
					Feature_b1[Ip,Is,0,j] = np.sum(bars[bars['dim']== 1]['death'] - bars[bars['dim']== 1]['birth'])
					Feature_b1[Ip,Is,1,j] = np.max(bars[bars['dim']== 1]['death'] - bars[bars['dim']== 1]['birth'])
					Feature_b1[Ip,Is,2,j] = np.mean(bars[bars['dim']== 1]['death'] - bars[bars['dim']== 1]['birth'])
					Feature_b1[Ip,Is,3,j] = np.min(bars[bars['dim']== 1]['birth'])
					Feature_b1[Ip,Is,4,j] = np.max(bars[bars['dim']== 1]['birth'])
					Feature_b1[Ip,Is,5,j] = np.min(bars[bars['dim']== 1]['death'])
					Feature_b1[Ip,Is,6,j] = np.max(bars[bars['dim']== 1]['death'])
				Feature_b1[:,:,:,2] = Feature_b1[:,:,:,1]-Feature_b1[:,:,:,0]

				# Calculate betti2 features
				if len(bars[bars['dim']== 2]['death']) > 0:
					Feature_b2[Ip,Is,0,j] = np.sum(bars[bars['dim']== 2]['death'] - bars[bars['dim']== 2]['birth'])
					Feature_b2[Ip,Is,1,j] = np.max(bars[bars['dim']== 2]['death'] - bars[bars['dim']== 2]['birth'])
					Feature_b2[Ip,Is,2,j] = np.mean(bars[bars['dim']== 2]['death'] - bars[bars['dim']== 2]['birth'])
					Feature_b2[Ip,Is,3,j] = np.min(bars[bars['dim']== 2]['birth'])
					Feature_b2[Ip,Is,4,j] = np.max(bars[bars['dim']== 2]['birth'])
					Feature_b2[Ip,Is,5,j] = np.min(bars[bars['dim']== 2]['death'])
					Feature_b2[Ip,Is,6,j] = np.max(bars[bars['dim']== 2]['death'])
				Feature_b2[:,:,:,2] = Feature_b2[:,:,:,1]-Feature_b2[:,:,:,0]

	return Feature_b1,Feature_b2

def main():

	f1,f2 = feature_cpx_tda('binding')
	f3,f4 = feature_cpx_tda('mutation')
	#print f1
	#print f2
	#print f1.shape
	#print f2.shape

	X = np.concatenate([f1.flatten(),f2.flatten(),f3.flatten(),f4.flatten()])
	np.save("X_h12.npy",X)
	#print X
	#print X.shape

	return

main()
