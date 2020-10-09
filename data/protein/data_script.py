import numpy as np
from Bio import Phylo
import pickle as pkl
from multiprocessing import Pool
        
class TreeDist(object):
     def __init__(self):
         filename = 'PF00144_tree.txt'
         self.tree = next(Phylo.parse(filename, 'newick'))
         self.leaves = list(self.tree.get_terminals())

     def id_to_id_dist(self, id1, id2):
         idx1, idx2 = -1, -1
         for i, leaf in enumerate(self.leaves):
             if id1 in leaf.name:
                 idx1 = i
             if id2 in leaf.name:
                 idx2 = i

         assert idx1 != -1, 'could not find id1: ' + id1
         assert idx2 != -1, 'could not find id2: ' + id2

         return self.tree.distance(self.leaves[idx1],
                                   self.leaves[idx2])

TD = TreeDist()

with open('bio_ids.pkl', 'rb') as file:
     ids = pkl.load(file)

with open('bio_labels.pkl', 'rb') as file: 
    labels=pkl.load(file)


I = [i for i,x in enumerate(labels) if x == 'Bacteroidetes']
N = len(I) # 6078

identity = []
for i in I:
    identity.append(ids[i])

class NewTreeDist(object):
    def __init__(self, ids):
        filename = 'PF00144_tree.txt'
        self.tree = next(Phylo.parse(filename, 'newick'))
        self.leaves = list(self.tree.get_terminals())
        self.look_up = [0]*len(ids)
        for i, leaf in enumerate(self.leaves):
            try:
                idx = identity.index(leaf.name.split('/')[0])
                self.look_up[idx] = i
            except ValueError: # not found in list
                pass
    def id_to_id_dist(self, idx0, idx1):
        return self.tree.distance(self.leaves[self.look_up[idx0]],
                                  self.leaves[self.look_up[idx1]])

NTD = NewTreeDist(ids)

def dist_fn(input):
     id0 = input[0]
     id1 = input[1]
     return NTD.id_to_id_dist(id0, id1)

dists = np.zeros([N,N])
if __name__ == '__main__':
     pool = Pool(processes=32)
     for i in range(N):
         subset2 = np.arange(i+1,N)
         subset1 = [i]*len(subset2)
         result = pool.map(dist_fn, zip(subset1, subset2))
         dists[i, i+1:] = result
         print(i) # liste med alle resultater
     pool.close()

dists = np.array(dists)
dists = dists + np.transpose(dists)
np.savetxt("dists.csv", dists, delimiter = ",")