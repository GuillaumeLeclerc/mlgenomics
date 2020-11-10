# load libraries

from __future__ import print_function
from scipy.spatial import ConvexHull
from skimage.transform import downscale_local_mean
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
import os as os
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
from scipy.io import loadmat
import numpy as np
import pandas as pd


# define functions

# function from here: https://github.com/weallen/STARmap/blob/master/python/analysis.py
def load_data(data_dir, prefix="Cell"):
        #expr = pd.read_csv(os.path.join(data_dir, "data_table.csv"), index_col=0)
    expr = pd.read_csv(os.path.join(data_dir, "cell_barcode_count.csv"), header=None)
    gene_names = pd.read_csv(os.path.join(data_dir, "cell_barcode_names.csv"),header=None)
    rownames = [i for i in range(expr.shape[0])]
    names = gene_names[2]
    names.name = "Gene"
    return pd.DataFrame(data=expr.values, columns=names, index=rownames)

# function from https://github.com/weallen/STARmap/blob/master/python/viz.py
def GetQHulls(labels):
    labels += 1
    Nlabels = labels.max()
    hulls = []
    coords = []
    num_cells = 0
    #cell_id = []
    #print('blah')
    for i in tqdm(range(Nlabels)):#enumerate(regionprops(labels)):
        #print(i,"/",Nlabels)
        curr_coords = np.argwhere(labels==i) # get all coordinates for a single cell label
        # size threshold of > 100 pixels and < 100000
        if curr_coords.shape[0] < 100000 and curr_coords.shape[0] > 1000: # if the cell shape is within threshold region, save the coordinates
            num_cells += 1
            hulls.append(ConvexHull(curr_coords))
            coords.append(curr_coords)
        #cell_id = np.append(cell_id, i)
    #print("Used %d / %d" % (num_cells, Nlabels))
    return hulls, coords

## my functions:
def normalize(counts):

    cts = np.array(counts)
    #cts = cts[np.array(np.where(np.sum(cts, axis = 1)!=0)).flatten(),:] # remove cells, where total library count is zero
    cell_median = np.median(cts, axis = 1) # get row-wise median
    cell_sum = np.sum(cts, axis = 1) # get row-wise sum
    counts_out = cts/(np.tile(cell_sum, (cts.shape[1],1)).transpose()) # divide each column by row-wise sum

    return counts_out

def process_2D(data_dir, gene_names,i):

    # load data
    image = np.load(os.path.join(data_dir,'labels.npz'))["labels"]
    counts = load_data(data_dir, prefix="")

    # process counts
    normalized_counts = pd.DataFrame(normalize(counts)) # normalize counts
    normalized_counts.columns = counts.columns
    normalized_counts = normalized_counts.reindex(columns = sorted(gene_names))
    normalized_counts = np.array(normalized_counts.fillna(0))

    # get coords
    qhulls,coords = GetQHulls(image)# get all coordinates corresponding to single cell
    all_centroids  = np.vstack([np.append(c.mean(0),(0,i)) for c in coords]) # centroids are the average coordinates
    counts_and_coords = np.concatenate((normalized_counts, all_centroids.astype('int')[range(normalized_counts.shape[0]),:]), axis = 1) # concat counts and coords

    return counts_and_coords

#def process_3D()

#  get union of all gene names (across 3D and 2D datasets)
genenames_3D = pd.read_csv('/Users/work/Documents/GitHub/mlgenomics/data_as_downloaded/sequentially_encoded_Wang_et_al_2018/gene_names.csv', header = 0)

dirs = os.listdir('/Users/work/Documents/GitHub/mlgenomics/data_as_downloaded/combinatorially_encoded/all_datasets')

ct = []

for i in range(len(dirs)):

    data_dir1 = os.path.join('/Users/work/Documents/GitHub/mlgenomics/data_as_downloaded/combinatorially_encoded/all_datasets',dirs[i])

    ct.append(load_data(data_dir1, prefix=""))

genenames = []
for i in range(len(dirs)):
    genenames.append(ct[i].columns)

all_genes = np.unique(np.concatenate((np.unique(np.concatenate(genenames)),(np.array(genenames_3D).flatten()))))

# process and append all the 2D datasets to one another

data_out = []

for i in range(len(dirs)):
    data_dir = os.path.join('/Users/work/Documents/GitHub/mlgenomics/data_as_downloaded/combinatorially_encoded/all_datasets',dirs[i])
    data_out.append(process_2D(data_dir, all_genes,i))

# save data out
np.save('Wang_2018_all_2D_processed_new.npy',data_out)
