# COMMENT:
# normalizing data as in paper:
#Single-cell data preprocessing: All single-cell analyses were implemented using a custom software
#package in Python for the analysis of STARmap experiments. The per-cell expression matrix was first
#normalized for the expression value Eij across all genes j for each cell i with the formula:
#$Nij = ln(1+median(Ei:)*(Eij/Σ Ei:))$ (see methods: https://science.sciencemag.org/content/sci/suppl/2018/06/20/science.aat5691.DC1/aat5691-Wang-SM.pdf)


from scipy.io import loadmat
import numpy as np
import pandas as pd

x = loadmat('/Users/work/Documents/GitHub/ML_genomics_spatial_project_2020/sequentially_encoded/20180123_BS10_light.mat')
geneinfo = pd.read_csv('/Users/work/Documents/GitHub/ML_genomics_spatial_project_2020/sequentially_encoded/gene_names.csv', header = 0)

# do normalization
e = np.matrix(x['expr'])
df = pd.DataFrame(expr)
df.columns = geneinfo['gene']
df2 = df.loc[df.sum(1) != 0] # remove cells where total library count is zero
Ei = df2.median(1) # get row-wise median
sumE1 = df2.sum(1) # get row-wise sum
temp = df2.div(sumE1, axis=0) # divide each column by the row-wise sum
norm_exp = np.log(1 + temp.multiply(Ei, axis = 'index')) # mulitply each column by the median

# get coordinates
c = np.matrix(x['goodLocs'])
cdf = pd.DataFrame(c)
cdf.columns = ['x','y','z']
cdf2 = cdf.loc[df.sum(1) != 0]

# save output matrix
outmtx = np.matrix(outdf)
np.savez('/Users/work/Documents/GitHub/ML_genomics_spatial_project_2020/outmtx.npz', outmtx)
