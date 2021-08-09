import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd
import codecs
import csv
import argparse
import h5py

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_clean(data):
    assert isinstance(data, np.ndarray) 
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0] 
    return data

def dict_from_group(group):                    
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group): 
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False): 
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))    #column -> var, row -> obs
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]   
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])    #稀疏矩阵压缩存储
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))    
    return mat, obs, var, uns


def prepro(filename):
    data_path = "dataset\\" + filename + "\\data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False) 
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray()) 
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):               
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:                                   #step 1
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:  #??
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata) #step 2: Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization.
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)  #不是非常理解
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)  #step 3: Logarithmize the data matrix.
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)  #step 4: Annotate highly variable genes
    if normalize_input:
        sc.pp.scale(adata)  #step 5 : Scale data to unit variance and zero mean.
    return adata



parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataname", default = "Quake_10x_Trachea", type = str)
parser.add_argument("--highly_genes", default =400)    

args = parser.parse_args()


X, Y = prepro(args.dataname)    #in preprocess， y = label
X = np.ceil(X).astype(np.int) 
count_X = X

adata = sc.AnnData(X)
adata.obs['Group'] = Y
adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=False, normalize_input=True, logtrans_input=True)   ##size_factor 要变
X = adata.X.astype(np.float32)
Y = np.array(adata.obs["Group"])
high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
count_X = count_X[:, high_variable].astype(np.float32)
size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
cluster_number = int(max(Y) - min(Y) + 1)

# np.set_printoptions(threshold=100000000)
# print("x shape is ", X.shape)
# print(X[0:10, 0:20])
# print("count x shape is ", count_X.shape)
# print("count x  is ", count_X)
# print("size_factor  is ", size_factor)
# print("size_factor shape is ", size_factor.shape)
# # np.set_printoptions(threshold=100000000)
# #print("cell_type is ", Y)
# print("cell_type shape is ", Y.shape)
# print("cluster_number shape is ", cluster_number)
# print("highly variable gene index is ", adata.var.highly_variable.index)

# X_test = X[0:100,:]
# count_X_test = count_X[0:100,:]


np.savetxt("X_400high.csv", X, delimiter=",")
np.savetxt("Count_X_400high.csv", count_X, delimiter=",")
np.savetxt("Size_factor.csv", size_factor, delimiter=",")

np.savetxt("annotation.csv", Y, delimiter=",")





