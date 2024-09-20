import pandas as pd
import anndata as ad
import numpy as np
import torch
import sys
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


from utils import correlation_matrix, correlation_matrix_distance, calculate_mae_rmse, calculate_cluster_labels, calculate_cluster_centroids, cluster_with_leiden


SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750

original_data = ad.read_h5ad("/workspace/AutoImpute/AutoImpute Model/data/multiome_preprocessed.h5ad")
original_data.var_names_make_unique()
ground_truth = original_data.X.toarray()
mask = torch.zeros(ground_truth.shape, dtype=torch.bool)
mask[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:] = True
nonzero_mask32 = (ground_truth[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:] != 0)

print('loading imputed data')
cell_metadata = pd.read_csv('/workspace/AutoImpute/AutoImpute Model/data/multiome_processed_cell_metadata.csv')
gene_metadata = pd.read_csv('/workspace/AutoImpute/AutoImpute Model/data/multiome_processed_gene_metadata.csv')
expression_data = pd.read_csv('/workspace/AutoImpute/AutoImpute Model/data/imputed_matrix.csv')
adata = ad.AnnData(X=expression_data.values, 
                   obs=cell_metadata, 
                   var=gene_metadata)
print(adata)
X_imputed = adata.X


print('calculating metrics')
### pearson
pearson_corr = pearsonr(X_imputed[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:][nonzero_mask32], ground_truth[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:][nonzero_mask32])[0]
### mae & rmse
mae, rmse = calculate_mae_rmse(X_imputed[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:], ground_truth[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:], nonzero_mask32)
### ari & nmi & purity & jaccard
ari, nmi, purity, jaccard = cluster_with_leiden(adata, resolution_values=[0.5, 0.6, 0.7])
# visualization
sc.tl.umap(adata)
plt.figure(figsize=(8, 6))
sc.pl.umap(adata, color='leiden', show=False)
plt.savefig(f"/workspace/AutoImpute/AutoImpute Model/visualization/umap_multiome.png", format="png")
plt.close()
print(f"UMAP plot saved as 'umap_multiome.png'.")

print(f"pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")