import anndata as ad
import pandas as pd
import scanpy as sc
import sys


SITE1_CELL = 17243
SITE2_CELL = 15226
SITE3_CELL = 14556
SITE4_CELL = 22224


adata = ad.read_h5ad('/workspace/AutoImpute/AutoImpute Model/data/multiome_processed.h5ad')
adata.var_names_make_unique()

#####################################################################################################################################
adata_GEX = adata[:, adata.var['feature_types'] == 'GEX'].copy()
adata_ATAC = adata[:, adata.var['feature_types'] == 'ATAC'].copy()
### step 1: normalize
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
### step 2: log transform
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ATAC)
### step 3: select highly variable features
sc.pp.highly_variable_genes(adata_GEX, subset=True)
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=4000,
    subset=True
)
adata = ad.concat([adata_ATAC, adata_GEX], axis=1, merge="first")   # left num_atac: ATAC, right 2832: GEX
adata.write_h5ad("/workspace/AutoImpute/AutoImpute Model/data/multiome_preprocessed.h5ad")
print(f"Finish preprocessing\n")
#####################################################################################################################################

adata.obs.to_csv('/workspace/AutoImpute/AutoImpute Model/data/multiome_processed_cell_metadata.csv')
adata.var.to_csv('/workspace/AutoImpute/AutoImpute Model/data/multiome_processed_gene_metadata.csv')
X = adata.X.toarray()
X[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, -2832:] = 0
data = pd.DataFrame(X, index=adata.obs.index, columns=adata.var.index)
print(data.head())

chunk_size = 10000

print(f'Start writing.\n')
with open('/workspace/AutoImpute/AutoImpute Model/data/multiome_missing.csv', 'w') as f:
    data.iloc[:0].to_csv(f, index=True)
    total_rows = len(data)
    for i in range(0, total_rows, chunk_size):
        data.iloc[i:i + chunk_size].to_csv(f, header=False, index=True)
        print(f'Written {i + chunk_size if i + chunk_size < total_rows else total_rows}/{total_rows} rows')

print("Finished writing the CSV file.")