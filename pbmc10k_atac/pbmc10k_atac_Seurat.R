library(Seurat)
library(SeuratDisk)

setwd('D:/Research/UMINTFS/')

# Load Data
pbmc.rna <- as.sparse(t(read.csv(file = "pbmc10k_rna_hvg_matched_cells.csv", sep = ",",
                               header = TRUE, row.names = 1)))

pbmc.rna <- CollapseSpeciesExpressionMatrix(pbmc.rna)

pbmc.ATAC <- as.sparse(t(read.csv(file = "pbmc10k_atac_hvg_matched_cells.csv", sep = ",",
                               header = TRUE, row.names = 1)))

# Note that since measurements were made in the same cells, the two matrices have identical
# column names
all.equal(colnames(pbmc.rna), colnames(pbmc.ATAC))

# Create a seurat object
pbmc <- CreateSeuratObject(counts = pbmc.rna)
ATAC_assay <- CreateAssayObject(counts = pbmc.ATAC)
pbmc[["ATAC"]] <- ATAC_assay


DefaultAssay(pbmc) <- "RNA"
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc)
pbmc <- ScaleData(pbmc)
pbmc <- RunPCA(pbmc, verbose = FALSE)
pbmc <- FindNeighbors(pbmc, dims = 1:30)
pbmc <- FindClusters(pbmc, resolution = 0.8, verbose = FALSE)

# selecting top features.
rna_markers <- FindAllMarkers(pbmc, assay = "RNA")

# saving markers
write.csv(rna_markers, 'pbmc10k_atac_Seurat_markers.csv')

# saving seurat cluters
write.csv(pbmc@meta.data$seurat_clusters, 'pbmc10k_atac_Seurat_metadata.csv')

# saving the seurat object as h5ad and h5seurat format.
SaveH5Seurat(pbmc, filename = "pbmc10k_atac_seurat.h5Seurat")
Convert("pbmc10k_atac_seurat.h5Seurat", dest = "h5ad")
