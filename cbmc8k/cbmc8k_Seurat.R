library(Seurat)
library(SeuratDisk)

setwd('D:/Research/OngoingProjects/UMINTFS/Codesv5/cbmc8k/')

# Load Data
cbmc.rna <- as.sparse(read.csv(file = "cbmc8k_rna_scaled.csv", sep = ",",
                               header = TRUE, row.names = 1))

cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)

cbmc.adt <- as.sparse(read.csv(file = "cbmc8k_adt_scaled.csv", sep = ",",
                                header = TRUE, row.names = 1))

# Note that since measurements were made in the same cells, the two matrices have identical
# column names
all.equal(colnames(cbmc.rna), colnames(cbmc.adt))

# Create a seurat object
cbmc <- CreateSeuratObject(counts = cbmc.rna)
adt_assay <- CreateAssayObject(counts = cbmc.adt)
cbmc[["ADT"]] <- adt_assay


DefaultAssay(cbmc) <- "RNA"
cbmc <- NormalizeData(cbmc)
cbmc <- FindVariableFeatures(cbmc)
cbmc <- ScaleData(cbmc)
cbmc <- RunPCA(cbmc, verbose = FALSE)
cbmc <- FindNeighbors(cbmc, dims = 1:30)
cbmc <- FindClusters(cbmc, resolution = 0.8, verbose = FALSE)

# selecting top features.
rna_markers <- FindAllMarkers(cbmc, assay = "RNA")

# saving markers
write.csv(rna_markers, 'cbmc8k_Seurat_markers.csv')

# saving seurat cluters
write.csv(cbmc@meta.data$seurat_clusters, 'cbmc8k_Seurat_metadata.csv')

# saving the seurat object as h5ad and h5seurat format.
SaveH5Seurat(cbmc, filename = "cbmc8k_seurat.h5Seurat")
Convert("cbmc8k_seurat.h5Seurat", dest = "h5ad")
