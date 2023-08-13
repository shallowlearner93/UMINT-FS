import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def LoadData(dataname, only_labels=False):
    dir = '/home/chayan/UMINTFS/data/'
        
    # ---------------datasets-------------------
    if dataname == 'cbmc8k':
        if only_labels == False:
            rna = pd.read_csv(dir+'PreprocessedData/cbmc8k_rna_scaled.csv',header = 0 , index_col= 0).T
            adt = pd.read_csv(dir+'PreprocessedData/cbmc8k_adt_scaled.csv',header = 0 , index_col= 0).T
            labels = pd.read_csv(dir+'Labels/cbmc8k_groundTruth.csv',header = 0 , index_col= 0)
            x1 = pd.DataFrame(MinMaxScaler().fit_transform(rna), index=rna.index, columns=rna.columns)
            x2 = pd.DataFrame(MinMaxScaler().fit_transform(adt), index=adt.index, columns=adt.columns)
            return x1, x2, labels
        else:
            labels = pd.read_csv(dir+'Labels/cbmc8k_groundTruth.csv',header = 0 , index_col= 0)
            return labels
    elif dataname == 'pbmc10k_atac':
        if only_labels == False:
            rna = pd.read_csv(dir+'PreprocessedData/pbmc10k_rna_hvg_matched_cells.csv',header = 0 , index_col= 0)
            adt = pd.read_csv(dir+'PreprocessedData/pbmc10k_atac_hvg_matched_cells.csv',header = 0 , index_col= 0)
            labels = pd.read_csv(dir+'Labels/pbmc10k_groundTruth_rna.csv',header = 0 , index_col= 0)
            x1 = pd.DataFrame(MinMaxScaler().fit_transform(rna), index=rna.index, columns=rna.columns)
            x2 = pd.DataFrame(MinMaxScaler().fit_transform(adt), index=adt.index, columns=adt.columns)
            return x1, x2,labels
        else:
            labels = pd.read_csv(dir+'Labels/pbmc10k_groundTruth_rna.csv',header = 0 , index_col= 0)
            return labels
    
    elif dataname == 'bmcite30k':
        if only_labels == False:
            rna = pd.read_csv(dir+'PreprocessedData/bmcite30k_rna_scaled.csv',header = 0 , index_col= 0).T
            adt = pd.read_csv(dir+'PreprocessedData/bmcite30k_adt_scaled.csv',header = 0 , index_col= 0).T
            labels = pd.read_csv(dir+'Labels/bmcite30k_groundTruth.csv',header = 0 , index_col= 0)
            x1 = pd.DataFrame(MinMaxScaler().fit_transform(rna), index=rna.index, columns=rna.columns)
            x2 = pd.DataFrame(MinMaxScaler().fit_transform(adt), index=adt.index, columns=adt.columns)
            return x1, x2, labels
        else:
            labels = pd.read_csv(dir+'Labels/bmcite30k_groundTruth.csv',header = 0 , index_col= 0)
            return labels
    
    elif dataname == 'kotliarov50k':
        if only_labels == False:
            rna = pd.read_csv(dir+'PreprocessedData/kotliarov50k_rna_scaled.csv',header = 0 , index_col= 0).T
            adt = pd.read_csv(dir+'PreprocessedData/kotliarov50k_adt_scaled.csv',header = 0 , index_col= 0).T
            labels = pd.read_csv(dir+'Labels/kotliarov50k_groundTruth.csv',header = 0 , index_col= 0)
            x1 = pd.DataFrame(MinMaxScaler().fit_transform(rna), index=rna.index, columns=rna.columns)
            x2 = pd.DataFrame(MinMaxScaler().fit_transform(adt), index=adt.index, columns=adt.columns)
            return x1, x2, labels
        else:
            labels = pd.read_csv(dir+'Labels/kotliarov50k_groundTruth.csv',header = 0 , index_col= 0)
            return labels
    
    else:
        print('Invalid dataname ',dataname,'\n')
        print('Available data: cbmc8k, pbmc10k_atac, bmcite30k, kotliarov50k')
        if only_labels == False:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            return pd.DataFrame()        
