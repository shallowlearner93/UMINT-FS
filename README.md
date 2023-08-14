# UMINT-FS
Maitra, C., Seal, D.B., Das, V., Vorobeychik, Y. and De, R.K., 2023. **UMINT-FS: UMINT-guided Feature Selection for multi-omics datasets**.
Extention of our earlier work **UMINT** [[1]](#1)

## Architectures for UMINT-guided feature selection techniques. 
![UMINTandUMINTFS](https://github.com/shallowlearner93/UMINT-FS/assets/113589317/c71915b0-7dff-4234-99d4-d3013b46f37e)
**Figure 1**: The network architectures for UMINT-FS working in an unsupervised and supervised setting.

## Requirements
To run **UMINT**, one needs to install `numpy`, `pandas`, `sklearn`, `scipy`, `tqdm` and `tensorflow` packages. Installation codes are as follows:
+ `pip install numpy`
+ `pip install pandas`
+ `pip install scikit-learn`
+ `pip install scipy`
+ `pip install tqdm`
+ `pip install tensorflow`

## Running UMINT and UMINT-FS.
### UMINT
To run **UMINT**, import `umint.py` from the `utills` directory and run the function `UMINT.train_umint`. All the parameters are mentioned below for better understanding. One can also follow a `*_UMINT.ipynb` file from any of the folders: `cbmc8k` and `pbmc10k_atac`

#### Parameters 
All input parameters are as follows: `data`, `hid_neuron`, `mid`, `lambda_act`, `lambda_weight`, `bs`, `epoch`, `verbose`, `val` and `sd`
+ `data`: List of input data matrices for training. [The input data matrices should be in the form of cells x features.]
+ `hid_neuron`: List of neurons in the modality encoding layer [modality wise].
+ `mid`: Dimension onto which the data is being projected. `default: 64`
+ `lambda_act`: Activity regularizer parameter. `default: 0`
+ `lambda_weight`: Kernel regularizer parameter. `default: 0`
+ `bs`: Training batch size. `default: 16`
+ `epoch`: Total number of iteration for training. `default: 25`
+ `verbose`: 1 for printing the output. `default: 1`
+ `val`: List of data matrices for validation. [The validation data matrices also should be in the same format as input data matrices i.e., cells x features.] `default: None`
+ `sd`: To reproduce the results set a seed value. `default: 0`

#### Code to run UMINT
To run **UMINT**, one needs to import the script umint (within the `utills` directory) first. An example is provided below. Let `x1_train` [cells x features] and `x2_train` [cells x features] be two training datasets, coming from two different omics modalities, and `x1_test` [cells x features], `x2_test` [cells x features] be their respective counterparts for validation.
```
data = [x1_train, x2_train]
val = [x1_test, x2_test]
hid_neuron = [128, 128]
mid = 64
lambda_act = 0.0001
lambda_weight = 0.001
bs=16
epochs=25
verbose=0

model, encoder, decoder = umint.UMINT.train_umint(data, hid_neuron, mid=mid, lambda_act=lambda_act,
                                                  bs=bs, epochs=epochs, lambda_weight=lambda_weight,
                                                  val=val, verbose=verbose)
```
#### Code to find the lower dimensional embedding.
Once **UMINT** is trained, to find the latent lower dimensional embedding produced by **UMINT**, run the code below.
```
data = [x1, x2]
low = encoder.predict(data) 
```
To integrate multiple modalities please change the input accordingly. The sizes of data, val and layer_neuron must match in order to run `umint.py` successfully. 

### UMINT-FS (Unsupervised)
To run **UMINT-FS** in an unsupervised manner, import `umint.py` from the `utills` directory and run the function `UMINT.UMINTFS`. All the parameters are mentioned below for better understanding. One can also follow a `*_UMINTFS_Unsupervised.ipynb` file from any of the folders: `cbmc8k` and `pbmc10k_atac`

#### Parameters 
All input parameters are as follows: `data`, `encoder`, `decoder`, `num_fs` and `FStype`.
+ `data`: List of input data matrices. [The input data matrices should be in the form of cells x features.]
+ `encoder`: The trained encoder of `UMINT`.
+ `decoder`: The trained decoder of `UMINT`.
+ `num_fs`: List of numbers (integers). Number of features you want to select from different omics modalities. 
+ `FStype`: Mention the feature selection type, i.e., `top` for selecting top feature and `bottom` for selecting useless features. `default: top`

#### Code to run UMINTFS in an Unsupervised manner.
```
data = [x1, x2]
num_fs = [64, 10]
# the i-th values of num_fs must be less then the number of features in the i-th omics modality.
# As this number indicates how many top feature the user need from this specific modality.

top_features = umint.UMINT.UMINTFS(data, encoder, decoder, num_fs)
```


### UMINT-FS (Supervised)
To run **UMINT-FS** in a supervised manner, import `umint.py` from the `utills` directory and run the function `UMINT.Find_All_Markers`. All the parameters are mentioned below for better understanding. One can also follow a `*_UMINTFS_Supervised.ipynb` file from any of the folders: `cbmc8k` and `pbmc10k_atac`

#### Parameters 
All input parameters are as follows: `data`, `y`, `encoder` and `num_fs`.
+ `data`: List of input data matrices. [The input data matrices should be in the form of cells x features.]
+ `y`: Cell-type labels for all the samples. 
+ `encoder`: The trained encoder of `UMINT`.
+ `num_fs`: List of numbers (integers). Number of features you want to select from different omics modalities. 

#### Code to run UMINTFS in a Supervised manner.
```
data = [x1, x2]
num_fs = [64, 10]
# the i-th values of num_fs must be less then the number of features in the i-th omics modality.
# As this number indicates how many top feature the user need from this specific modality.

Markers = umint.UMINT.Find_All_Markers(data, y, encoder, num_fs)
```
### Find_Markers
One can also find a particular cell-type specific markers using the function `Find_Markers` from the script `umint.py`.

#### Code to find a particular cell-type specific markers.
```
data = [x1, x2]
num_fs = [10,5]
cell_type = 'B'

Markers = umint.UMINT.Find_Markers(data, y, encoder, cell_type, num_fs)
```
The `cell_type` must be present in the array `y`. 


# Dataset Source
--------------
The datasets used in this work can be downloaded from the following link.
https://doi.org/10.5281/zenodo.7723340

# References
--------------
<a id="1">[1]</a>
Maitra, C., Seal, D.B., Das, V. and De, R.K., 2023. Unsupervised neural network for single cell Multi-omics INTegration (UMINT): an application to health and disease. Frontiers in Molecular Biosciences, 10, p.1184748.
