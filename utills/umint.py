import warnings
warnings.simplefilter(action='ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#importing libraries
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import tensorflow as tf
import numpy.linalg as la
from tensorflow import keras
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
from keras import regularizers, initializers, Model
from tensorflow.keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split



class UMINT:


    ##################################  UMINT  ##########################################
    
    def Encoder(inputs, Dim, mid=64, lambda_act=0, lambda_weight=0, sd=0):
        layer1 = [Dense(Dim[i], activation="relu",
                        activity_regularizer=regularizers.l1(lambda_act),
                        kernel_regularizer=regularizers.l2(lambda_weight), 
                        kernel_initializer=initializers.he_uniform(sd), 
                        bias_initializer = initializers.Constant(0.1),
                        name = 'Modality'+str(i+1)+'_Encoding_layer'
                        )(inputs[i]) for i in range(len(inputs))]
        
        Outputs = Dense(mid, activation="relu", 
                        activity_regularizer=regularizers.l1(lambda_act),
                        kernel_regularizer=regularizers.l2(lambda_weight),
                        kernel_initializer=initializers.he_uniform(sd), 
                        bias_initializer = initializers.Constant(0.1),
                        name='Bottleneck'
                        )(tf.keras.layers.Concatenate(axis=1)(layer1))
        
        return Outputs
        
        
    def Decoder(inputs, Dim, in_out_neuron, lambda_act=0, lambda_weight=0, sd=0):
        layer1 = [Dense(Dim[i], activation="relu",
                        activity_regularizer=regularizers.l1(lambda_act),
                        kernel_regularizer=regularizers.l2(lambda_weight), 
                        kernel_initializer=initializers.he_uniform(sd), 
                        bias_initializer = initializers.Constant(0.1),
                        name = 'Modality'+str(i+1)+'_Decoding_layer'
                        )(inputs) for i in range(len(in_out_neuron))]
        
        
        Outputs = [Dense(in_out_neuron[i], activation="linear",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(sd), 
                      bias_initializer = tf.keras.initializers.Zeros(),
                      name = 'Modality'+str(i+1)+'_Reconstruction_layer'
                      )(layer1[i]) for i in range(len(in_out_neuron))]
        
        
        return Outputs
        
        
    def Build_UMINT(data, Dim, mid=64, lambda_act=0, lambda_weight=0, sd=0):
        in_out_neuron = [i.shape[1] for i in data]
        inputs = [Input(shape=(i,)) for i in in_out_neuron]
        
        Enc = UMINT.Encoder(inputs, Dim, mid=mid, lambda_act=lambda_act, lambda_weight=lambda_weight, sd=sd)
        Dec = UMINT.Decoder(Enc, Dim, in_out_neuron, lambda_act=lambda_act, lambda_weight=lambda_weight, sd=sd)
        
        umint = Model(inputs = inputs, outputs = Dec, name = "UMINT")
        encoder = Model(inputs = inputs, outputs = Enc, name = "UMINT_encoder")
        decoder = Model(inputs = Enc, outputs = Dec, name = "UMINT_decoder")
        
        return umint, encoder, decoder
        
        
    def train_umint(data, hid_neuron, mid=64, bs=16, epochs=25, lambda_act=0, lambda_weight=0, sd=0, val=None, verbose=1):
        
        Umint, UmintEncoder, UmintDecoder = UMINT.Build_UMINT(data, hid_neuron, mid=mid, lambda_act=lambda_act, lambda_weight=lambda_weight, sd=sd)
        Umint.compile(optimizer=Adam(), loss=mean_squared_error)
        
        if val == None:
            Umint.fit(x=data, y=data, batch_size=bs, epochs=epochs, verbose=verbose)
        else:
            Umint.fit(x=data, y=data, validation_data=(val, val), batch_size=16, epochs=25, verbose=verbose)
        
        return Umint, UmintEncoder, UmintDecoder
    
    
    
    ############################  Modality Influence Score #########################################
    
    def modalityRank(encoder, data):
        modality = len(data)
        idx = ['Modality_'+str(i) for i in range(modality)]
        weights = encoder.get_weights()
        hid_neuron = [weights[2*i].shape[1] for i in range(modality)]
        splits = [sum(hid_neuron[0:x:1]) for x in range(modality+1)]
        
        hidden_in = [np.dot(np.array(data[i]), weights[2*i]) + weights[2*i+1] for i in range(modality)]
        hidden_out = [np.maximum(hidden_in[i], 0) for i in range(modality)]
        mid_in = [np.dot(hidden_out[i], weights[2*modality][splits[i]:splits[i+1],:]) for i in range(modality)]
        
        bscore = [np.linalg.norm(mid_in[i], axis =0) for i in range(modality)]
        total = np.sum(bscore, axis = 0)
        contrib = bscore/total
        
        modScore = [np.linalg.norm(contrib[i]) for i in range(modality)]
        modScore = [i/sum(modScore) for i in modScore]
        
        contrib = pd.DataFrame(contrib, index=idx)
        modScore = pd.DataFrame(modScore, index=idx).T
        
        return modScore, contrib
    
    ############################  UMINTFS (Unsupervised)  ##########################################
    
    
    
    def Encoder_scores(weights, data, splits):
        
        modality = len(data)
        
        hidden_in = [np.dot(np.array(data[i]), weights[2*i]) + weights[2*i+1] for i in range(modality)]
        hidden_out = [np.maximum(hidden_in[i], 0) for i in range(modality)]
        mid_in = [np.dot(hidden_out[i], weights[2*modality][splits[i]:splits[i+1],:]) for i in range(modality)]
        
        #total = np.sum(np.abs(mid_in), axis=0)
        #feature_score = [np.mean(np.abs(mid_in[i])/total, axis=0) for i in range(modality)]
        feature_score = [np.mean(np.abs(mid_in[i]), axis=0) for i in range(modality)]
        
        feature_score_norm = [feature_score[i]/np.max(feature_score[i]) for i in range(modality)]
        #feature_score_norm = feature_score
        
        return feature_score_norm
    
    
    def Decoder_scores(weights, data):
    
        modality = len(data)
        feature_decode = [np.dot(np.abs(weights[2*i]), np.abs(weights[2*(modality+i)])) for i in range(modality)]
        
        feature_score = [la.norm(feature_decode[i], axis=1) for i in range(modality)]
        
        feature_score_norm = [feature_score[i]/np.max(feature_score[i]) for i in range(modality)]
        
        return feature_score_norm
      
        
    def bottleneckScore(data, encoder, decoder):
        
        encoder_weights = encoder.get_weights()
        decoder_weights = decoder.get_weights()
        
        modality = len(data)
        hid_neuron = [encoder.get_weights()[2*i].shape[1] for i in range(modality)]
        splits = [sum(hid_neuron[0:x:1]) for x in range(modality+1)]
        index = ['Modality_'+str(i+1) for i in range(modality)]
        
        Encoder_score = UMINT.Encoder_scores(encoder_weights, data, splits)
        Decoder_score = UMINT.Decoder_scores(decoder_weights, data)

        total_score = (np.array(Encoder_score) + np.array(Decoder_score))/2.0
        
        bScore_df = pd.DataFrame(total_score, index=index)
        
        return bScore_df
          
    
    def top_feature(data, encoder, decoder):
        
        encoder_weights = encoder.get_weights()
        decoder_weights = decoder.get_weights()
        
        modality = len(data)
        hid_neuron = [encoder.get_weights()[2*i].shape[1] for i in range(modality)]
        splits = [sum(hid_neuron[0:x:1]) for x in range(modality+1)]
        
        bScore_df = UMINT.bottleneckScore(data, encoder, decoder)

        feature_weights = [np.dot(np.abs(encoder_weights[2*i]),
                                  np.abs(encoder_weights[2*modality][splits[i]:splits[i+1],:]))
                           for i in range(modality)]
        
        features_score = [la.norm(feature_weights[i] * np.array(bScore_df)[i], axis=1) for i in range(modality)]
        
        
        features_score_df = [pd.DataFrame() for i in range(modality)]
        for i in range(modality):
            features_score_df[i]['feature_name'] = data[i].columns 
            features_score_df[i]['feature_score'] = features_score[i]

        return features_score_df
    
    
    
    def UMINTFS(data, encoder, decoder, num_fs, FStype='top'):
        
        features_score_df = UMINT.top_feature(data, encoder, decoder)
        top_features = []

        if FStype == 'top':
            for i in range(len(data)):
                top_features.append((features_score_df[i].nlargest(num_fs[i], 'feature_score')).reset_index(drop=True))
        else:
            for i in range(len(data)):
                top_features.append((features_score_df[i].nsmallest(num_fs[i], 'feature_score')).reset_index(drop=True))

        return top_features
        
        

    ################################  UMINTFS (Supervised)  ##########################################	
    
    

    def Find_All_Markers(data, y, encoder, num_fs):
        
        markers = {}
        for cell_type in tqdm(np.unique(y)):
        	markers[cell_type] = UMINT.Find_Markers(data, y, encoder, cell_type, num_fs)
        	
        return markers
    
    def Find_Markers(data, y, encoder, cell_type, num_fs):
        # parameters
        modality = len(data)
        hid_neuron = [encoder.get_weights()[2*i].shape[1] for i in range(modality)]
        splits = [sum(hid_neuron[0:x:1]) for x in range(modality+1)]
        
        # Labels encoding
        le = preprocessing.LabelEncoder()
        y = pd.DataFrame(le.fit_transform(np.ravel(y)),index=data[0].index)
        y_classes = le.classes_
        loc = list(y_classes).index(cell_type)
        
        # labels for one vs rest classification
        new_lab = [1 if i==loc else 0 for i in y[0]]
        
        # An MLP Classifier
        inp = encoder.get_layer('Bottleneck').output
        hid = Dense(32, activation='relu', name='hidden')(inp)
        out = Dense(1, activation='sigmoid', name='predictions')(hid)
        classifier = Model(inputs=encoder.inputs, outputs=out)
        
        # Training a classifier
        classifier.compile(loss='binary_crossentropy', optimizer='adam')
        classifier.fit(data, np.array(new_lab), epochs=25, batch_size=64,verbose=0)
        
        # Extracting the trained weights
        cl = classifier.get_weights()
        
        # Celltype specific samples
        sub_data = [data[i][y[0]==loc] for i in range(modality)]
        
        # Assigning a score
        res_data = [np.dot(np.dot(np.dot(abs(cl[2*i]), abs(cl[2*len(data)][splits[i]:splits[i+1],:])),abs(cl[6])), abs(cl[8])).reshape((data[i].shape[1],)) for i in range(modality)]
        res_data_all = [[res_data[j]*np.array(sub_data[j].iloc[i,:]) for i in range(sub_data[j].shape[0])] for j in range(modality)]
        res_data_all_avg = [pd.DataFrame(np.mean(res_data_all[i], axis=0)) for i in range(modality)]
        top_features_data = [res_data_all_avg[i].nlargest(num_fs[i], 0) for i in range(modality)]
        
        # Finding Markers
        markers = [list(data[i].columns[top_features_data[i].index]) for i in range(modality)]
        
        return markers
    

    
    ############################  Several Plots  ##########################################    
    
    def contributionHeatmap(df, **kwargs):
        
        index = kwargs['index']
        
        #top_ft, temp1 = UMINT.top_feature(data, encoder, decoder, top, hid_neuron, index=index, thresold=thres)
        #temp1 = topBottleneckScore(df, top, **kwargs)
        
        plt.figure(figsize=(10,10))
        plt.imshow(df, cmap='Purples')
        plt.yticks(range(df.shape[0]), df.index)
        plt.xticks(range(df.shape[1]), range(df.shape[1]))
        plt.colorbar(orientation = 'horizontal')
        plt.show()
        
        
    def plotDistributions(df, **kwargs):

        index = kwargs['index']
        row = kwargs['row']
        col = kwargs['col']
        count = kwargs['count']
        ft = kwargs['fontsize']
        fs = kwargs['figsize']
        thres = kwargs['thresold']
        if thres == None:
            fig = plt.figure(figsize=fs)
            for i in range(row):
                for j in range(col):
                    fig.add_subplot(row, col, count)
                    df.T[index[count-1]].hist()
                    plt.xlabel(index[j], fontsize=ft)
                    count = count + 1
            plt.show()
            
        else:
            fig = plt.figure(figsize=fs)
            for i in range(row):
                for j in range(col):
                    fig.add_subplot(row, col, count)
                    df.T[index[count-1]].hist()
                    plt.axvline(df.T.quantile(thres[j][i])[0], color='r', linestyle='--')
                    plt.axvline(df.T.quantile(thres[j][i+1])[0], color='r', linestyle='--') 
                    plt.xlabel(index[j], fontsize=ft)
                    count = count + 1
            plt.show()

    
    def ADT_plots(data, adt, top_features):
        fig = plt.figure(figsize=(25, 10))
        row = 2
        col = 5
        count = 1
        ft = 11.75
        if type(top_features[0]) == pd.core.frame.DataFrame:
            for i in range(row):
                for j in range(col):
                    fig.add_subplot(row, col, count)
                    plt.scatter(data[:,0], data[:,1], c=adt[top_features[1]['feature_name'][count-1]],
                                cmap='Blues', s=2)
                    plt.colorbar()
                    plt.xlabel('UMAP1', fontsize=ft) 
                    plt.ylabel('UMAP2', fontsize=ft) 
                    # displaying the title
                    plt.title(top_features[1]['feature_name'][count-1], fontsize=ft)
                    count = count + 1
        else:  
            for i in range(row):
                for j in range(col):
                    fig.add_subplot(row, col, count)
                    plt.scatter(data[:,0], data[:,1], c=adt[top_features[count-1]], cmap='Blues', s=2)
                    plt.colorbar()
                    plt.xlabel('UMAP1', fontsize=ft) 
                    plt.ylabel('UMAP2', fontsize=ft) 
                    # displaying the title
                    plt.title(top_features[count-1], fontsize=ft)
                    count = count + 1
            
        plt.show()
