#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:32:27 2019

@author: marionaidoo
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, SpectralClustering #For clustering
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import os                     # For os related operations
import sys                    # For data size


np.random.seed(0) 
# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Format / Surpress Exponential notation

def get_PCA(x_train,x_valid,x_test,num_of_comp = 2):
 
    pca = PCA(n_components = num_of_comp)
    
    pca.fit(x_train)
    
    principalComponents_train = pca.transform(x_train)
    principalComponents_valid = pca.transform(x_valid)
    principalComponents_test = pca.transform(x_test)
    
    PcaDf_train = pd.DataFrame(data = principalComponents_train
                 , columns = ['pca1', 'pca2'])
    
    PcaDf_test = pd.DataFrame(data = principalComponents_test
                 , columns = ['pca1', 'pca2'])
    
    PcaDf_valid = pd.DataFrame(data = principalComponents_valid
                 , columns = ['pca1', 'pca2'])
    
    return PcaDf_train,PcaDf_valid,PcaDf_test

def get_kMeans(df_train,outliers_fraction,n_clust=7):
    kmeans_model = KMeans(n_clusters=n_clust, random_state=42,n_jobs=-1).fit(df_train)
    
    #########
    x_clusters = kmeans_model.predict(df_train) #predicting the K-Means using Test dataset
    x_clust_center=kmeans_model.cluster_centers_ #get the centres
    #Euclidean Distance between centres
    dist = [np.linalg.norm(x-y) for x,y in zip(pd.DataFrame(df_train).values,x_clust_center[x_clusters])] 

    km_y_pred=np.array(dist) #Convert to an array
    km_y_pred[dist>=np.percentile(dist,99)]=1 #Anomalous score based on the 99% percentile
    km_y_pred[dist<np.percentile(dist,99)]=0
    
    #df_train['Id'] = pd.Series(x_clusters, index = df_train.index) #Sert the Index to the Training set
    #df_train['AnomalyInd'] = pd.Series(km_y_pred, index = df_train.index) #Sert the Index to the Training set
    
    df_merged = df_train.merge(pd.DataFrame(x_clusters), how='left', left_index=True, right_index=True)
    df_merged_final = df_merged.merge(pd.DataFrame(km_y_pred), how='left', left_index=True, right_index=True)
   ###########Save the Model

  #  return kmeans_model,x_clusters,x_clust_center,km_y_pred #df_train is inclusive of centre values
    return kmeans_model,df_merged_final,x_clust_center,km_y_pred #df_train is inclusive of centre values


def get_testkMeans(df_test,kmeans_model,outliers_fraction,n_clust=7):

    #########
    x_clusters = kmeans_model.predict(df_test) #predicting the K-Means using Test dataset
    x_clust_center=kmeans_model.cluster_centers_ #get the centres
    #Euclidean Distance between centres
    dist = [np.linalg.norm(x-y) for x,y in zip(pd.DataFrame(df_test).values,x_clust_center[x_clusters])] 

    km_y_pred=np.array(dist) #Convert to an array
    km_y_pred[dist>=np.percentile(dist,99)]=1 #Anomalous score based on the 99% percentile
    km_y_pred[dist<np.percentile(dist,99)]=0
    
    #df_train['Id'] = pd.Series(x_clusters, index = df_train.index) #Sert the Index to the Training set
    #df_train['AnomalyInd'] = pd.Series(km_y_pred, index = df_train.index) #Sert the Index to the Training set
    
    df_merged = df_test.merge(pd.DataFrame(x_clusters), how='left', left_index=True, right_index=True)
    df_merged_final = df_merged.merge(pd.DataFrame(km_y_pred), how='left', left_index=True, right_index=True)
   ###########Save the Model

  #  return kmeans_model,x_clusters,x_clust_center,km_y_pred #df_train is inclusive of centre values
    return kmeans_model,df_merged_final,x_clust_center,km_y_pred #df_train is inclusive of centre values

def get_clustermodel(df_train,df_test,outliers_fraction):
    #Remove once pfunction in implemented
    #df_train = s_train
    #df_test = s_test
    
    ocsvm_max_train = 10000
    n_samples_train = df_train.shape[0]
      
     # define models:
    iforest = IsolationForest(max_samples=100, random_state=42, behaviour="new", contamination=outliers_fraction)
    lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30,metric='minkowski',contamination=outliers_fraction,novelty=True)
    ocsvm = OneClassSVM(kernel='linear',gamma='auto', coef0=0.0, tol=0.001, nu=outliers_fraction, \
                shrinking=True, cache_size=500, verbose=False, max_iter=-1)
    print('end of iForest,lof and OCSVM model creation')
    
    iforest_model = iforest.fit(df_train)
    print('end of iForest model training')
    
    #Local Outlier Factor only looks at the local neighbourhood of a data point and hence cannot make predictions on out of sample data points. 
    #Hence we work directly with X_test here.
    lof_model = lof.fit(df_train) 
    print('Local Outlier Factor test model completed')
    ocsvm_model =  ocsvm.fit(df_train[:min(ocsvm_max_train, n_samples_train - 1)])
    print('end of ocsvm model training!')
    
    #Anomaly Score
    iforest_anomalyscore = iforest.decision_function(df_test)#Predicts the anomaly score
    lof_anomalyscore = lof.decision_function(df_test)
    ocsvm_anomalyscore = ocsvm.decision_function(df_test)
    print('end of models - Anomaly score!')  
    
    

    #Outliers / Anomaly data Points
    #LOF - Use the Negative Factor (Value is output in Negative so get the distcint)
    
   # lof_outlier = lof_model.predict(df_test)
    #iforest_outlier = iforest_model.predict(df_test)
    #ocsvm_outlier = ocsvm_model.predict(df_test)
    

    
    #lof_y_pred=np.array(lof_outlier) #Convert to an array
    #lof_y_pred[lof_y_pred == 1] = 0
    #lof_y_pred[lof_y_pred == -1] = 1 #Anomalous score based LOF prediction

    #iforest_y_pred=np.array(iforest_outlier) #Convert to an array
    #iforest_y_pred[iforest_y_pred == 1] = 0
   # iforest_y_pred[iforest_y_pred == -1] = 1 #Anomalous score based iForest prediction
    
    #ocsvm_y_pred=np.array(ocsvm_outlier) #Convert to an array
    #ocsvm_y_pred[ocsvm_y_pred * (-1) == -1] = 1 #Anomalous score based OCSVM prediction
    #ocsvm_y_pred[ocsvm_y_pred * (-1) == 1] = 0
    
#    iforest_y_pred=np.array(iforest_anomalyscore) #Convert to an array
#    iforest_y_pred[iforest_anomalyscore>=np.percentile(iforest_anomalyscore,99)]=1 #Anomalous score based on the 99% percentile
#    iforest_y_pred[iforest_anomalyscore<np.percentile(iforest_anomalyscore,99)]=0
    
   # ocsvm_y_pred=np.array(ocsvm_anomalyscore) #Convert to an array
    #ocsvm_y_pred[ocsvm_anomalyscore>=np.percentile(ocsvm_anomalyscore,99)]=1 #Anomalous score based on the 99% percentile
    #ocsvm_y_pred[ocsvm_anomalyscore<np.percentile(ocsvm_anomalyscore,99)]=0
    
    return iforest_model,lof_model,ocsvm_model ,iforest_anomalyscore,lof_anomalyscore,ocsvm_anomalyscore



def get_predictclustermodel(lof_model,iforest_model,ocsvm_model,df_test):

#Outliers / Anomaly data Points
    #LOF - Use the Negative Factor (Value is output in Negative so get the distcint)
    
    lof_outlier = lof_model.predict(df_test)
    iforest_outlier = iforest_model.predict(df_test)
    ocsvm_outlier = ocsvm_model.predict(df_test)
    

    
    lof_y_pred=np.array(lof_outlier) #Convert to an array
    lof_y_pred[lof_y_pred == 1] = 0
    lof_y_pred[lof_y_pred == -1] = 1 #Anomalous score based LOF prediction

    iforest_y_pred=np.array(iforest_outlier) #Convert to an array
    iforest_y_pred[iforest_y_pred == 1] = 0
    iforest_y_pred[iforest_y_pred == -1] = 1 #Anomalous score based iForest prediction
    
    ocsvm_y_pred=np.array(ocsvm_outlier) #Convert to an array
    ocsvm_y_pred[ocsvm_y_pred * (-1) == -1] = 1 #Anomalous score based OCSVM prediction
    ocsvm_y_pred[ocsvm_y_pred * (-1) == 1] = 0
    
    #df_test['iForestAnomalyInd'] = pd.Series(iforest_y_pred, index = df_test.index) #Sert the Index to the Training set
    #df_test['lofAnomalyInd'] = pd.Series(lof_y_pred, index = df_test.index) #Sert the Index to the Training set
    #df_test['ocsvmAnomalyInd'] = pd.Series(ocsvm_y_pred, index = df_test.index) #Sert the Index to the Training set
    
    df_merged_1 = df_test.merge(pd.DataFrame(iforest_y_pred), how='left', left_index=True, right_index=True)
    df_merged_2 = df_merged_1.merge(pd.DataFrame(lof_y_pred), how='left', left_index=True, right_index=True)
    df_test = df_merged_2.merge(pd.DataFrame(ocsvm_y_pred), how='left', left_index=True, right_index=True)
    
    df_test.rename(columns={df_test.iloc[:,-1].name : 'ocsvmAnomalyInd',
                         df_test.iloc[:,-2].name : 'lofAnomalyInd',
                          df_test.iloc[:,-3].name : 'iForestAnomalyInd'}, inplace=True)
    
    return iforest_y_pred,lof_y_pred,ocsvm_y_pred,df_test