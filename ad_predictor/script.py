import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import data_preprocessing
import logistic_regression, svmClassifier, kNearestNeighbors, randomForest

os.chdir(r'C:\Users\Katerina\Documents\GitHub\ml_projects\ad_predictor')
#read data
train_data = pd.read_csv('data/train.csv')

#data balance
score_split = train_data.groupby(['score']).count()
score_split.reset_index(inplace=True)
score_split = pd.DataFrame(score_split,columns=['score','V2'])
score_split.rename(columns={'V2':'count'},inplace=True)

#data preprocessing
preprocess = data_preprocessing.preprocessing(train_data)
categorical_var = preprocess.split_numerical_categorical()[1]
numerical_var = preprocess.split_numerical_categorical()[0]
low_corr_var = preprocess.numerical_feature_selection(numerical_var)

#logistic regression classifier
score_LR = {}
for C in [0.1,1,10,100,1000]:
    regr = logistic_regression.logisticRegression(train_data, low_corr_var, C)
    score_LR[C] = regr.classifier()

score_LR_df = pd.DataFrame(list(score_LR.values()),index=list(score_LR.keys()),columns=['precision','recall','accuracy'])
score_LR_df[['precision0','precision1']] = score_LR_df['precision'].apply(pd.Series)
score_LR_df[['recall0','recall1']] = score_LR_df['recall'].apply(pd.Series)
score_LR_df = pd.DataFrame(score_LR_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_LR_df);

#SVM classifier
score_svm = {}
for C in [0.1,1,10,100,1000]:
    regr = svmClassifier.svmClassifier(train_data, low_corr_var, C)
    score_svm[C] = regr.classifier()

score_svm_df = pd.DataFrame(list(score_svm.values()),index=list(score_svm.keys()),columns=['precision','recall','accuracy'])
score_svm_df[['precision0','precision1']] = score_svm_df['precision'].apply(pd.Series)
score_svm_df[['recall0','recall1']] = score_svm_df['recall'].apply(pd.Series)
score_svm_df = pd.DataFrame(score_svm_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_svm_df);

#K Nearest Neighbors
score_knn = {}
for n in [1,3,5,10]:
    regr = kNearestNeighbors.kNearestNeighbors(train_data, low_corr_var, n)
    score_knn[n] = regr.classifier()

score_knn_df = pd.DataFrame(list(score_knn.values()),index=list(score_knn.keys()),columns=['precision','recall','accuracy'])
score_knn_df[['precision0','precision1']] = score_knn_df['precision'].apply(pd.Series)
score_knn_df[['recall0','recall1']] = score_knn_df['recall'].apply(pd.Series)
score_knn_df = pd.DataFrame(score_knn_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_knn_df);

#Randon Forest Classifier
score_rf = {}
corr_numeric = train_data[list(numerical_var)].corr()
for n in [5,10,50,100,200]:
    for d in [None,2,5,10]:
        for l in [1,3,5,10]:
            regr = randomForest.randomForest(train_data, corr_numeric, min_corr=0.0005, estimators=n, depth=d, leaf=l)
            score_rf[n,d,l] = regr.classifier()

score_rf_df = pd.DataFrame(list(score_rf.values()),index=list(score_rf.keys()),columns=['precision','recall','accuracy'])
score_rf_df[['precision0','precision1']] = score_rf_df['precision'].apply(pd.Series)
score_rf_df[['recall0','recall1']] = score_rf_df['recall'].apply(pd.Series)
score_rf_df = pd.DataFrame(score_rf_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig = plt.figure(figsize=(8,5))
plt.plot(list(score_rf_df['recall1']))
plt.plot(list(score_rf_df['precision1']))
plt.plot(list(score_rf_df['accuracy']))
plt.xlabel('RF set of hyperparameters (estimators,depth, min_sample_leaf)')
plt.legend(['recall1','precision1','accuracy']);

score_RF_corr = {}
for corr in [0.0008,0.0005,0.0002]:
    regr = randomForest.randomForest(train_data, corr_numeric, min_corr=corr, estimators=200, depth=None, leaf=1)
    score_RF_corr[corr] = regr.classifier()

score_RF_corr_df = pd.DataFrame(list(score_RF_corr.values()),index=list(score_RF_corr.keys()),columns=['precision','recall','accuracy'])
score_RF_corr_df[['precision0','precision1']] = score_RF_corr_df['precision'].apply(pd.Series)
score_RF_corr_df[['recall0','recall1']] = score_RF_corr_df['recall'].apply(pd.Series)
score_RF_corr_df = pd.DataFrame(score_RF_corr_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_RF_corr_df);
plt.savefig('figs/rf_heatmap.png')