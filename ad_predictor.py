import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')

score_split = train_data.groupby(['score']).count()
score_split.reset_index(inplace=True)
score_split = pd.DataFrame(score_split,columns=['score','V2'])
score_split.rename(columns={'V2':'count'},inplace=True)

#split numerical from categorical
train_data.replace([np.inf, -np.inf], np.nan,inplace=True)#replace inf with na
train_data.fillna(0,inplace=True)#replace na with 0

num_descr = pd.DataFrame(train_data.describe())
numerical_var=list(num_descr.columns[1:])
categorical_var=[i for i in list(train_data.columns) if i not in list(num_descr.columns)]

#numerical feature selection
corr_numeric = train_data[list(numerical_var)].corr()
import seaborn as sns; sns.set()
fig, ax = plt.subplots(figsize=(20,10))
ax = sns.heatmap(corr_numeric)

low_corr = corr_numeric[abs(corr_numeric)<0.0005]
low_corr.fillna(0,inplace=True)
low_corr_var = [i for i in list(low_corr.sum().index) if low_corr.sum()[i]!=0]

#Logistic regression classifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def Logistic_Regression(regularization):
    features = low_corr_var
    X = train_data[features]

    # Scaled features:normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    y = train_data['score']
    X_train, X_test, y_train, y_test = (
        train_test_split(X_scaled, y, random_state=43))

    clf = LogisticRegression(C=regularization).fit(X_train, y_train)
    prediction_LR = clf.predict(X_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_LR)
    return precision, recall, (y_test == prediction_LR).sum() / len(y_test)

score_LR = {}
for C in [0.1,1,10,100,1000]:
    score_LR[C] = Logistic_Regression(C)

score_LR_df = pd.DataFrame(list(score_LR.values()),index=list(score_LR.keys()),columns=['precision','recall','accuracy'])
score_LR_df[['precision0','precision1']] = score_LR_df['precision'].apply(pd.Series)
score_LR_df[['recall0','recall1']] = score_LR_df['recall'].apply(pd.Series)
score_LR_df = pd.DataFrame(score_LR_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_LR_df);

#SVM classifier
from sklearn import svm


def SVM(regularization):
    features = low_corr_var
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

    clf = svm.SVC(C=regularization)
    clf_svm = clf.fit(x_train, y_train)
    prediction_svm = clf_svm.predict(x_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_svm)
    return precision, recall, (y_test == prediction_svm).sum() / len(y_test)

score_svm = {}
for C in [0.1,1,10,100,1000]:
    score_svm[C] = SVM(C)

score_svm_df = pd.DataFrame(list(score_svm.values()),index=list(score_svm.keys()),columns=['precision','recall','accuracy'])
score_svm_df[['precision0','precision1']] = score_svm_df['precision'].apply(pd.Series)
score_svm_df[['recall0','recall1']] = score_svm_df['recall'].apply(pd.Series)
score_svm_df = pd.DataFrame(score_svm_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_svm_df);

#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


def K_nearest_neighbors(neighbors):
    features = low_corr_var
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf_knn = clf.fit(x_train, y_train)
    prediction_knn = clf_knn.predict(x_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_knn)
    return precision, recall, (y_test == prediction_knn).sum() / len(y_test)

score_knn = {}
for n in [1,3,5,10]:
    score_knn[n] = K_nearest_neighbors(n)

score_knn_df = pd.DataFrame(list(score_knn.values()),index=list(score_knn.keys()),columns=['precision','recall','accuracy'])
score_knn_df[['precision0','precision1']] = score_knn_df['precision'].apply(pd.Series)
score_knn_df[['recall0','recall1']] = score_knn_df['recall'].apply(pd.Series)
score_knn_df = pd.DataFrame(score_knn_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_knn_df);

#Randon Forest Classifier
def feature_sel(min_corr):
    low_corr = corr_numeric[abs(corr_numeric)<min_corr]
    low_corr.fillna(0,inplace=True)
    low_corr_var = [i for i in list(low_corr.sum().index) if low_corr.sum()[i]!=0]
    return low_corr_var


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score


def RandomForest(min_corr, estimators, depth, leaf):
    features = feature_sel(min_corr)
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

    # random forest classifier with n_estimators=10 (default)
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, min_samples_leaf=leaf,
                                 n_jobs=-1, random_state=43)
    clf_rf = clf.fit(x_train, y_train)
    prediction_rf = clf_rf.predict(x_test)
    # print(estimators,depth,leaf, clf,accuracy_score(y_test,prediction_rf))

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_rf)
    return precision, recall, (y_test == prediction_rf).sum() / len(y_test)

score_rf = {}
for n in [5,10,50,100,200]:
    for d in [None,2,5,10]:
        for l in [1,3,5,10]:
            score_rf[n,d,l] = RandomForest(0.0005,n,d,l)

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
    score_RF_corr[corr] = RandomForest(corr,estimators=200,depth=None,leaf=1)

score_RF_corr_df = pd.DataFrame(list(score_RF_corr.values()),index=list(score_RF_corr.keys()),columns=['precision','recall','accuracy'])
score_RF_corr_df[['precision0','precision1']] = score_RF_corr_df['precision'].apply(pd.Series)
score_RF_corr_df[['recall0','recall1']] = score_RF_corr_df['recall'].apply(pd.Series)
score_RF_corr_df = pd.DataFrame(score_RF_corr_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.heatmap(score_RF_corr_df);

#Random Forest CV
from sklearn.model_selection import KFold, cross_val_score

features=feature_sel(min_corr=0.0005)
X = train_data[features]
min_max_scaler = preprocessing.MinMaxScaler()
x_1 = min_max_scaler.fit_transform(X)

y=train_data['score']

rf = RandomForestClassifier(n_jobs = -1,n_estimators=200,max_depth=None,random_state=43)
k_fold = KFold(n_splits=5)
score_cv = cross_val_score(rf,x_1,y,cv=k_fold,n_jobs=-1)

score_cv_a = cross_val_score(rf,x_1,y,cv=k_fold,n_jobs=-1,scoring='accuracy')
score_cv_p = cross_val_score(rf,x_1,y,cv=k_fold,n_jobs=-1,scoring='precision')
score_cv_r = cross_val_score(rf,x_1,y,cv=k_fold,n_jobs=-1,scoring='recall')
score_cv_f = cross_val_score(rf,x_1,y,cv=k_fold,n_jobs=-1,scoring='f1')

fig = plt.figure(figsize=(15,8))
plt.plot(score_cv_a)
plt.plot(score_cv_p)
plt.plot(score_cv_r)
plt.plot(score_cv_f)
plt.legend(['accuracy','prediction','recall','f1']);

#Gradient Boosting Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score


def GradientBoosting_GridSearch(estimators, depth):
    features = feature_sel(0.0005)
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

    # random forest classifier with n_estimators=10 (default)
    clf = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth, random_state=43)
    clf_gb = clf.fit(x_train, y_train)
    prediction_gb = clf_gb.predict(x_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_gb)
    return precision, recall, (y_test == prediction_gb).sum() / len(y_test)

score_gb = {}
for n in [5,10,50,100,200]:
    for d in [None,2,5,10,15]:
        score_gb[n,d] = GradientBoosting_GridSearch(n,l)

score_gb_df = pd.DataFrame(list(score_gb.values()),index=list(score_gb.keys()),columns=['precision','recall','accuracy'])
score_gb_df[['precision0','precision1']] = score_gb_df['precision'].apply(pd.Series)
score_gb_df[['recall0','recall1']] = score_gb_df['recall'].apply(pd.Series)
score_gb_df = pd.DataFrame(score_gb_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig = plt.figure(figsize=(8,5))
plt.plot(list(score_gb_df['recall1']))
plt.plot(list(score_gb_df['precision1']))
plt.plot(list(score_gb_df['accuracy']))
plt.xlabel('Gradient Boosting set of hyperparameters (estimators,depth)')
plt.legend(['recall1','precision1','accuracy']);

#Categorical Feauture selection
train_data['country_scale'] = list(pd.factorize(train_data['V4'])[0])
train_data['ad_text'] = train_data['V5'].str.len()
train_data.replace([np.inf, -np.inf], np.nan,inplace=True)#replace inf with na
train_data.fillna(0,inplace=True)#replace na with 0
train_data['sector_scale'] = list(pd.factorize(train_data['V6'])[0])
train_data['category_scale'] = list(pd.factorize(train_data['V42'])[0])

sets = [train_data['ad_text'][train_data['score']==1],train_data['ad_text'][train_data['score']==0]]
fig = plt.figure(figsize=(8,5))
plt.hist(sets,normed=True)
plt.ylabel('Normalized frequency')
plt.xlabel('Text length')
plt.title('Ad score dependence on text length');

# Create CountVectorizer from keywords vocabulary
keywords = [i.split(',') for i in list(set(train_data['V61']))]
keys = [i for j in keywords for i in j]
keys = list((set(keys)))

from sklearn.feature_extraction.text import CountVectorizer


def Countvectorizer_matrix(dataset, column_name, keywords):
    count_vector = CountVectorizer(vocabulary=keywords)
    keyword_matrix = count_vector.fit_transform(list(dataset[column_name]))  # .toarray()

    return keyword_matrix

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add)], 'csr')

#Merging all categorical features to train the classifier
def X_categorical(dataset,keys):
    keyword_matrix = Countvectorizer_matrix(train_data,'V61',keys)

    X_keyword = keyword_matrix
    new_features = dataset[['country_scale','ad_text','sector_scale','category_scale']]
    X_categorical = add_feature(X_keyword,new_features)
    return X_categorical


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score


def RandomForest_c(estimators, depth, leaf):
    X = X_categorical(train_data, keys)
    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    # random forest classifier with n_estimators=10 (default)
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, min_samples_leaf=leaf,
                                 n_jobs=-1, random_state=43)
    clf_rf_c = clf.fit(x_train, y_train)
    prediction_rf_c = clf_rf_c.predict(x_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_rf_c)
    return precision, recall, (y_test == prediction_rf_c).sum() / len(y_test)

score_rf_c = {}
for n in [5,10,50,100,200]:
    for d in [None,2,5,10]:
        for l in [1,3,5,10]:
            score_rf_c[n,d,l] = RandomForest_c(n,d,l)

score_rf_c_df = pd.DataFrame(list(score_rf_c.values()),index=list(score_rf_c.keys()),columns=['precision','recall','accuracy'])
score_rf_c_df[['precision0','precision1']] = score_rf_c_df['precision'].apply(pd.Series)
score_rf_c_df[['recall0','recall1']] = score_rf_c_df['recall'].apply(pd.Series)
score_rf_c_df = pd.DataFrame(score_rf_c_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig = plt.figure(figsize=(10,6))
plt.plot(list(score_rf_c_df['recall1']))
plt.plot(list(score_rf_c_df['precision1']))
plt.plot(list(score_rf_c_df['accuracy']))
plt.xlabel('RF for categorical features: set of hyperparameters (estimators,depth,min_sample_leaf)')
plt.legend(['recall1','precision1','accuracy']);

#Optimum hyperparameters for random forest classifier for categorical features
X = X_categorical(train_data,keys)
y=train_data['score']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

#random forest classifier with n_estimators=10 (default)
clf = RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=1,
                             n_jobs=-1,random_state=43)
clf_rf_c = clf.fit(x_train,y_train)
prediction_rf_c = clf_rf_c.predict(x_test)

ac = accuracy_score(y_test,clf_rf_c.predict(x_test))
precision, recall, fscore, support = precision_recall_fscore_support(y_test,clf_rf_c.predict(x_test))
print('Precision: {}, Recall: {}, Accuracy: {}'.format(precision,recall,ac))
cm = confusion_matrix(y_test,clf_rf_c.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d");

#Combine two RF classifiers: numerical + categorical variables

def RF_num(estimators, depth):
    features = feature_sel(0.0005)
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y = train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

    # random forest classifier with n_estimators=10 (default)
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth,
                                 n_jobs=-1, random_state=43)
    clf_rf_num = clf.fit(x_train, y_train)
    prob_rf_num = clf_rf_num.predict_proba(x_test)

    return prob_rf_num

def RF_cat(estimators,depth):

    X = X_categorical(train_data,keys)
    y=train_data['score']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    #random forest classifier with n_estimators=10 (default)
    clf = RandomForestClassifier(n_estimators=estimators,max_depth=depth,
                                 n_jobs=-1,random_state=43)
    clf_rf_c = clf.fit(x_train,y_train)
    prob_rf_c = clf_rf_c.predict_proba(x_test)
    return prob_rf_c,y_test

prob_c = [i[1] for i in RF_cat(200,None)[0]]
clf_cat_num = pd.DataFrame(prob_c,columns=['Categorical'])
clf_cat_num['Numerical'] = [i[1] for i in RF_num(200,None)]
clf_cat_num['score'] = list(RF_cat(200,None)[1])


def RF_combined(estimators, depth, leaf):
    X = clf_cat_num[['Categorical', 'Numerical']]

    y = clf_cat_num['score']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    # random forest classifier with n_estimators=10 (default)
    clf_rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, min_samples_leaf=leaf, random_state=43,
                                    n_jobs=-1)
    clf_rf_all = clf_rf.fit(x_train, y_train)

    prediction_all = clf_rf_all.predict(x_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_all)
    return precision, recall, (y_test == prediction_all).sum() / len(y_test)

score_rf_all = {}
for n in [5,10,50,100,200]:
    for d in [None,2,5,10]:
        for l in [1,3,5,10]:
            score_rf_all[n,d,l] = RF_combined(n,d,l)

score_rf_all_df = pd.DataFrame(list(score_rf_all.values()),index=list(score_rf_all.keys()),columns=['precision','recall','accuracy'])
score_rf_all_df[['precision0','precision1']] = score_rf_all_df['precision'].apply(pd.Series)
score_rf_all_df[['recall0','recall1']] = score_rf_all_df['recall'].apply(pd.Series)
score_rf_all_df = pd.DataFrame(score_rf_all_df,columns=['precision0','precision1','recall0','recall1','accuracy'])

fig = plt.figure(figsize=(8,5))
plt.plot(list(score_rf_all_df['recall1']))
plt.plot(list(score_rf_all_df['precision1']))
plt.plot(list(score_rf_all_df['accuracy']))
plt.xlabel('RF combined set of hyperparameters (estimators,depth, min_sample_leaf)')
plt.legend(['recall1','precision1','accuracy']);

#Score on the training set
X = clf_cat_num[['Categorical','Numerical']]

y=clf_cat_num['score']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=3,random_state=43,n_jobs=-1)
clf_rf_all = clf_rf.fit(x_train,y_train)

prediction_all = clf_rf_all.predict(x_test)

ac = accuracy_score(y_test,clf_rf_all.predict(x_test))
precision, recall, fscore, support = precision_recall_fscore_support(y_test,clf_rf_all.predict(x_test))
print('Precision: {}, Recall: {}, Accuracy: {}'.format(precision,recall,ac))
cm = confusion_matrix(y_test,clf_rf_all.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d");

#Fit model on test set
test_data = pd.read_csv('test.csv')
test_data.replace([np.inf, -np.inf], np.nan,inplace=True)#replace inf with na
test_data.fillna(0,inplace=True)#replace na with 0

def RF_classifier_num():
    features=feature_sel(min_corr=0.0005)
    X = train_data[features]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_1 = min_max_scaler.fit_transform(X)

    y=train_data['score']
    #random forest classifier with n_estimators=10 (default)
    clf = RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=1,
                                 n_jobs=-1,random_state=43)
    clf_rf = clf.fit(x_1,y)
    return clf_rf

def RF_classifier_cat():
    X = X_categorical(train_data,keys)
    y=train_data['score']
    clf = RandomForestClassifier(n_estimators=200,max_depth=None,
                                 n_jobs=-1,random_state=43)
    clf_rf_c = clf.fit(X,y)
    return clf_rf_c

def RF_classifier_combined():
    prob_c = [i[1] for i in RF_cat(200,None)[0]]
    clf_cat_num = pd.DataFrame(prob_c,columns=['Categorical'])
    clf_cat_num['Numerical'] = [i[1] for i in RF_num(200,None)]
    clf_cat_num['score'] = list(RF_cat(200,None)[1])

    X = clf_cat_num[['Categorical','Numerical']]

    y=clf_cat_num['score']

    #random forest classifier with n_estimators=10 (default)
    clf_rf = RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=3,random_state=43,n_jobs=-1)
    clf_rf_all = clf_rf.fit(X,y)
    return clf_rf_all

RF_num_pred = RF_classifier_num().predict_proba(min_max_scaler.fit_transform(test_data[feature_sel(0.0005)]))


def X_test(dataset, keys):
    dataset['country_scale'] = list(pd.factorize(dataset['V4'])[0])
    dataset['ad_text'] = dataset['V5'].str.len()
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace inf with na
    dataset.fillna(0, inplace=True)  # replace na with 0
    dataset['sector_scale'] = list(pd.factorize(dataset['V6'])[0])
    dataset['category_scale'] = list(pd.factorize(dataset['V42'])[0])

    keyword_matrix = Countvectorizer_matrix(dataset, 'V61', keys)

    X_keyword = keyword_matrix
    new_features = dataset[['country_scale', 'ad_text', 'sector_scale', 'category_scale']]
    X_categorical_test = add_feature(X_keyword, new_features)

    return X_categorical_test

RF_cat_pred = RF_classifier_cat().predict_proba(X_test(test_data,keys))

predict_test = [i[1] for i in RF_cat_pred]
clf_cat_num_test = pd.DataFrame(predict_test,columns=['Categorical'])
clf_cat_num_test['Numerical'] = [i[1] for i in RF_num_pred]

RF_combine = RF_classifier_combined().predict_proba(clf_cat_num_test)

pred_prob = [(i,j) for i,j in zip(test_data['creativeId'],[i[1] for i in RF_combine])]
pred_prob_df = pd.DataFrame(pred_prob,columns=['creativeId','probability'])
pred_prob_df.to_csv('probabilities.csv')