#https://maviator.github.io
#https://medium.com/@mohtedibf

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# get titanic & test csv files as a DataFrame
train = pd.read_csv(“input/train.csv”)

#Checking for missing data
NAs = pd.concat([train.isnull().sum()], axis=1, keys=[‘Train’])
NAs[NAs.sum(axis=1) > 0]

# At this point we will drop the Cabin feature since it is missing a lot of the data
train.pop(‘Cabin’)
train.pop(‘Name’)
train.pop(‘Ticket’)

# Filling missing Age values with mean
train[‘Age’] = train[‘Age’].fillna(train[‘Age’].mean())

# Filling missing Embarked values with most common value
train[‘Embarked’] = train[‘Embarked’].fillna(train[‘Embarked’].mode()[0])

train[‘Pclass’] = train[‘Pclass’].apply(str)

# Getting Dummies from all other categorical vars
for col in train.dtypes[train.dtypes == ‘object’].index:
 for_dummy = train.pop(col)
 train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)

 # Prepare data for training models
 labels = train.pop(‘Survived’)

 from sklearn.model_selection import train_test_split

 x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)

 from sklearn.ensemble import GradientBoostingClassifier

 model = GradientBoostingClassifier()
 model.fit(x_train, y_train)
 y_pred = model.predict(x_test)

 from sklearn.metrics import roc_curve, auc

 false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
 roc_auc = auc(false_positive_rate, true_positive_rate)

 #learning rates
 learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
 train_results = []
 test_results = []
 for eta in learning_rates:
     model = GradientBoostingClassifier(learning_rate=eta)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(learning_rates, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(learning_rates, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘learning
 rate’)
 plt.show()

 #n_estimators
 n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
 train_results = []
 test_results = []
 for estimator in n_estimators:
     model = GradientBoostingClassifier(n_estimators=estimator)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(n_estimators, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(n_estimators, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘n_estimators’)
 plt.show()

 #max_depth
 max_depths = np.linspace(1, 32, 32, endpoint=True)
 train_results = []
 test_results = []
 for max_depth in max_depths:
     model = GradientBoostingClassifier(max_depth=max_depth)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(max_depths, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(max_depths, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘Tree
 depth’)
 plt.show()

 #min_samples_split
 min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
 train_results = []
 test_results = []
 for min_samples_split in min_samples_splits:
     model = GradientBoostingClassifier(min_samples_split=min_samples_split)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(min_samples_splits, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(min_samples_splits, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘min
 samples
 split’)
 plt.show()

 #min_samples_leaf
 min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
 train_results = []
 test_results = []
 for min_samples_leaf in min_samples_leafs:
     model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(min_samples_leafs, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(min_samples_leafs, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘min
 samples
 leaf’)
 plt.show()

 #max_features
 max_features = list(range(1, train.shape[1]))
 train_results = []
 test_results = []
 for max_feature in max_features:
     model = GradientBoostingClassifier(max_features=max_feature)
     model.fit(x_train, y_train)
     train_pred = model.predict(x_train)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     train_results.append(roc_auc)
     y_pred = model.predict(x_test)
     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
     roc_auc = auc(false_positive_rate, true_positive_rate)
     test_results.append(roc_auc)
 from matplotlib.legend_handler import HandlerLine2D

 line1, = plt.plot(max_features, train_results, ‘b’, label =”Train
 AUC”)
 line2, = plt.plot(max_features, test_results, ‘r’, label =”Test
 AUC”)
 plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
 plt.ylabel(‘AUC
 score’)
 plt.xlabel(‘max
 features’)
 plt.show()