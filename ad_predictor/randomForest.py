from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class randomForest():
    def __init__(self, data, corr, min_corr, estimators, depth, leaf):
        self.data = data
        self.corr = corr
        self.min_corr = min_corr
        self.estimators = estimators
        self.depth = depth
        self.leaf = leaf

    def feature_sel(self):
        low_corr = self.corr[abs(self.corr)<self.min_corr]
        low_corr.fillna(0,inplace=True)
        low_corr_var = [i for i in list(low_corr.sum().index) if low_corr.sum()[i]!=0]
        return low_corr_var

    def classifier(self):
        features = self.feature_sel()
        X = self.data[features]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_1 = min_max_scaler.fit_transform(X)

        y = self.data['score']
        x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

        # random forest classifier with n_estimators=10 (default)
        clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, min_samples_leaf=self.leaf,
                                    n_jobs=-1, random_state=43)
        clf_rf = clf.fit(x_train, y_train)
        prediction_rf = clf_rf.predict(x_test)
        # print(estimators,depth,leaf, clf,accuracy_score(y_test,prediction_rf))

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_rf)
        return precision, recall, (y_test == prediction_rf).sum() / len(y_test)