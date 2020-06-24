from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class gridSearch():
    def __init__(self, data, corr, min_corr, estimators, depth):
        self.data = data
        self.corr = corr
        self.estimators = estimators
        self.depth = depth
    
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
        clf = GradientBoostingClassifier(n_estimators=self.estimators, max_depth=self.depth, random_state=43)
        clf_gb = clf.fit(x_train, y_train)
        prediction_gb = clf_gb.predict(x_test)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_gb)
        return precision, recall, (y_test == prediction_gb).sum() / len(y_test)
