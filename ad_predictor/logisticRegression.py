from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

class logisticRegression():

    def __init__(self, data, corr, regularization):
        self.data = data
        self.corr = corr
        self.regularization = regularization

    def classifier(self):
        features = self.corr
        X = self.data[features]

        # Scaled features:normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)

        y = self.data['score']
        X_train, X_test, y_train, y_test = (
            train_test_split(X_scaled, y, random_state=43))

        clf = LogisticRegression(C=self.regularization).fit(X_train, y_train)
        prediction_LR = clf.predict(X_test)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_LR)
        return precision, recall, (y_test == prediction_LR).sum() / len(y_test)