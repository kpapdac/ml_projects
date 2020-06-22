from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

class kNearestNeighbors():
    def __init__(self, data, corr, neighbors):
        self.data = data
        self.corr = corr
        self.neighbors = neighbors

    def classifier(self):
        features = self.corr
        X = self.data[features]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_1 = min_max_scaler.fit_transform(X)

        y = self.data['score']
        x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=43)

        clf = KNeighborsClassifier(n_neighbors=self.neighbors)
        clf_knn = clf.fit(x_train, y_train)
        prediction_knn = clf_knn.predict(x_test)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_knn)
        return precision, recall, (y_test == prediction_knn).sum() / len(y_test)
