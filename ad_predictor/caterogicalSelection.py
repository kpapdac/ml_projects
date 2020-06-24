from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class categoricalSelection():
    def __init__(self, data, column_name, keywords, estimators, depth, leaf):
        self.data = data
        self.column_name = column_name
        self.keywords = keywords
        self.feature_to_add = self.data[['country_scale','ad_text','sector_scale','category_scale']]
        self.estimators = estimators
        self.depth = depth
        self.leaf = leaf

    def Countvectorizer_matrix(self):
        count_vector = CountVectorizer(vocabulary=self.keywords)
        keyword_matrix = count_vector.fit_transform(list(self.data[self.column_name]))  # .toarray()

        return keyword_matrix

    def add_feature(self, matrix):
        """
        Returns sparse feature matrix with added feature.
        feature_to_add can also be a list of features.
        """
        from scipy.sparse import csr_matrix, hstack
        return hstack([matrix, csr_matrix(self.feature_to_add)], 'csr')

    #Merging all categorical features to train the classifier
    def X_categorical(self):
        keyword_matrix = self.Countvectorizer_matrix()

        X_keyword = keyword_matrix
        #new_features = self.data[['country_scale','ad_text','sector_scale','category_scale']]
        X_categorical = self.add_feature(matrix = X_keyword)
        return X_categorical

    def classifier(self):
        X = self.X_categorical()
        y = self.data['score']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

        # random forest classifier with n_estimators=10 (default)
        clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, min_samples_leaf=self.leaf,
                                    n_jobs=-1, random_state=43)
        clf_rf_c = clf.fit(x_train, y_train)
        prediction_rf_c = clf_rf_c.predict(x_test)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction_rf_c)
        return precision, recall, (y_test == prediction_rf_c).sum() / len(y_test)
