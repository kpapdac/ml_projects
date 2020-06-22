#Economic Sector classification from Company Description

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('Data_ML.xlsx')

data_bysector = data.groupby('TRBC Economic Sector Name').count()
plt.barh(data_bysector.index, data_bysector.Identifier);
data.groupby('TRBC Economic Sector Name').count()

# Description length invariant with Company sector
# (Max character limit 1000!)
data.fillna('',inplace=True)
data['Length'] = data['Business Description'].str.len()
data['CompanyTerms'] = data['Company Name'].str.split()
data['Sector_scale'] = list(pd.factorize(data['TRBC Economic Sector Name'])[0])

#Dataframe manipulation to show list of sector names vs class indices
sector_class = data.groupby(['TRBC Economic Sector Name','Sector_scale']).count()
sector_class.reset_index(inplace=True)
sector_class = pd.DataFrame(sector_class,columns=['TRBC Economic Sector Name','Sector_scale'])
sector_class.head(10)

#Description length distribution doesn't show any particular dependence on sector name
length_bysector = [data['Length'][data['Sector_scale']==0],data['Length'][data['Sector_scale']==1],
                  data['Length'][data['Sector_scale']==2],data['Length'][data['Sector_scale']==3],
                  data['Length'][data['Sector_scale']==4],data['Length'][data['Sector_scale']==5],
                  data['Length'][data['Sector_scale']==6],data['Length'][data['Sector_scale']==7],
                  data['Length'][data['Sector_scale']==8],data['Length'][data['Sector_scale']==9]]

fig=plt.figure(figsize=(15,8))

ax2 = fig.add_subplot(111)
ax2.hist(length_bysector,alpha=0.5,);
ax2.legend(['Basic Materials','Consumer Cyclicals','Consumer Non-Cyclicals','Energy','Financials','Healthcare','Industrials',
          'Technology','Telecommunications Services','Utilities'],fontsize=15)
ax2.set_title('Description Length by sector',size=15)
ax2.tick_params(axis='both', which='major', labelsize=10)

plt.show()

#Lemmatization/ Tokenization process - Tfidf Matrix generation

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

lemmatizer = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r'\w+')

def lemma_tokens(tokens, lemmatizer):
    stemmed = []
    for item in tokens:
        if item.isdigit()==False:
            stemmed.append(lemmatizer.lemmatize(item))
    return stemmed

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    lemmas = lemma_tokens(tokens, lemmatizer)
    return lemmas

# Include Company names in stop words
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
for word in list(data['CompanyTerms']):
    stop_words.union(word)

from sklearn.feature_extraction.text import TfidfVectorizer
def Tfidf_Matrix(text,stoplist):
    tfidf_vectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stoplist)#tokenize text
    tfidf_transf = tfidf_vectorizer.fit_transform(text)#create term-frequency matrix
    return tfidf_transf,tfidf_vectorizer

Tfidf_vector = Tfidf_Matrix(list(data['Business Description']),stop_words)
tfidf_matrix = Tfidf_vector[0]
tfidf_vectoriz = Tfidf_vector[1]

#Visualize tfidf most significant terms for the Finance sector

%matplotlib inline

def Visualize_tfidf_sector(dataset,min_sum):
    sector_indices = np.array(list(dataset.index))
    indices = np.where(sector_indices)[0]
    tfidf_sectordocs = tfidf_matrix.tocsr()[sector_indices,:]
    tfidf_sumsector = np.sum(tfidf_sectordocs,axis=0)

    #Consider as 0 all vector whose tfidf sum score across sector docs is less than the threshold (<min_sum).
    tfidf_sumsector[tfidf_sumsector<min_sum] = 0
    r, c = np.nonzero(tfidf_sumsector[0,:])
    terms = [tfidf_vectoriz.get_feature_names()[c1] for c1 in c]

    fig, ax = plt.subplots(figsize=(20, 10))
    for i in c:
        ax.scatter(i, tfidf_sumsector[0, i],c='y')

    #Annotations
    texts = []

    for i,txt in zip(c,terms):
        value = tfidf_sumsector[0,i]
        texts.append(plt.text(i,value,txt))
    return plt.show()

Finance = data[data['Sector_scale']==9]
Visualize_tfidf_sector(Finance,2)

#Model1

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

#Feature selection, Train/Test split
X = tfidf_matrix
y = data['Sector_scale']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45, shuffle=True)

#Naive Bayes Classifier
model=MultinomialNB(alpha=0.1)
NB_fit = model.fit(X_train,y_train)
prediction = NB_fit.predict(X_test)
score = precision_recall_fscore_support(y_test,prediction)

#Performance
fig = plt.figure(figsize=(18,8))
plt.plot(score[0])
plt.plot(score[1])
plt.plot(score[2])
plt.legend(['precision','recall','fscore'])
plt.title('Classification Model 1: score');

#CV
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

X = tfidf_matrix
y = data['Sector_scale']
scores_cv = cross_val_score(model, X, y, cv=6)

predictions_cv = cross_val_predict(model, X, y, cv=6)
hist_data = [y,predictions_cv]
plt.figure(figsize=(15, 5))
plt.hist(hist_data,alpha=0.6)
plt.legend(['observed class','predicted class'])
plt.xlabel('Class')
plt.title('Distribution of the predicted vs the observed classes');

fig = plt.figure(figsize=(15,5))
plt.plot(scores_cv)
plt.xlabel('k-fold')
plt.ylabel('Score')
plt.title('Classification Model 1: Cross validation score');

#Balanced data splits
def balanced_data_split(dataset, class_name):
    classes = list(set(dataset[class_name]))
    class_size = {}
    for clas in classes:
        class_size[clas] = len(dataset[dataset[class_name] == clas])
    minimum_class = sorted(class_size.items(), key=lambda x: x[1])[0][0]
    minimum_size = sorted(class_size.items(), key=lambda x: x[1])[0][1]

    balanced_dataset = dataset[dataset[class_name] == minimum_class]

    for clas in classes:
        if clas != minimum_class:
            imbalanced_clas = dataset[dataset[class_name] == clas]
            balanced_clas = imbalanced_clas.take(np.random.permutation(len(imbalanced_clas))[:minimum_size])
            balanced = balanced_dataset.append(balanced_clas, ignore_index=True)
            balanced_dataset = balanced
    return balanced_dataset

equal_classes = balanced_data_split(data,'Sector_scale')
equal_classes_distr = equal_classes.groupby(['TRBC Economic Sector Name','Sector_scale']).count()
equal_classes_distr.reset_index(inplace=True)
equal_classes_distr = pd.DataFrame(equal_classes_distr,columns=['TRBC Economic Sector Name','Sector_scale','Identifier'])
equal_classes_distr.head(10)

Tfidf_vector_bd = Tfidf_Matrix(list(equal_classes['Business Description']),stop_words)
tfidf_matrix_bd = Tfidf_vector_bd[0]
tfidf_vectoriz_bd = Tfidf_vector_bd[1]

X_bd = tfidf_matrix_bd
y_bd = equal_classes['Sector_scale']

X_train_bd, X_test_bd, y_train_bd, y_test_bd = train_test_split(X_bd, y_bd, test_size=0.33, random_state=45, shuffle=True)
NB_fit = model.fit(X_train_bd,y_train_bd)
prediction = NB_fit.predict(X_test_bd)
score_bd = precision_recall_fscore_support(y_test_bd,prediction)

scores_cv = cross_val_score(model, X_bd, y_bd, cv=6)

fig=plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.plot(score[0])
ax1.plot(score[1])
ax1.plot(score[2])
ax1.legend(['precision','recall','fscore'])
ax1.set_title('Classification Model 1 score: Imbalanced dataset',size=10)
ax1.set_xlabel('Class',fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)

ax2 = fig.add_subplot(222)
ax2.plot(score_bd[0])
ax2.plot(score_bd[1])
ax2.plot(score_bd[2])
ax2.legend(['precision','recall','fscore'])
ax2.set_title('Classification Model 1 score: Balanced dataset',size=10)
ax2.set_xlabel('Class',fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)

plt.show()

Industry = data[data['Sector_scale']==4]
Visualize_tfidf_sector(Industry,2)

#Model2

from sklearn.feature_extraction.text import CountVectorizer


def Cooccurencies(dataset, column_name, freq_threshold, stopwords_list):
    docs = list(dataset[column_name])
    count_model = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords_list)
    X = count_model.fit_transform(docs)
    Xc = (X.T * X)  # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0)  # fill same word cooccurence to 0

    cooccur = np.argwhere(Xc > freq_threshold)

    cooccurencies = []
    for i in range(cooccur.shape[0]):
        index1 = cooccur[i][0]
        index2 = cooccur[i][1]
        # print(Xc[index1,index2])
        word1 = count_model.get_feature_names()[index1]
        word2 = count_model.get_feature_names()[index2]
        if word1.isdigit() == False and word2.isdigit() == False:
            cooccurencies.append((word1, word2))

    keywords1 = [i[0] for i in cooccurencies]
    keywords2 = [i[1] for i in cooccurencies]
    keywords = list(set(keywords1 + keywords2))
    return keywords


def Cooccurencies_matrix(dataset, column_name, stopwords_list, keywords):
    cooc = CountVectorizer(tokenizer=tokenize, stop_words=stopwords_list, vocabulary=keywords)
    cooccur_matrix = cooc.fit_transform(list(dataset[column_name]))  # .toarray()

    return cooccur_matrix

keywords = Cooccurencies(equal_classes,'Business Description',500,stop_words)
cooccur_matrix = Cooccurencies_matrix(data,'Business Description',stop_words,keywords)

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add)], 'csr')

#Naive Bayes Classifier
X = tfidf_matrix
X_newfeat = add_feature(X,cooccur_matrix)
y = data['Sector_scale']
X_train, X_test, y_train, y_test = train_test_split(X_newfeat, y, test_size=0.33, random_state=45, shuffle=True)

model=MultinomialNB(alpha=0.1)
NB_fit = model.fit(X_train,y_train)
prediction_md2 = NB_fit.predict(X_test)
score_md2 = precision_recall_fscore_support(y_test,prediction_md2)

fig=plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.plot(score[0])
ax1.plot(score[1])
ax1.plot(score[2])
ax1.legend(['precision','recall','fscore'])
ax1.set_title('Classification Model 1 score: Imbalanced dataset',size=10)
ax1.set_xlabel('Class',fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)

ax2 = fig.add_subplot(222)
ax2.plot(score_md2[0])
ax2.plot(score_md2[1])
ax2.plot(score_md2[2])
ax2.legend(['precision','recall','fscore'])
ax2.set_title('Classification Model 2 score: Imbalanced dataset',size=10)
ax2.set_xlabel('Class',fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)

plt.show()

#Model3

keywords_name = Cooccurencies(equal_classes,'Company Name',3,stopwords_list=text.ENGLISH_STOP_WORDS)

nameterms_matrix = Cooccurencies_matrix(data,'Business Description',stop_words,keywords)

#Naive Bayes Classifier
X_newfeat2 = add_feature(X_newfeat,nameterms_matrix)
y = data['Sector_scale']
X_train, X_test, y_train, y_test = train_test_split(X_newfeat2, y, test_size=0.33, random_state=45, shuffle=True)

model=MultinomialNB(alpha=0.1)
NB_fit = model.fit(X_train,y_train)
prediction_md3 = NB_fit.predict(X_test)
score_md3 = precision_recall_fscore_support(y_test,prediction_md3)

fig=plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.plot(score_md2[0])
ax1.plot(score_md2[1])
ax1.plot(score_md2[2])
ax1.legend(['precision','recall','fscore'])
ax1.set_title('Classification Model 2 score: Imbalanced dataset',size=10)
ax1.set_xlabel('Class',fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)

ax2 = fig.add_subplot(222)
ax2.plot(score_md3[0])
ax2.plot(score_md3[1])
ax2.plot(score_md3[2])
ax2.legend(['precision','recall','fscore'])
ax2.set_title('Classification Model 3 score: Imbalanced dataset',size=10)
ax2.set_xlabel('Class',fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)

plt.show()

y_predict = [(i,j) for i,j in zip(y_test,prediction_md3)]
from collections import Counter
y_predict_vol = Counter(y_predict)
predict_vol_df = pd.DataFrame(list(y_predict_vol.items()),columns=['true-predicted','strength'])
predict_vol_df[['true', 'predicted']] = predict_vol_df['true-predicted'].apply(pd.Series)

fig = plt.figure(figsize=(15,5))
plt.scatter(predict_vol_df['true'], predict_vol_df['predicted'], s=predict_vol_df['strength'], alpha=0.5)
plt.ylabel('Predicted class')
plt.xlabel('True class')
plt.title('Model3 performance');

