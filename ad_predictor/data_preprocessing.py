import pandas as pd
import numpy as np

class preprocessing():
    
    def __init__(self, data):
        self.data = data

    def split_numerical_categorical(self):
        self.data.replace([np.inf, -np.inf], np.nan,inplace=True)#replace inf with na
        self.data.fillna(0,inplace=True)#replace na with 0
        num_descr = pd.DataFrame(self.data.describe())
        numerical_var=list(num_descr.columns[1:])
        categorical_var=[i for i in list(self.data.columns) if i not in list(num_descr.columns)]
        return numerical_var, categorical_var

    def numerical_feature_selection(self, numerical_var):
        corr_numeric = self.data[list(numerical_var)].corr()
        low_corr = corr_numeric[abs(corr_numeric)<0.0005]
        low_corr.fillna(0,inplace=True)
        low_corr_var = [i for i in list(low_corr.sum().index) if low_corr.sum()[i]!=0]
        return low_corr_var



