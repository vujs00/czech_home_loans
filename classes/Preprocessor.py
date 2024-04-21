# -*- coding: utf-8 -*-
"""
"""

#------------------------------------------------------------#
# STEP 1: libraries                                          #
#------------------------------------------------------------#

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_classif

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


class Preprocessor():

    
    def __init__(self, set_seed, target, df, encoding_metric):
        self.set_seed: int = set_seed
        self.target: str = target
        self.df: pd.DataFrame = df
        self.encoding_metric: str = encoding_metric
    
    
    def split_train_test(self):
        
        X = self.df.loc[:, self.df.columns != self.target]
        y = self.df.loc[:, self.target]
        
        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, test_size=0.3, random_state=self.set_seed)
        
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        
        self.df_train = y_train.join(X_train, how = 'left')
        self.df_test = y_test.join(X_test, how = 'left')

        return self.df_train, self.df_test

    
    def list_categorical(self):
        
        self.cat_vars = pd.DataFrame(list(self.df))
        self.cat_vars.rename(columns = {0 : 'variable_name'}, inplace = True)
        self.cat_vars =\
            self.cat_vars.loc[((self.cat_vars['variable_name'].str.contains('cd'))
                               | (self.cat_vars['variable_name'].str.contains('flg')))
                              & (self.cat_vars['variable_name'] != self.target)]
        
        return self.cat_vars
    
    
    def bin_and_transform(self):

        # X and y.
        var_names = list(self.df_train.loc[:, self.df_train.columns !=\
                                           self.target].columns)
        X = self.df_train[var_names]
        y = self.df_train[self.target]

        # Define and fit binning.
        selection_criteria = {
            'iv' : {'min' : 0.05},
            'gini' : {'min' : 0.1}
            }
        binning_process = BinningProcess(variable_names = var_names,
                                         categorical_variables = list(self.cat_vars),
                                         selection_criteria = selection_criteria,
                                         min_n_bins = 2, max_n_bins = 10)
        binning_process.fit(X, y)
        #binning_process.information()
        self.performance_summary = binning_process.summary()

        # Apply to train.
        X_binned = binning_process.transform(X, metric=self.encoding_metric)
        y = pd.DataFrame(y).reset_index()
        self.df_train = y.join(X_binned, how = 'left')
        self.df_train = self.df_train.drop(['index'], axis = 1)
        
        # Apply to test.
        X_test = self.df_test[var_names]
        y_test = self.df_test[self.target]
        y_test = pd.DataFrame(y_test).reset_index()
        X_test_binned = binning_process.transform(X_test, 
                                                  metric=self.encoding_metric)
        self.df_test = y_test.join(X_test_binned, how = 'left')
        self.df_test = self.df_test.drop(['index'], axis = 1)
        
        return (self.df_train, self.df_test)


    def apply_one_hot(self):
        
        if self.encoding_metric == 'bins':
            self.df_train = pd.get_dummies(self.df_train)
            self.df_test = pd.get_dummies(self.df_test)
        else:
            pass
    
        return (self.df_train, self.df_test, self.performance_summary)

    
    def remove_multicollinearity(self):
        
        # Define selector.
        selector = SelectNonCollinear(correlation_threshold = 0.5, 
                                      scoring = f_classif)
        
        # Prepare X and y.
        y_train = self.df_train.loc[:, self.target]
        X_train = self.df_train.loc[:, self.df_train.columns != self.target]
        
        X_train_array = np.nan_to_num(X_train)
        y_train_array = y_train.to_numpy()
        
        # Fit.
        selector.fit(X_train_array, y_train_array)
        mask = selector.get_support()
        
        X_train = pd.DataFrame(X_train_array[:, mask], columns =\
                               np.array(list(X_train))[mask])
        
        # Create a helper to retain the y variable in the operations below.
        helper = [[1]]
        helper = pd.DataFrame(helper, columns = [self.target])
        helper = pd.concat([helper, X_train])
        helper = list(helper)
        
        self.df_train = self.df_train[helper]
        self.df_test = self.df_test[helper]
                
        return (self.df_train, self.df_test)
    
    
    def run(self):
        
        self.split_train_test()
        self.list_categorical()
        self.bin_and_transform()
        self.apply_one_hot()
        self.remove_multicollinearity()
        
        return (self.df_train, self.df_test, self.performance_summary)



