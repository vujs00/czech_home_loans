# -*- coding: utf-8 -*-
"""
"""

#------------------------------------------------------------#
# STEP 1: libraries                                          #
#------------------------------------------------------------#

import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_classif

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


class Preprocessor():

    
    def __init__(self, set_seed, target, df, undersample, 
                 decorrelate, oot_year, encoding_metric):
        self.set_seed: int = set_seed
        self.target: str = target
        self.df: pd.DataFrame = df
        self.undersample: bool = undersample
        self.decorrelate: bool = decorrelate
        self.oot_year: int = oot_year # format yyyymm.
        self.encoding_metric: str = encoding_metric
    
    
    def retain_oot(self):
        
        # This could be automated, but there is no need for the
        #  point of this thesis.
        self.df_oot = self.df.loc[self.df['obs_yyyymm'] == self.oot_year]
        self.df_oot = self.df_oot.drop(['obs_yyyymm'], axis = 1)
        self.df = self.df.loc[self.df['obs_yyyymm'] != self.oot_year]
        self.df = self.df.drop(['obs_yyyymm'], axis = 1)
        
        return self.df, self.df_oot
        
    
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


    def undersample_train(self):
        
        if self.undersample:
            
            X_train = self.df_train.loc[:, self.df_train.columns != self.target]
            y_train = self.df_train.loc[:, self.target]
        
            rus = RandomUnderSampler(random_state = self.set_seed,
                                     sampling_strategy = 0.25)
        
            X_train, y_train = rus.fit_resample(X_train, y_train)
            y_train = pd.DataFrame(y_train)
            self.df_train = y_train.join(X_train, how = 'left')
            
        else:
            pass
        return self.df_train

    
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
        self.binning_process = BinningProcess(variable_names = var_names,
                                              categorical_variables = list(self.cat_vars),
                                              selection_criteria = selection_criteria,
                                              min_n_bins = 2, max_n_bins = 10)
        self.binning_process.fit(X, y)
        #binning_process.information()
        self.performance_summary = self.binning_process.summary()

        # Apply to train.
        X_binned = self.binning_process.transform(X, metric='woe')
        y = pd.DataFrame(y).reset_index()
        self.df_train = y.join(X_binned, how = 'left')
        self.df_train = self.df_train.drop(['index'], axis = 1)
        
        # Apply to test.
        X_test = self.df_test[var_names]
        y_test = self.df_test[self.target]
        y_test = pd.DataFrame(y_test).reset_index()
        X_test_binned = self.binning_process.transform(X_test, 
                                                       metric='woe')
        self.df_test = y_test.join(X_test_binned, how = 'left')
        self.df_test = self.df_test.drop(['index'], axis = 1)
        
        # Apply to oot.
        X_oot = self.df_oot[var_names]
        y_oot = self.df_oot[self.target]
        y_oot = pd.DataFrame(y_oot).reset_index()
        X_oot_binned = self.binning_process.transform(X_oot,
                                                      metric='woe')
        self.df_oot = y_oot.join(X_oot_binned, how = 'left')
        self.df_oot = self.df_oot.drop(['index'], axis = 1)
        
        return (self.df_train, self.df_test, self.df_oot)

    
    def remove_multicollinearity(self):
        
        if self.decorrelate:
                    
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
            self.df_oot = self.df_oot[helper]
                        
        else:
            pass
            
        return (self.df_train, self.df_test, self.df_oot, 
                self.performance_summary)
    
    
    def run(self):
        
        self.retain_oot()
        self.split_train_test()
        self.undersample_train()
        self.list_categorical()
        self.bin_and_transform()
        self.remove_multicollinearity()
        
        return (self.df_train, self.df_test, self.df_oot,
                self.performance_summary)



