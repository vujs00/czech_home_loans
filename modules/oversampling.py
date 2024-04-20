# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def list_categorical(variable_mapping, target):
    ''' Create a list of all categorical features. This is necessary to keep the
        SMOTENC algorithm informed about these features.
        '''
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1) &
                                            (variable_mapping['variable_name'] != target)]
    variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]
    categorical_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['flg', 'str'])]
    categorical_features = categorical_features[['variable_name']]
    categorical_features = categorical_features['variable_name'].values.tolist()
    
    return categorical_features


def apply_smotenc(df_train, target, set_seed, variable_mapping):
    
    categorical_features = list_categorical(variable_mapping, target)
    
    X_train = df_train.drop([target], axis = 1)
    y_train = df_train[[target]]
    
    imputer = SimpleImputer(strategy = 'most_frequent')
    imputer = imputer.fit(X_train)
    X_train[:] = imputer.transform(X_train)
    
    helper = df_train.loc[df_train[target] == 1].shape[0]
    smotenc = SMOTENC(categorical_features, random_state = set_seed, 
                      k_neighbors = 5, sampling_strategy = {1:2*helper})
    
    X_smot, y_smot = smotenc.fit_resample(X_train, y_train)
    df_train_smot = y_smot.join(X_smot, how = 'left')
    
    return df_train_smot


