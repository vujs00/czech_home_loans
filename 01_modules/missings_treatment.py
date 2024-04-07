# -*- coding: utf-8 -*-
"""

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def impute_deliquency_entries(df, variable_mapping):
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1)]
    variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]

    eligible_features = variable_mapping.loc[variable_mapping['variable_name'].str.contains('deliquency')]
    eligible_features = list(eligible_features['variable_name'])
    
    uneligible_features = variable_mapping.loc[~variable_mapping['variable_name'].str.contains('deliquency')]
    uneligible_features = list(uneligible_features['variable_name'])
    
    helper_1 = df[eligible_features]
    helper_2 = df[uneligible_features]
    
    helper_1 = helper_1.fillna(0)

    df = helper_1.join(helper_2, how = 'left')
    
    return df


def impute_counts_and_flags_entries(df, variable_mapping):
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1)]
    variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]

    eligible_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['flg', 'count'])]
    eligible_features = list(eligible_features['variable_name'])
    
    uneligible_features = variable_mapping.loc[~variable_mapping['analytical_type_cd'].isin(['flg', 'count'])]
    uneligible_features = list(uneligible_features['variable_name'])
    
    helper_1 = df[eligible_features]
    helper_2 = df[uneligible_features]
    
    helper_1 = helper_1.fillna(0)

    df = helper_1.join(helper_2, how = 'left')
    
    return df


def remove_unenrichable_features(df, variable_mapping):
    ''' The function removes amount features that have a high number of
        missings. '''
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1)]

    eligible_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['amt'])]
    eligible_features = list(eligible_features['variable_name'])
        
    helper_1 = df[eligible_features]
    helper_1 = helper_1.isnull().sum().reset_index()
    helper_1.rename(columns = {'index' : 'variable_name',
                               0 : 'nan_count'},
                    inplace = True)
    helper_1 = helper_1.loc[helper_1['nan_count'] >= 0.25*df.shape[0]]
    helper_1 = list(helper_1['variable_name'])    
    
    df = df.drop(columns = helper_1, axis = 1)
    
    # Update the variable_mapping list.
    variable_mapping = variable_mapping.loc[variable_mapping['variable_name'].isin(list(df))]
    
    return df, variable_mapping


def treat_missings(df, variable_mapping):
    
    df = impute_deliquency_entries(df, variable_mapping)
    df = impute_counts_and_flags_entries(df, variable_mapping)
    df, variable_mapping = remove_unenrichable_features(df, variable_mapping)
    
    return df, variable_mapping


def remove_nans(df):
    ''' Brute removal of nan values. '''    
    df = df.dropna()
    return df

