# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression

df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_train_binned.parquet')
df_smot_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_train_binned.parquet')
df_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_test_binned.parquet')
df_smot_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_test_binned.parquet')

ginis_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_univariate_gini_coefficients.parquet'
ginis_smot_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_univariate_smot_gini_coefficients.parquet'

df_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_df_train_shortlisted.parquet'
df_smot_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_df_smot_train_shortlisted.parquet'
df_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_df_test_shortlisted.parquet'
df_smot_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\05_-_df_smot_test_shortlisted.parquet'

#------------------------------------------------------------#
# STEP 2: selection based on IV                              #
#------------------------------------------------------------#

df_train = sc.var_filter(df_train, y = 'default_event_flg', iv_limit = 0.02,
                         missing_limit = 0.7, identical_limit = 0.95,
                         positive = 'bad|1')

df_smot_train = sc.var_filter(df_smot_train, y = 'default_event_flg', 
                              iv_limit = 0.02, missing_limit = 0.7, 
                              identical_limit = 0.95, positive = 'bad|1')

#------------------------------------------------------------#
# STEP 3.1: univariate regressions on original dataset       #
#------------------------------------------------------------#

y_train = df_train.loc[:, 'default_event_flg']
X_train = df_train.loc[:, df_train.columns != 'default_event_flg']

# Figure what the parameters of the function mean.
logit = LogisticRegression()

# Run univariate models and store them into dictionaries.
models = {}
predictions = {}
performances = {}
ginis = {}

for column in X_train:
    x_train = X_train[[column]]
    models[str(column)] = logit.fit(x_train, y_train)
    predictions[str(column)] = logit.predict_proba(x_train)[:,1]
    performances[str(column)] = sc.perf_eva(y_train, predictions[str(column)])
    ginis[str(column)] = performances[str(column)]['Gini']

del models
del predictions
del performances

# Transform all X matrix ginis into a data frame.    
ginis = pd.DataFrame.from_dict(ginis, orient = 'index').reset_index()
ginis.rename(columns = {'index' : 'variable_name',
                        0 : 'gini_coeff'}, inplace = True)

ginis.to_parquet(ginis_export_path)

#------------------------------------------------------------#
# STEP 3.2: selection and export on original dataset         #
#------------------------------------------------------------#

''' Keep only variables that have gini > 0.1. '''

# Create a helper in order to retain the y variable in the operations below.
helper = [['default_event_flg', 1]]
helper = pd.DataFrame(helper, columns = ['variable_name', 'gini_coeff'])

# Reduce datasets.
ginis = pd.concat([helper, ginis])
del helper
ginis = ginis.loc[ginis['gini_coeff'] >= 0.1]

helper = list(ginis['variable_name'])
df_train = df_train[helper]
df_test = df_test[helper]

del helper
del ginis

df_train.to_parquet(df_train_export_path)
df_test.to_parquet(df_test_export_path)

#------------------------------------------------------------#
# STEP 4.1: univariate regressions on oversampled dataset    #
#------------------------------------------------------------#

y_smot_train = df_smot_train.loc[:, 'default_event_flg']
X_smot_train = df_smot_train.loc[:, df_smot_train.columns != 'default_event_flg']

# Figure what the parameters of the function mean.
logit = LogisticRegression()

# Run univariate models and store them into dictionaries.
models = {}
predictions = {}
performances = {}
ginis = {}

for column in X_smot_train:
    x_smot_train = X_smot_train[[column]]
    models[str(column)] = logit.fit(x_smot_train, y_smot_train)
    predictions[str(column)] = logit.predict_proba(x_smot_train)[:,1]
    performances[str(column)] = sc.perf_eva(y_smot_train, predictions[str(column)])
    ginis[str(column)] = performances[str(column)]['Gini']

del models
del predictions
del performances

# Transform all X matrix ginis into a data frame.    
ginis = pd.DataFrame.from_dict(ginis, orient = 'index').reset_index()
ginis.rename(columns = {'index' : 'variable_name',
                        0 : 'gini_coeff'}, inplace = True)

ginis.to_parquet(ginis_smot_export_path)

#------------------------------------------------------------#
# STEP 4.2: selection and export on oversampled dataset      #
#------------------------------------------------------------#

''' Keep only variables that have gini > 0.1. '''

# Create a helper in order to retain the y variable in the operations below.
helper = [['default_event_flg', 1]]
helper = pd.DataFrame(helper, columns = ['variable_name', 'gini_coeff'])

# Reduce datasets.
ginis = pd.concat([helper, ginis])
del helper
ginis = ginis.loc[ginis['gini_coeff'] >= 0.1]

helper = list(ginis['variable_name'])
df_smot_train = df_smot_train[helper]
df_smot_test = df_smot_test[helper]

del helper
del ginis

df_smot_test.to_parquet(df_smot_train_export_path)
df_smot_test.to_parquet(df_smot_test_export_path)
