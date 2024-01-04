# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
from sklearn.impute import SimpleImputer
#from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC

df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet')

df_smot_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\03_-_df_smot_train.parquet'

variable_mapping = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_variable_mapping.parquet')
variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1) &
                                        (variable_mapping['variable_name'] != 'default_event_flg')]
variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]
categorical_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['flg', 'str'])]
categorical_features = categorical_features[['variable_name']]
categorical_features = categorical_features['variable_name'].values.tolist()

#------------------------------------------------------------#
# STEP 3: SMOTENC                                            #
#------------------------------------------------------------#

X_train = df_train.drop(['default_event_flg'], axis = 1)
y_train = df_train[['default_event_flg']]

imputer = SimpleImputer(strategy = 'most_frequent')
imputer = imputer.fit(X_train)
X_train[:] = imputer.transform(X_train)

smotenc = SMOTENC(categorical_features, random_state = 130816, k_neighbors = 5)

X_smot, y_smot = smotenc.fit_resample(X_train, y_train)
df_train_smot = y_smot.join(X_smot, how = 'left')

df_train_smot.to_parquet(df_smot_train_export_path)
