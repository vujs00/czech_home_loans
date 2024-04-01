# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

df_train = pd.read_parquet(interim_library_path + r'\01-2_-_df_train.parquet')

df_smot_train_export_path = interim_library_path + r'\01-4_-_df_smot_train.parquet'

# Create a list of all categorical features. This is necessary to keep the
# SMOTENC algorithm informed about these features.
variable_mapping = pd.read_parquet(interim_library_path + r'\01-1_-_variable_mapping.parquet')
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

smotenc = SMOTENC(categorical_features, random_state = set_seed, 
                  k_neighbors = 5)

X_smot, y_smot = smotenc.fit_resample(X_train, y_train)
df_train_smot = y_smot.join(X_smot, how = 'left')

df_train_smot.to_parquet(df_smot_train_export_path)

del df_train, df_smot_train_export_path, variable_mapping, categorical_features,
del X_train, y_train, imputer, smotenc, X_smot, y_smot, df_train_smot
