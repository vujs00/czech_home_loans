# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_classif

df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_train_binned.parquet')

df_smot_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_train_binned.parquet')

list(df_smot_train)

df_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_test_binned.parquet')
df_smot_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_test_binned.parquet')

df_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_train.parquet'
df_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_test.parquet'
df_smot_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_smot_train.parquet'
df_smot_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_smot_test.parquet'

#------------------------------------------------------------#
# STEP 2: define correlation-based selection                 #
#------------------------------------------------------------#

def multicollinearity(df_train, df_test):

    # Define selector.
    selector = SelectNonCollinear(correlation_threshold = 0.5, 
                                  scoring = f_classif)
    
    # Prepare X and y.
    y_train = df_train.loc[:, 'default_event_flg']
    X_train = df_train.loc[:, df_train.columns != 'default_event_flg']
    
    X_train_array = np.nan_to_num(X_train)
    y_train_array = y_train.to_numpy()
    
    selector.fit(X_train_array, y_train_array)
    mask = selector.get_support()
    
    X_train = pd.DataFrame(X_train_array[:, mask], columns = np.array(list(X_train))[mask])
    
    # Repack into data frame.
    # Create a helper in order to retain the y variable in the operations below.
    helper = [[1]]
    helper = pd.DataFrame(helper, columns = ['default_event_flg'])
    helper = pd.concat([helper, X_train])
    helper = list(helper)
    
    df_train_2 = df_train[helper]
    df_test_2 = df_test[helper]
            
    return df_train_2, df_test_2

#------------------------------------------------------------#
# STEP 3: apply function                                     #
#------------------------------------------------------------#

# Run.
dfs_reduced = multicollinearity(df_train = df_train, df_test = df_test)

# Extract the results.
df_train = dfs_reduced[0]
df_test = dfs_reduced[1]

# Run.
dfs_smot_reduced = multicollinearity(df_train = df_smot_train, 
                                     df_test = df_smot_test)

# Extract the results.
df_smot_train = dfs_smot_reduced[0]
df_smot_test = dfs_smot_reduced[1]

del dfs_reduced, dfs_smot_reduced

# Export.
df_train.to_parquet(df_train_export_path)
df_test.to_parquet(df_test_export_path)

df_smot_train.to_parquet(df_smot_train_export_path)
df_smot_test.to_parquet(df_smot_test_export_path)




