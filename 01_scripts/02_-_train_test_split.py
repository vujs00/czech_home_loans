# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_df.parquet')

train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet'
test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_test.parquet'

#------------------------------------------------------------#
# STEP 2: feature selection                                  #
#------------------------------------------------------------#

X = df.loc[:, df.columns != 'default_event_flg']
y = df.loc[:, 'default_event_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 13816)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

df_train = y_train.join(X_train, how = 'left')
df_test = y_test.join(X_test, how = 'left')

df_train.to_parquet(train_export_path)
df_test.to_parquet(test_export_path)
