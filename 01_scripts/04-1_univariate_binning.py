# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:45:27 2023

@author: JF13832
"""


import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression

#df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet')
df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet')
df_smot_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\03_-_df_smot_train.parquet')
df_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_test.parquet')
df_smot_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_test.parquet')




''' Tryouts '''

#retail_behavioral_score_categorical
#days_in_deliquency_6m_avg_count
#retail_behavioral_score_categorical

#import scorecardpy as sc

variable = 'retail_behavioral_score_categorical'

list(df_smot_train)

# Make WOE.
x = df_smot_train[variable].values
y = df_smot_train['default_event_flg'].values

optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
optb.fit(x, y)
optb.status
optb.splits
binning_table = optb.binning_table
binning_table.build()
binning_table.plot(metric="woe", show_bin_labels =True)
binning_table.plot(metric="event_rate")
x_transform_woe = optb.transform(x, metric="woe")
result = pd.Series(x_transform_woe).value_counts()

# Apply to test.
X_test = df_test[[variable]]
x_test_transform_woe = optb.transform(X_test, metric="woe")

