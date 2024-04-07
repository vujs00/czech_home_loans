# -*- coding: utf-8 -*-
"""

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import numpy as np

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def apply_iqr(df, variable_mapping, target):
    
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    outliers = (df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1) &
                                            (variable_mapping['variable_name'] != target)]
    variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]
    eligible_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['ratio'])]
    eligible_features = list(eligible_features['variable_name'])
    
    outliers = outliers[eligible_features]
    
    df = df[~outliers.any(axis = 1)]
    
    return df, outliers

