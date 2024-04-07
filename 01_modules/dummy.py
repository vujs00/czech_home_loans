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


def create_dummies(df, variable_mapping):
    ''' The function converts string variables into dummy variables. '''
    
    variable_mapping = variable_mapping.loc[(variable_mapping['use_flg'] == 1)]
    variable_mapping = variable_mapping[['variable_name', 'analytical_type_cd']]
    eligible_features = variable_mapping.loc[variable_mapping['analytical_type_cd'].isin(['str'])]
    eligible_features = list(eligible_features['variable_name'])
    
    uneligible_features = variable_mapping.loc[~variable_mapping['analytical_type_cd'].isin(['str'])]
    uneligible_features = list(uneligible_features['variable_name'])
    
    helper_1 = df[eligible_features]
    helper_2 = df[uneligible_features]

    helper_1 = pd.get_dummies(helper_1)
    
    df = helper_1.join(helper_2, how = 'left')
    
    return df

