# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic
"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import numpy as np
import pandas as pd
import datetime

df = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\czech_mortgages_dataset_v2.csv')

variable_mapping = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\mapping_tables.xlsx',
                                 sheet_name = 'variables')

df = df.rename(columns = variable_mapping.set_index('source_variable_name')['variable_name'])

export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_df.parquet'

#------------------------------------------------------------#
# STEP 2: recode boolean to flag                             #
#------------------------------------------------------------#

# Another way of selecting is to take their name from the variable_mapping.
# This approach is however more robust to typos in that dataframe.
helper = df.select_dtypes(include = 'boolean')
df[list(helper)] = df[list(helper)].astype(int)

# Recode the variable names to type 'flg'.
helper = pd.DataFrame(list(helper))
helper.rename(columns = {0 : 'variable_name'}, inplace = True)
helper['new_variable_name'] = helper['variable_name'].str[:-4] + 'flg' # Automate the recognition of the ending.
df = df.rename(columns = helper.set_index('variable_name')['new_variable_name'])

del helper

# Adapt the variable_mapping dataset with the new names.
variable_mapping.loc[variable_mapping['analytical_type_cd'] == 'boolean',
                     'variable_name'] = variable_mapping['variable_name'].str[:-4] + 'flg'

variable_mapping.loc[variable_mapping['analytical_type_cd'] == 'boolean',
                     'analytical_type_cd'] = 'flg'

variable_mapping.to_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_variable_mapping.parquet')

#------------------------------------------------------------#
# STEP 3: missings treatment map                             #
#------------------------------------------------------------#

''' Creating an overview of missing values and declaring their treatment
    based on the variable characteristic. '''

helper = df.isnull().sum().reset_index()
helper.rename(columns = {'index' : 'variable_name',
                         0 : 'nan_count'}, inplace = True)
helper['nan_pct'] = helper['nan_count']/200000
helper = helper.sort_values(by = 'nan_pct')

# Add descriptions of variables.
helper = pd.merge(helper,
                  variable_mapping[['variable_name', 'analytical_type_cd', 'use_flg']],
                  how = 'left',
                  left_on = 'variable_name',
                  right_on = 'variable_name')

''' First treatment round '''

# Declare treatment approaches that are mapped.
helper.loc[helper['analytical_type_cd'].isin(['flg', 'count']),
           'nan_treatment_str'] = 'put 0'

#helper.loc[helper['analytical_type_cd'].isin(['amt', 'count', 'flg', 'int']),
#           'nan_treatment_str'] = 'put 0'

treatment_map = helper[['variable_name', 'nan_treatment_str']]
treatment_map['proactive_treatment_flg'] = 0
treatment_map.loc[treatment_map['nan_treatment_str'].str.contains('no treatment') == False,
                  'proactive_treatment_flg'] = 1

del helper

''' The treatment approaches chosen above are implemented in this step.
    The code is able to input 0s as this is currently the only treatment.
    If the complexity of treatment increases, i.e. averages, medians or such
    are used, the code would need to be adjusted accordingly. '''

treatment_map = treatment_map.loc[treatment_map['proactive_treatment_flg'] == 1]

helper = list(treatment_map['variable_name'])
df[helper] = df[helper].fillna(0)
del helper

df.loc[df['employment_type_cd'].isna() == True,
       'employment_type_cd'] = 'no information'

#helper.to_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_summary_table_2.xlsx',
#                index = False)

#------------------------------------------------------------#
# STEP 5: exclude uneccessary variables and nan values       #
#------------------------------------------------------------#

helper = variable_mapping.loc[variable_mapping['use_flg'] == 0][['variable_name']]

df = df.drop(helper['variable_name'].values.tolist(), axis = 1)

#df = df.dropna()

del helper

#------------------------------------------------------------#
# LAST STEP: export                                          #
#------------------------------------------------------------#

df.to_parquet(export_path)


