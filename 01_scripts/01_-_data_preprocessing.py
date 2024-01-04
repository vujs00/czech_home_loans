# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

df = df.rename(columns = variable_mapping.set_index('source_variable_name')['variable_name'])

export_path = interim_library_path + r'\01_-_df.parquet'

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

# Update the types of the variable mapping.
variable_mapping.to_parquet(interim_library_path + r'\01_-_variable_mapping.parquet')

#------------------------------------------------------------#
# STEP 3: exclude uneccessary variables                      #
#------------------------------------------------------------#

helper = variable_mapping.loc[variable_mapping['use_flg'] == 0][['variable_name']]

df = df.drop(helper['variable_name'].values.tolist(), axis = 1)

del helper

#------------------------------------------------------------#
# LAST STEP: export                                          #
#------------------------------------------------------------#

df.to_parquet(export_path)

del df, export_path, variable_mapping
