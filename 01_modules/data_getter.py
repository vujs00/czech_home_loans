# -*- coding: utf-8 -*-
"""
    The following functions are are used to import the czech mortgages dataset.
    They are also used to perform some basic data wrangling operations.
"""

#------------------------------------------------------------#
# STEP 1: libraries                                          #
#------------------------------------------------------------#

#export_path = interim_library_path + r'\01-1_-_df.parquet'

import pandas as pd

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def rename_columns(df0, variable_mapping0):
    
    df = df0.copy(deep = True)
    variable_mapping = variable_mapping0.copy(deep = True)
    df = df.rename(columns = variable_mapping.set_index('source_variable_name')['variable_name'])
    return df


def bool_to_flg(df, variable_mapping0):
    ''' The function recodes boolean columns to flag. It also updates the
        mapping table to note that certain columns are now flag-types.
        
        Inputs:
        - df: master dataset,
        - variable_mapping: manually created mapping file,
        - interim_library_path: export path for updated version of variable_
                                mapping dataframe. 
        '''
    
    # Another way of selecting is to take their name from the variable_mapping.
    # This approach is however more robust to typos in that dataframe.
    helper = df.select_dtypes(include = 'boolean')
    df[list(helper)] = df[list(helper)].astype(int)
    
    # Recode the variable names to type 'flg'.
    helper = pd.DataFrame(list(helper))
    helper.rename(columns = {0 : 'variable_name'}, inplace = True)
    helper['new_variable_name'] = helper['variable_name'].str[:-4] + 'flg' # Automate the recognition of the ending.
    df = df.rename(columns = helper.set_index('variable_name')['new_variable_name'])

    # Adapt the variable_mapping dataset with the new names.
    # This is done because the initial variable mapping has bool for these columns.
    variable_mapping = variable_mapping0.copy(deep = True)
    
    variable_mapping.loc[variable_mapping['analytical_type_cd'] == 'boolean',
                         'variable_name'] = variable_mapping['variable_name'].str[:-4] + 'flg'
    
    variable_mapping.loc[variable_mapping['analytical_type_cd'] == 'boolean',
                         'analytical_type_cd'] = 'flg'

    return df, variable_mapping


def wrangle_marital_status(df):
    ''' The function re-codes the marital_status variable.
        
        Inputs:
            - df: master dataset,
            - variable_mapping: mapping table. 
            '''

    df.loc[df['marital_status_cd'].isin(['Svobodn²(ß)']),
           'marital_status_cd'] = 'single'

    df.loc[df['marital_status_cd'].isin(['Äenat²',
                                         'Vdanß',
                                         'Reg.partner']),
           'marital_status_cd'] = 'partnered'

    df.loc[df['marital_status_cd'].isin(['Rozveden²(ß)']),
           'marital_status_cd'] = 'divorced'

    df.loc[df['marital_status_cd'].isin(['Vdovec',
                                         'Vdova']),
           'marital_status_cd'] = 'widowed'

    df.loc[df['marital_status_cd'].isin(['Nezadßno',
                                         'Zem°el(a)']),
           'marital_status_cd'] = 'other'
    
    df.loc[df['marital_status_cd'].isnull(),
           'marital_status_cd'] = 'other'
            
    return df


def exclude_features(df, variable_mapping):
    ''' The function excludes variables according to a manually created
        mapping table.
        
        Inputs:
            - df: master dataset,
            - variable_mapping: mapping table. 
            '''
            
    # The use flag was constructed manually by inspecting the variables.
    helper = variable_mapping.loc[variable_mapping['use_flg'] == 0][['variable_name']]
    df = df.drop(helper['variable_name'].values.tolist(), axis = 1)
    return df


def ingest_data(df0, variable_mapping0):
    ''' The function runs the functions defined above.
        
        Inputs:
            - df: master dataset,
            - variable_mapping: mapping table,
            - interim_library_path: see bool_to_flg.
            '''
    
    df = rename_columns(df0, variable_mapping0)
    df, variable_mapping = bool_to_flg(df, variable_mapping0)
    df = wrangle_marital_status(df)
    df = exclude_features(df, variable_mapping)
    
    return df, variable_mapping



