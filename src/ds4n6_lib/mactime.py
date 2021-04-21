
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4mctm

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import time
import inspect
import xmltodict
import json
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as et

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui
import ds4n6_lib.utils  as d4utl

###############################################################################
# IDEAS
###############################################################################
# is_deleted()
# is_file()
# is_dir() / is_folder() - level
# ext() # filter by Extension
# nofn  # exclude $FILE_NAME entries

###############################################################################
# FUNCTIONS
###############################################################################

# FILE READING FUNCTIONS ######################################################

# FILE READING FUNCTIONS ######################################################

def read_data(evdl, **kwargs):
    if d4.debug >= 3:
        print("DEBUG: [mctm] read_data")

    return d4com.read_data_common(evdl, **kwargs)

# HARMONIZATION FUNCTIONS #####################################################

def harmonize(df, **kwargs):
    """ Convert DF in HAM format

        Args: 
            df (pandas.DataFrame): DF to harmonize
            kwargs(dict): harmonize options
        Returns: 
            pandas.DataFrame in HAM Format
    """
    objtype = d4com.data_identify(df)

    if objtype == "pandas_dataframe-mactime-raw":
        # Specific Harmonization Pre-Processing -----------------------------------
        df = df.rename(columns={"Type": "MACB"})

        df['Type_']        = df['Mode'].str.extract('^(.)')
        df['PrevType_']    = df['Mode'].str.extract('^..(.)')
        df['Permissions_'] = df['Mode'].str.extract('^...(.........)')

        # Deleted / Reallocated
        df['Deleted_']     = df['File Name'].str.contains(r'\ \(deleted\)$|\ \(deleted-reallocated\)$')
        df['Reallocated_'] = df['File Name'].str.contains(r'\ \(deleted-reallocated\)$')

        # [FT] Tag -> Tag_ | DriveLetter_ | VSS_ | EVOName_ | EvidenceName_ | Partition_ | FilePath_
        # FT
        if re.search(r'^[A-Z]\[vss[0-9][0-9]\]{.*}:', df['File Name'].iloc[0]):
            fncolsdf  = df['File Name'].str.split(":", 1, expand=True).rename(columns={0: "Tag_", 1: "FilePath_"})
            fncolsdf['FilePath-Hash_'] = fncolsdf['FilePath_'].str.lower().apply(hash)
            fncolsdf['FSType_']   = '-'
            df['Hostname_']    = '-'
            df['SHA256_Hash_'] = '-'

            fncols2df = fncolsdf['Tag_'].str.extract(r'([A-Z])\[vss(.*)\]{(.*)}', expand=True).rename(columns={0: "DriveLetter_", 1: "VSS_", 2: "EVOName_"})
            fncols2df['VSS_'] = fncols2df['VSS_'].astype(int)

            fncols3df = fncols2df['EVOName_'].str.extract('(.*)-ft-p(.*)', expand=True).rename(columns={0: "EvidenceName_", 1: "Partition_"})
            fncols3df['Partition_'] = fncols3df['Partition_'].astype(int)

            df = pd.concat([df, fncols2df, fncols3df, fncolsdf], axis=1)

        else:
            fncolsdf  = df['File Name'].str.split(":", 1, expand=True).rename(columns={0: "Tag_", 1: "FilePath_"})
            df = pd.concat([df, fncolsdf], axis=1)
            df['Hostname_']     = '-'
            df['EVOName_']      = '-'
            df['EvidenceName_'] = '-'
            df['Partition_']    = '-'
            df['FSType_']       = '-'
            df['DriveLetter_']  = '-'
            df['VSS_']          = '-'
            df['TSNTFSAttr_']   = '-'
            df['SHA256_Hash_']  = '-'

        # Deal with "($FILE_NAME)" string
        tsntfsattrmap = {True: 'FILE_NAME', False: 'STD_INFO'}
        df['TSNTFSAttr_']  = df['FilePath_'].str.contains(r'\ \(\$FILE_NAME\)$').map(tsntfsattrmap)
        df['FilePath_']    = df['FilePath_'].str.replace(r'\ \(\$FILE_NAME\)$','')

        df['FilePath_'] = df['FilePath_'].str.replace(r'\ \(deleted\)$|\ \(deleted-reallocated\)$','')
        
        # Generic Harmonization ---------------------------------------------------
        df = d4com.harmonize_common(df, **kwargs)

        # Specific Harmonization Post-Processing ----------------------------------

        return df

# CORE FUNCTIONS (simple, analysis, etc.) #####################################

# simple ======================================================================

def simple_func(df, *args, **kwargs):
    """ Reformat the input df so the data is presented to the analyst in the
        friendliest possible way

    Parameters:
    df  (pd.dataframe):  Input data 
    
    Returns:
    pd.DataFrame: Optionally it will return the filtered dataframe, 
                  only if ret=True is set, constant & hidden columns included
                  If ret_out=True is set, then the output just as it is shown
                  (without constant/hidden columns) will be return
    """

    if d4.debug >= 3:
        print("DEBUG: [mctm] [simple_func()]")

    windows = kwargs.get('windows', True) 

    # Variables ----------------------------------------------------------------
    hiddencols =  ['File_Name', 'FilePath-Hash_', 'SHA256_Hash_']

    if windows :
        nonwincols = ['UID', 'GID', 'Mode', 'Permissions_']
        hiddencols = hiddencols + nonwincols

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)


# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4mctm")
class Ds4n6MctmAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_mactime")
class Ds4n6MactimeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

# ANALYSIS ####################################################################

# analysis() function =========================================================
def analysis(*args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    return analysis_func(*args, **kwargs)

def analysis_func(*args, **kwargs):
    """ Umbrella function that redirects to different types of analysis 
        available on the input data

    Parameters:
    obj:          Input data (typically DF or dict of DFs)
    
    Returns:
    pd.DataFrame: Refer to each specific analysis function
    """

    def syntax():
        print('Syntax: analysis(obj, "analysis_type")\n')
        d4list("str-help")
        return

    def d4list(objtype):

        # Analysis Modules Available for this objective
        # anlav = False
        print("Available fstl analysis types:")
        print("- No analysis functions defined yet.")
        return

        # TEMPLATE
        #if objtype == "str-help" or objtype == "str-list" or  re.search("^pandas_dataframe-fstl-mactime-standard", objtype):
        #    anlav = True
        #    print("- XXXXXXXXXX:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX (Input: fstldf)")

        # if anlav == False:
        #     print('- No analysis modules available for this object ('+objtype+').')

    nargs = len(args)

    if nargs == 0:
        syntax()
        return

    obj = args[0]

    objtype = d4com.data_identify(obj)

    if isinstance(obj, str):
        if obj == "list":
            d4list(objtype)
            return
        if obj == "help":
            syntax()
            return

    if nargs == 1:
        syntax()
        return

    anltype = args[1]

    if not isinstance(anltype, str):
        syntax()
        return

    if anltype == "help":
        syntax()
        return
    elif anltype == "list":
        d4list(objtype)
        return

    # TEMPLATE
    # If object is a dict of dfs
    #elif re.search("^pandas_dataframe-evtx_file_df", objtype):
    #    if anltype == "XXXXXXXXXXX":
    #        return XXXXXXXXXXXXXXXXXXXXX(*args, **kwargs)
    #else:
    #    print("ERROR: [fstl] Unsupported input data.")
    #    return

