# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4ksa

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
import ds4n6_lib.d4      as d4
import ds4n6_lib.common  as d4com
import ds4n6_lib.gui     as d4gui
import ds4n6_lib.utils   as d4utl

###############################################################################
# FUNCTIONS
###############################################################################

# FILE READING FUNCTIONS ######################################################

def read_data(evdl, **kwargs):
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

    # (1) kansa will probably be invoked with 'tool=kansa', but kansa is in 
    #     reality an orchestrator, so we will only populate the D4_Tool_ column 
    #     to kansa only if we have not been able to determine what is the 
    #     underlying tool that kansa is using for execution in the endpoint

    # Specific Harmonization Pre-Processing -----------------------------------
    if not 'D4_Orchestrator_' in df.columns:
        df.insert(0, 'D4_Orchestrator_', "kansa")
    else:
        # If the D4_Orchestrator_ col exists, we are in a recursive call
        return df

    objtype = d4com.data_identify(df)

    # Generic Harmonization ---------------------------------------------------

    # Since kansa is an orchestrator, let's try to identify the specific 
    # data type and apply the corresponding harmonization function.
    # If we can, we will execute the generic one.
    if "unknown" in objtype:
        df = d4com.harmonize_common(df, **kwargs)
    else:
        # Let's try to harmonize this specific df
        # WARNING: Since we no longer identify datatype by DF cols, this will
        #          not work
        df = d4com.harmonize(df)

    # Specific Harmonization Post-Processing ----------------------------------
    df['D4_Hostname_'] = df['PSComputerName']

    if df['D4_Plugin_'].iloc[0] == "Tasklistv":
        df['D4_DataType_'] = "pslist"
        df['D4_DataType_'] = df['D4_DataType_'].astype('category')

        # Rename columns
        df = df.rename(columns={'ImageName': 'Name_', 'PID': 'PID_', 
                                'SessionName': 'SessionName_', 
                                'SessionNum': 'Session_', 'MemUsage': 'MemUsage_', 
                                'Status': 'Status_', 'UserName': 'UserName_', 
                                'CPUTime': 'CPUTime_', 'WindowTitle': 'WindowTitle_'
                               })

    elif df['D4_Plugin_'].iloc[0] == "SvcAll":
        df['D4_DataType_'] = "svclist"
        df['D4_DataType_'] = df['D4_DataType_'].astype('category')

        # Rename columns
        df = df.rename(columns={'Name': 'Name_', 'DisplayName': 'DisplayName_', 
                                'PathName': 'FilePath_', 'StartName': 'UserName_',
                                'StartMode': 'StartMode_', 'State': 'State_',
                                'TotalSessions': 'TotalSessions_', 
                                'Description': 'Description_'
                               })

    return df

# ANALYSIS FUNCTIONS ======================================================

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
    if d4.debug >= 4:
        print("DEBUG: [ksa] [simple_func()]")

    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

# analysis ====================================================================
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
        anlav = False
        print("Available kansa analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-kansa", objtype):
            anlav = True
            print("- kansa_files:  No.events kansa file (Input: ksadfs)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')

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

    # ksadfs ------------------------------------------------------------------
    if   re.search("^dict-pandas_dataframe-kansa", objtype):
        if anltype == "kansa_files":
            return analysis_kansa_files(*args, **kwargs)

    print("INFO: [d4ksa] No analysis functions available for this data type ("+objtype+")")

def analysis_kansa_files(*args, **kwargs):
    """ Analysis that gives kansa files

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """
    dfs = args[0]

    objtype = d4com.data_identify(dfs)

    if objtype != "dict-pandas_dataframe-kansa":
        print("ERROR: Invalid object for function: "+objtype)
        print("       Input object should be:      dict-pandas_dataframe-kansa")
        return

    outdf = pd.DataFrame([],columns=['File','NEntries'])
    row = pd.Series()

    for key in dfs.keys():
        row['File']  = key
        row['NEntries'] = len(dfs[key])

        outdf = outdf.append(row,ignore_index=True)

    return outdf

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4ksa")
class Ds4n6KsaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_kansa")
class Ds4n6KansaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


