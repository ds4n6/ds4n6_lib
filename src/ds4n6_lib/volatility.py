# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4vol

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
import json
import pickle

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
from IPython.display import display, Markdown, HTML

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui
import ds4n6_lib.utils  as d4utl

###############################################################################
# FUNCTIONS
###############################################################################

# FILE READING FUNCTIONS ######################################################

def read_data(evdl, **kwargs):
    """ Read data from files or a folder

        Args: 
            evdl (str): path to file/folder source
            kwargs: read options
        Returns: 
            pandas.Dataframe or dictionary of pandas.DataFrame
    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

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
    plugin       = kwargs.get('plugin',       None)
    hostname     = kwargs.get('hostname',     None)
    
    # Specific Harmonization Pre-Processing -----------------------------------
    if hostname is not None:
        df['D4_Hostname_'] = hostname
    if hostname is not None:
        df['D4_Plugin_'] = plugin
    if not df.index.empty and df.index[0] == ">":
        df.reset_index(drop=True, inplace=True)

    # Generic Harmonization ---------------------------------------------------
    df = d4com.harmonize_common(df, **kwargs)

    # Specific Harmonization Post-Processing ----------------------------------

    # pslist  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if plugin == "pslist":
        df['D4_DataType_'] = "pslist"
        df['D4_DataType_'] = df['D4_DataType_'].astype('category')

        # Rename columns
        df = df.rename(columns={'Name': 'Name_', 'PID': 'PID_', 'PPID': 'PPID_', 
                                'Thds': 'Threads_', 'Hnds': 'Handles_',
                                'Sess': 'Session_', 'Wow64': 'Wow64_',
                                'Start': 'Start_TStamp_', 'Exit': 'Exit_TStamp_'
                               })

        # Adjust data types
        df['Session_']      = df['Session_'].str.replace('^--*$','-1')
        df['Session_']      = df['Session_'].astype(int)
        df['Handles_']      = df['Handles_'].str.replace('^--*$','-1')
        df['Handles_']      = df['Handles_'].astype(int)
        df['Start_TStamp_'] = pd.to_datetime(df['Start_TStamp_'])
        df['Exit_TStamp_']  = pd.to_datetime(df['Exit_TStamp_'])

    # psscan  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif plugin == "psscan":
        df['D4_DataType_'] = "pslist"
        df['D4_DataType_'] = df['D4_DataType_'].astype('category')

        # Rename columns
        df = df.rename(columns={'Name': 'Name_', 'PID': 'PID_', 'PPID': 'PPID_', 
                                'Time created': 'Start_TStamp_', 'Time exited': 'Exit_TStamp_'
                               })

        # Adjust data types
        df['Start_TStamp_'] = pd.to_datetime(df['Start_TStamp_'])
        df['Exit_TStamp_']  = pd.to_datetime(df['Exit_TStamp_'])

    # return ------------------------------------------------------------------
    return df

# ANALYSIS FUNCTIONS ##########################################################

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
        print("DEBUG: [vol] [simple_func()]")

    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

# analysis() ==================================================================
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
        print("Available volatility analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-volatility", objtype):
            anlav = True
            print("- volatility_files:  No.events volatility file (Input: voldfs)")

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

    # voldfs ------------------------------------------------------------------
    if   re.search("^dict-pandas_dataframe-volatility", objtype):
        if anltype == "volatility_files":
            return analysis_volatility_files(*args, **kwargs)

    print("INFO: [d4vol] No analysis functions available for this data type ("+objtype+")")

# ANALYSIS FUNCTIONS ==========================================================

def analysis_volatility_files(*args, **kwargs):
    """ Analysis that gives volatility files

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """
    dfs = args[0]

    objtype = d4com.data_identify(dfs)

    if not re.search("^dict-pandas_dataframe-volatility", objtype):
        print("ERROR: Invalid object for function: "+objtype)
        print("       Input object should be:      dict-pandas_dataframe-volatility")
        return

    outdf = pd.DataFrame([],columns=['NEntries', 'VolFile'])
    row = pd.Series()

    for key in dfs.keys():
        row['VolFile']  = key
        row['NEntries'] = len(dfs[key])

        outdf = outdf.append(row,ignore_index=True).sort_values(by=['VolFile']).reset_index(drop=True)

    return outdf

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4vol")
class Ds4n6VolAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_volatility")
class Ds4n6VolatilityAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

