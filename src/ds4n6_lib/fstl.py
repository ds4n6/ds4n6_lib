# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4fstl

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

def read_data(evdl, **kwargs):
    return d4com.read_data_common(evdl, **kwargs)
    
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
    if d4.debug >= 4:
        print("DEBUG: [fstl] [simple_func()]")

    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4fstl")
class Ds4n6FSTLAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

    def nofn(self):
        return self._obj[~self._obj['FileName'].str.contains(r"\ \(\$FILE_NAME\)")]
        
    def is_deleted(self):
        if 'FileName' in self._obj.columns:
            return self._obj[self._obj['FileName'].str.contains(r"\ \(deleted\)$")]
        elif 'Deleted_' in self._obj.columns:
            return self._obj.query('Deleted_ == True')
        
    def is_file(self):
        return self._obj.query('Type_ == "r" | PrevType_ == "r"')
        
    def is_dir(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    # Same as is_dir()
    def is_directory(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    # Same as is_dir()
    def is_folder(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    def ext(self,ext):
        return self._obj[self._obj['FileName'].str.contains(r"\."+ext+"$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(\$FILE_NAME\)$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(deleted\)$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(\$FILE_NAME\)\ \(deleted\)$")]

    def just_basename(self):
        return self._obj['FileName'].str.replace('.*/','')
        
    def ts_m(self,exclusive=False):
        if exclusive == False:
            return self._obj.query(r'MACB.str.contains("^m...$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^m\.\.\.$")',engine="python")
        
    def ts_a(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^.a..$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.a\.\.$")',engine="python")
        
    def ts_c(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^..c.$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.\.c\.$")',engine="python")
        
    def ts_b(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^...b$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.\.\.b$")',engine="python")
   
@pd.api.extensions.register_dataframe_accessor("d4_fstl")
class Ds4n6_FSTLAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

    def nofn(self):
        return self._obj[~self._obj['FileName'].str.contains(r"\ \(\$FILE_NAME\)")]
        
    def is_deleted(self):
        if 'FileName' in self._obj.columns:
            return self._obj[self._obj['FileName'].str.contains(r"\ \(deleted\)$")]
        elif 'Deleted_' in self._obj.columns:
            return self._obj.query('Deleted_ == True')
        
    def is_file(self):
        return self._obj.query('Type_ == "r" | PrevType_ == "r"')
        
    def is_dir(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    # Same as is_dir()
    def is_directory(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    # Same as is_dir()
    def is_folder(self,level=0):
        return self._obj.query('Type_ == "d"')
        
    def ext(self,ext):
        return self._obj[self._obj['FileName'].str.contains(r"\."+ext+"$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(\$FILE_NAME\)$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(deleted\)$") | self._obj['FileName'].str.contains(r"\."+ext+r"\ \(\$FILE_NAME\)\ \(deleted\)$")]

    def just_basename(self):
        return self._obj['FileName'].str.replace('.*/','')
        
    def ts_m(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^m...$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^m\.\.\.$")',engine="python")
        
    def ts_a(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^.a..$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.a\.\.$")',engine="python")
        
    def ts_c(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^..c.$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.\.c\.$")',engine="python")
        
    def ts_b(self,exclusive=False):
        if exclusive == False:
            return self._obj.query('MACB.str.contains("^...b$")',engine="python")
        else:
            return self._obj.query(r'MACB.str.contains("^\.\.\.b$")',engine="python")
   
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


