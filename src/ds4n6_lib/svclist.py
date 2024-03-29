# DS4N6
#
# Description: Library of functions to apply Data Science to forensics artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4svclst

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
import pickle

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui
import ds4n6_lib.utils  as d4utl
import ds4n6_lib.unx    as d4unx

###############################################################################
# FUNCTIONS
###############################################################################

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

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Artifact-specific argument parsing =======================================
    hiddencolsuser = kwargs.get('hiddencols',  [])

    # Variables ================================================================
    hiddencolsdef =  []

    # Merge artifact hiddencols with user-specified hiddencols + update kwargs
    hiddencols = hiddencolsuser + hiddencolsdef 
    kwargs['hiddencols'] = hiddencols

    dfout = df

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Pre-Processing ==========================================================

    # Call to simple_common ===================================================
    dfout = d4com.simple_common(df, *args, **kwargs, maxdfbprintlines=maxdfbprintlines)

    # Post-Processing =========================================================

    # Return ==================================================================
    return dfout

# analysis ====================================================================
def analysis(obj, *args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    return analysis_func(obj, *args, **kwargs)

def analysis_func(obj, *args, **kwargs):
    """ Umbrella function that redirects to different types of analysis 
        available on the input data

    Parameters:
    obj:          Input data (typically DF or dict of DFs)
    
    Returns:
    pd.DataFrame: Refer to each specific analysis function
    """

    # SUB-FUNCTIONS ###########################################################
    def syntax():
        print('Syntax: analysis(obj, "analysis_type")\n')
        d4list("str-help")
        return

    def d4list(objtype):

        # Analysis Modules Available for this objective
        anlav = False
        print("Available XXXXX analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-XXXXX", objtype):
            anlav = True
            print("- XXXXX_files:  No.events XXXXX file (Input: XXXdfs)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')

    # FUNCTION BODY ###########################################################
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    thisdatatype = "XXXXXXXX-THIS_DATA_TYPE"

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
        if thisdatatype is not None:
            if re.search("^dict-pandas_dataframe-"+thisdatatype, objtype) or re.search("^pandas_dataframe-"+thisdatatype, objtype):
                d4list(objtype)
            else:
                syntax()
        else:
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

    # ANALYSIS FUNCTIONS ======================================================

    # XXXdfs ------------------------------------------------------------------
    if   re.search("^dict-pandas_dataframe-XXXXX", objtype):
        if anltype == "XXXXX_files":
            return analysis_XXXXX_files(*args, **kwargs)

    print("INFO: [d4XXX] No analysis functions available for this data type ("+objtype+")")

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4svclst")
class Ds4n6SvcListAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


@pd.api.extensions.register_dataframe_accessor("d4_svclist")
class Ds4n6SvcListAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)
