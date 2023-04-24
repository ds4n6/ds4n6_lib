# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4pslst

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
import ds4n6_lib.unx    as d4unx
from ds4n6_lib.knowledge import critical_processes, boot_start_processes, process_parents

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
        print("Available pslist analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^pandas_dataframe-pslist-ham", objtype):
            anlav = True
            print("- process_stats:            Show process statistics               (Input: pslistdf)")
            print("- unfrequent_processes:     Identify unfrequent processes         (Input: pslistdf)")
            print("- boot_time_anomalies:      Identify boot time proccess anomalies (Input: pslistdf)")
            print("- parent_process_anomalies: Identify parent process anomalies     (Input: pslistdf)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

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

    # pslistdf ----------------------------------------------------------------
    if   re.search("^pandas_dataframe-pslist-ham", objtype):
        if   anltype == "process_stats":
            return analysis_process_stats(*args, **kwargs)
        elif anltype == "unfrequent_processes":
            return analysis_unfrequent_processes(*args, **kwargs)
        elif anltype == "boot_time_anomalies":
            return analysis_boot_time_anomalies(*args, **kwargs)
        elif anltype == "parent_process_anomalies":
            return analysis_parent_process_anomalies(*args, **kwargs)

    print("INFO: [d4pslst] No analysis functions available for this data type ("+objtype+")")

# ANALYSIS FUNCTIONS ==========================================================

def analysis_process_stats(*args, **kwargs):
    """ Show Process Statistics

        Args: 
        obj:          Input data (HAM process DF)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Argument parsing
    df = args[0]

    if 'Exit_TStamp_' in df.columns:
        print("Running:")
        display(df.query('Exit_TStamp_.isna()', engine="python")['Name_'].value_counts())
        print("")
        print("Dead:")
        display(df.query('Exit_TStamp_.notna()', engine="python")['Name_'].value_counts())
        print("")
    else:
        display(df['Name_'].value_counts())

def analysis_unfrequent_processes(*args, **kwargs):
    """ Analysis that find unfrequent processes

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Argument parsing
    pslistdf = args[0]

    n = kwargs.get('n', 3)

    print("Threshold: "+str(n))
    print("")

    pscntdf  = pd.DataFrame(pslistdf['Name_'].value_counts()).reset_index().rename(columns={'Name_': 'Count', 'index': 'Name_'})
    pscntdf['Count'] = pscntdf['Count'].astype(int)
    pscntndf = pscntdf.query('Count <= @n', engine="python")
    
    print("No. Processes with less than " + str(n) +" occurrences: " + str(len(pscntndf)))
    return pscntndf


def analysis_boot_time_anomalies(*args, **kwargs):
    """ Find anomalies at boot time

    Parameters:
    pslistdf (pd.DataFrame): Dataframe with pslist info
    secs (int):              Interval allowed for processes to start after boot 
    
    Returns:
    pd.DataFrame: Processes that don't follow the standard start time pattern

    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Argument parsing
    df = args[0]

    secs = kwargs.get('secs', 30)

    # Verify field requirements
    if not 'Start_TStamp_' in df.columns:
        print("ERROR: Cannot run analysis. Start_TStamp_ column not present.")
        return

    print("Min. Start Timestamp Processes:")
    display(df[df['Start_TStamp_'] == df['Start_TStamp_'].min()])

    if 'Session_' in df.columns:
        bootps = df[df['Name_'].isin(boot_start_processes)  & (df['Session_'] <= 1) & df['Exit_TStamp_'].isnull() ]
    else:
        bootps = df[df['Name_'].isin(boot_start_processes)  & df['Exit_TStamp_'].isnull() ]

    return bootps[bootps['Start_TStamp_'] >= bootps['Start_TStamp_'].min() + pd.Timedelta(seconds=secs)]

def analysis_parent_process_anomalies(*args, **kwargs):
    """ Find anomalies in parent processes

    Parameters:
    pslistdf (pd.DataFrame): Dataframe with pslist info
    critical_only (bool): Only critical process
    
    Returns:
    None
    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Argument parsing
    df = args[0]

    critical_only = kwargs.get('critical_only', True)

    # Verify field requirements
    if not 'PPID_' in df.columns:
        print("ERROR: Cannot run analysis. PPID_ column not present.")
        return

    if 'Exit_TStamp_' in df.columns:
        df_alive = df[df['Exit_TStamp_'].isna()]
    else:
        df_alive = df

    hnpid  = df_alive[['D4_Hostname_', 'Name_', 'PID_']]
    hnppid = df_alive[['D4_Hostname_', 'Name_', 'PPID_']]
    family_ext = pd.merge(hnppid, hnpid, left_on=['D4_Hostname_', 'PPID_'], right_on=['D4_Hostname_', 'PID_'], how='left').dropna()
    family = family_ext.drop(columns=['D4_Hostname_', 'PPID_', 'PID_']).rename(columns={'Name__x': 'Child', 'Name__y': 'Parent'}).reset_index().drop(columns=['index'])

    if critical_only :
        thisfamily = family.query('Child == @critical_processes')
    else:
        thisfamily = family

    family_unknown = pd.merge(thisfamily, process_parents, indicator=True, how='outer').query( '_merge=="left_only"').drop( '_merge', axis=1)

    display(family_unknown.groupby(["Child", "Parent"]).size().sort_values(ascending=False))
    display(family_unknown)

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4pslst")
class Ds4n6PslstAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_pslist")
class Ds4n6PslistAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

