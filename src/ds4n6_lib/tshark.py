# DS4N6
#
# Description: Library of functions to apply Data Science to forensics artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4tshrk

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
import subprocess
import json

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

# FILE READING FUNCTIONS ######################################################

def read_data(evdl, **kwargs):
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
        
    if bool(re.search(r"\.pcap$", evdl, re.IGNORECASE)):
        return read_tshark_pcap(evdl, **kwargs)
    
    elif bool(re.search(r"\.json$", evdl, re.IGNORECASE)):
        return read_pcap_json(evdl, **kwargs)
    elif bool(re.search(r"\.csv$", evdl, re.IGNORECASE)):
        return read_pcap_csv(evdl, **kwargs)
    
    else:
        print("ERROR: Unable to read input file. Unsupported file extension.")
        return

def read_tshark_pcap(evdl, **kwargs):
    """ Read pcap data from to json file
        Args: 
            pcapf (str): path to file source
            kwargs: read options
        Returns: 
            .json file
    """
    cmd = "tshark -r " + evdl + " -T ek -j "'http tcp ip'" -P -V -x > " + evdl+'.json'
    print(cmd)
    subprocess.Popen(cmd, shell = True, 
                                 stdout=subprocess.PIPE)
    
    evdl = evdl+'.json'
    
    return read_pcap_json(evdl,**kwargs)

def read_pcap_json(evdl, **kwargs):
    """ Read pcap data from from a json file
        Args: 
            evdl (str): path to file source
            kwargs: read options
        Returns: 
            pandas.DataFrame (in the future a dictionary of pandas.DataFrame)
    """
    n
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Parse Arguments
    tool                     = kwargs.get('tool',                    '')
    hostname                 = kwargs.get('hostname',                '')
    do_harmonize             = kwargs.get('harmonize',               True)
    use_pickle               = kwargs.get('use_pickle'              , True)

    pklrawf = evdl+'.raw.pkl'
    pklhtmf = evdl+'.htm.pkl'

    if os.path.exists(pklhtmf) and use_pickle  and do_harmonize :

        # Read from pickle
        print("- Saved Harmonized pickle file found:")
        print("      "+pklhtmf)
        print("- Reading data from HAM pickle file...")
        dfs = pickle.load(open(pklhtmf, "rb"))
        print("- Done.")
        print("")

    else:
        print("- No saved Harmonized pickle file found.")
        print("")
        
        
        

    
    with open(evdl, 'r') as f:
        data = [json.loads(line) for line in f]
    dfs = pd.json_normalize(data)
    
    return dfs

def read_pcap_csv(evdl, **kwargs):
    """ Read pcap data from from a json file
        Args: 
            evdl (str): path to file source
            kwargs: read options
        Returns: 
            pandas.DataFrame (in the future a dictionary of pandas.DataFrame)
    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Parse Arguments
    tool                     = kwargs.get('tool',                    '')
    hostname                 = kwargs.get('hostname',                '')
    do_harmonize             = kwargs.get('harmonize',               True)
    use_pickle               = kwargs.get('use_pickle'              , True)
    
    output = pd.read_csv(evdl)
    output = output.rename(columns={'ip.src': 'Source_IP', 'ip.dst': 'Destination_IP', 'tcp.srcport': 'Source_TCP_Port', 'tcp.dstport': 'Destination_TCP_Port', 'frame.time': 'Frame_Time', '_ws.col.Protocol': 'Protocol', '_ws.col.Info': 'Info'})
    
    return output
    
    
# HARMONIZATION FUNCTIONS #####################################################

def harmonize(df, **kwargs):
    """ Function description

        Args: 

        Returns: 

        Raises:
    """

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    orchestrator = kwargs.get('orchestrator', None)
    tool         = kwargs.get('tool',         None)
    plugin       = kwargs.get('plugin',       None)
    hostname     = kwargs.get('hostname',     None)

    # Specific Harmonization Pre-Processing ===================================

    # Generic Harmonization ===================================================
    df = d4com.harmonize_common(df, **kwargs)

    # Specific Harmonization Post-Processing ==================================

    # return ==================================================================
    # WARNING: For artifact-modules only
    # df['D4_DataType_'] = 'DATA_TYPE_HERE'

    return df

# ANALYSIS FUNCTIONS ##########################################################

# simple ======================================================================
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

    # Variables ================================================================
    hiddencols =  []

    dfout = df

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Pre-Processing ==========================================================

    # Call to simple_common ===================================================
    dfout = d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

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

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

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
    thisdatatype = None

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

@pd.api.extensions.register_dataframe_accessor("d4tshrk")
class Ds4n6TshrkAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


@pd.api.extensions.register_dataframe_accessor("d4_tshark")
class Ds4n6TsharkAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)
