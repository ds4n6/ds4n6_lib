# DS4N6
#
# Description: Library of functions to apply Data Science to forensics artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4evtx_parser

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import time
import pickle
import inspect
from tqdm import tqdm

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import Evtx.Evtx as evtx
import Evtx.Views as e_views

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui
import ds4n6_lib.utils  as d4utl
import ds4n6_lib.unx    as d4unx
import ds4n6_lib.evtx   as d4evtx

###############################################################################
# FUNCTIONS
###############################################################################

# .evtx READING FUNCTIONS #####################################################

# Read evtx File(s) / Directory ===============================================

def evtx_xml(evtxf):
    """ Converts evtx file to xml string
    Input:
        evtxf: evtx filename.
    Return:
        return the filename as xml string.
    """  

    thistr = ''
    with evtx.Evtx(evtxf) as log:
        thistr = e_views.XML_HEADER
        thistr = thistr + '<Events>'
        for record in tqdm(log.records()):
            try:
                thistr = thistr + record.xml()
            except:
                # key error 138 in Microsoft-Windows-Ntfs%4Operational.evtx file.
                #print("XML Error: %s" % e)
                pass
        thistr = thistr + '</Events>'

    return thistr

def run_tool(path,verbose=True, **kwargs):
    """ Read evtx files or directories
    Input:
        evtxf: evtx filename/directory.
    Return:
        return the filename as xml string.
    """     

    # Parse Arguments
    tool                     = kwargs.get('tool',                    'evtx_parser')
    hostname                 = kwargs.get('hostname',                '')
    do_harmonize             = kwargs.get('harmonize',               True)
    build_all_df             = kwargs.get('build_all_df'            , False)
    plugin                   = 'windows_evtx_record'
    
    # Identifies if input is file or directory and calls the corresponding function

    # Input possibilities:
    # - Single evtx file
    # - Folder with multiple evtx files for one host
    # - Folder with subfolders for multiple hosts,
    #   each of them containing one (or more) evtx files

    pathtype = ""
    if os.path.isdir(path):
        pathtype = "d"
    elif os.path.isfile(path):
        pathtype = "f"
    
    dfs = {}

    if pathtype == "f":
        dfs = read_evtx_file(path,verbose, tool=tool, hostname=hostname, plugin=plugin, orchestrator=None)
    elif pathtype == "d":
        dfs = read_evtx_directory(path,verbose, tool=tool, hostname=hostname, plugin=plugin, orchestrator=None)
    else:
        raise Exception('The File or Path not exist.')
    
    # Harmonize - - - - - - - - - - - - - - - - - - - - - - - - - -
    if do_harmonize :
        print("- Harmonizing pandas dataframes: ")
        
        orchestrator = dfs['D4_Orchestrator_'].iloc[0]        
        tool         = dfs['D4_Tool_'].iloc[0]        
        plugin       = dfs['D4_Plugin_'].iloc[0]        
        hostname     = dfs['D4_Hostname_'].iloc[0]        
        
        if pathtype == "f":
            dfs = harmonize(dfs, tool=tool, plugin="windows_evtx_record", hostname=hostname, orchestrator=orchestrator)
        elif pathtype == "d":
            keys = dfs.keys()
            for key in keys:
                print('  + %-45s ... ' % (key), end='')
                if  len(dfs[key]) > 0:
                    dfs[key] = harmonize(dfs[key], tool=tool, plugin=key, hostname=hostname, orchestrator=orchestrator)
                    print(' [OK]')
                else:
                    print(' [EMPTY]')

                if build_all_df :
                    dfs['all'] = pd.concat([dfs['all'], dfs[key]], ignore_index=True)

        print("- Done Harmonizing.\n")

    
    return dfs


def read_evtx_file(evtxf, verbose=False, **kwargs):
    """ Read evtx file.
    Input:
        evtxf: evtx filename        
        verbose: True by default.
    Return:
        return xml string.
    """   
    
    orchestrator = kwargs.get('orchestrator', None)
    tool         = kwargs.get('tool',         None)
    plugin       = kwargs.get('plugin',       None)
    hostname     = kwargs.get('hostname',     None)
      
    # Read & Print file size
    fsize = os.path.getsize(evtxf)
    if verbose :
        print("- Size: "+str(fsize))
    
    print("- Processing:")
    evtxfbase = re.sub('^.*\\\\', '', evtxf)
    print("  + "+evtxfbase)
        
    # Read file (evtx or xml)
    _filename, file_extension = os.path.splitext(evtxf)
    if file_extension == ".evtx":
        xmlstr = evtx_xml(evtxf)
    elif file_extension == ".xml" or file_extension == ".XML":
        with open(evtxf, 'r') as file:
            xmlstr = file.read()
    else:
        print("ERROR: Unsupported file extension / format")
        return

    if xmlstr == "":
        print("ERROR: No XML was read")
        return 

    dfs = {}
        
    try:        
        dfs = d4utl.xml_to_df(xmlstr,sep=" > ").d4evtx.column_types_set() 
    except:
        dfs = pd.DataFrame()          
            
    datatype     = "evtx-raw"        
    # Insert Tool / Data related HAM Columns
    if not 'D4_Hostname_' in dfs.columns:
        dfs.insert(0, 'D4_Hostname_', hostname)
    if not 'D4_Plugin_' in dfs.columns:
        dfs.insert(0, 'D4_Plugin_', plugin)
    if not 'D4_Tool_' in dfs.columns:
        dfs.insert(0, 'D4_Tool_', tool)
    if not 'D4_Orchestrator_' in dfs.columns:         
        dfs.insert(0, 'D4_Orchestrator_', orchestrator)
    if not 'D4_DataType_' in dfs.columns:
        dfs.insert(0, 'D4_DataType_', datatype)
    
    if not 'evtxFileName_' in dfs.columns:
        dfs.insert(0, 'evtxFileName_', evtxfbase)
    
    return dfs

def read_evtx_directory(path,verbose=True, **kwargs):
    """ Read directory of evtx files.
    Input:
        path: evtx filename/directory.
        verbose: True by default.
    Return:
        return the directory filenames as one xml string.
    """      

    dfs={}
    
    print("- Processing:")
    
    # See if we have 2 levels
    files = glob.glob(path+"/*/*.evtx") 
    nl2files=len(files)
    if nl2files != 0:
        # We seem to have 2 levels (dir/hostdirs/*.evtx)
        for file in files:
            # We'll derive the hostname from the dirname
            dirname = os.path.dirname(file)
            dirnamebase = os.path.basename(dirname)
            hostname=dirnamebase
            print("- Reading "+file)
            #dfs=read_evtx_file(file,hostname)
            dfs[hostname]=read_evtx_file(file,hostname, **kwargs)
            evtxfbase=re.sub('^.*\\\\','',file)
            dfs[hostname]['evtxFileName_']=evtxfbase
    else:
        # We seem to have only one level (hostdir/*.evtx)
        files = glob.glob(path+"/*.evtx") 
        nl1files=len(files)
        if nl1files != 0:
            for file in files:
                dirname = os.path.dirname(file)
                dirnamebase = os.path.basename(dirname)
                print("- Reading "+file)
                evtxfbase = re.sub('^.*\\\\','',file)
                dfs[evtxfbase] = read_evtx_file(file, **kwargs)
                dfs['evtxFileName_'] = evtxfbase
        else:
            print("ERROR: Could find any evtx file")

    return dfs


# HARMONIZATION FUNCTIONS #####################################################

def harmonize(df, **kwargs):

    # Specific Harmonization Pre-Processing ----------------------------------- 

    # Name index as pevtnum (plaso evt number) and reset it (as a std col)
#    df.index = df.index.set_names("pevtnum")
#    df = df.reset_index()

    # Resort columns
#    cols = df.columns.tolist()
#    cols.insert(len(cols), cols.pop(cols.index('message')))
#    cols.insert(len(cols), cols.pop(cols.index('pevtnum')))
#    cols.insert(0, cols.pop(cols.index('timestamp_desc')))
#    df = df.reindex(columns=cols)

#    df['timestamp_desc'] = df['timestamp_desc'].str.replace(' Time$','').str.replace('Modification','Modif.')

    # Generic Harmonization ---------------------------------------------------
    #df = d4com.harmonize_common(df, **kwargs)

    # Specific Harmonization Post-Processing ----------------------------------

    # Convert timestamp to datetime
#    nowus = int(time.time()*1000000)
#    df.insert(5, 'Timestamp_', None)
#    df['Timestamp_'] = df['timestamp']
#    df['Timestamp_'] = df['Timestamp_'].mask(df['Timestamp_'].lt(0),0)
#    df['Timestamp_'] = df['Timestamp_'].mask(df['Timestamp_'].gt(nowus),0)
#    df['Timestamp_'] = pd.to_datetime(df['Timestamp_'], unit="us")

    # Resort columns
    #cols = df.columns.tolist()

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

    def syntax():
        print('Syntax: analysis(obj, "analysis_type")\n')
        d4list("str-help")
        return

    def d4list(objtype):

        # Analysis Modules Available for this objective
        anlav = False
        print("Available XXXXX analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.match("^dict-pandas_dataframe-XXXXX", objtype):
            anlav = True
            print("- XXXXX_files:  No.events XXXXX file (Input: XXXdfs)")

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

    # ANALYSIS FUNCTIONS ======================================================

    # XXXdfs ------------------------------------------------------------------
    # if   re.match("^dict-pandas_dataframe-XXXXX", objtype):
    #     if anltype == "XXXXX_files":
    #         return analysis_XXXXX_files(*args, **kwargs)

    print("INFO: [d4evtx] No analysis functions available for this data type ("+objtype+")")

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4evtx_parser")
@pd.api.extensions.register_dataframe_accessor("d4_evtx_parser")
class Ds4n6EvtxParserAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

