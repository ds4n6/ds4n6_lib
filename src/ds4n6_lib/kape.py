# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4kp

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
import math
from tqdm import tqdm
import xml.etree.ElementTree as et

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
import ipywidgets as widgets
from ipywidgets import Layout

from traitlets import traitlets
# ML
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

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # (1) kansa will probably be invoked with 'tool=kansa', but kansa is in 
    #     reality an orchestrator, so we will only populate the Tool_ column 
    #     to kansa only if we have not been able to determine what is the 
    #     underlying tool that kansa is using for execution in the endpoint

    # Specific Harmonization Pre-Processing -----------------------------------
    if not 'D4_Orchestrator_' in df.columns:
        df.insert(0, 'D4_Orchestrator_', "kape")
    else:
        # If the Orchestrator_ col exists, we are in a recursive call
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
        df = d4com.harmonize(df)

    # Specific Harmonization Post-Processing ----------------------------------
    d4pdf = df['D4_Plugin_'].iloc[0]

    if isinstance(d4pdf, float):
        if math.isnan(d4pdf):
            print('ERROR: The D4_Plugin_ column is NaN. Cannot fully Harmonize data.')
            print('       Did you specify the "pluginisdfname=True"')
            print('       when reading the input files?')
            return

    if re.search("^Registry-Services-", d4pdf):
        df['D4_Plugin_']   = "Registry-Services"
        df['D4_DataType_'] = "svclist"

        # Rename columns
        df = df.rename(columns={'Name':               'Name_', 
                                'DisplayName':        'DisplayName_',
                                'ImagePath':          'FilePath_',
                                'StartMode':          'StartMode_',
                                'ServiceType':        'ServiceType_',
                                'ServiceDLL':         'ServiceDLL_',
                                'Group':              'Group_',
                                'Description':        'Description_',
                                'RequiredPrivileges': 'RequiredPrivileges_'
                               })
    elif re.search("^EventLogs-EvtxECmd", d4pdf):
        df['D4_Plugin_']   = "EventLogs-EvtxECmd"
        df['D4_DataType_'] = "evtx"
        # Rename columns
        df = df.rename(columns={'Name':               'Name_', 
                                'DisplayName':        'DisplayName_',
                                'ImagePath':          'FilePath_',
                                'StartMode':          'StartMode_',
                                'ServiceType':        'ServiceType_',
                                'ServiceDLL':         'ServiceDLL_',
                                'Group':              'Group_',
                                'Description':        'Description_',
                                'RequiredPrivileges': 'RequiredPrivileges_',
                                'EventId':            'EventID_',
                                'SourceFile':         'evtxFileName_',
                                'TimeCreated':        'Timestamp',
                                'UserId':             'TargetUserSid',
                                'UserName':           'TargetUserName',
                                'Computer':           'WorkstationName'
                               })
        df['FileName_'] = df['ExecutableInfo'].str.split('\\').str[-1]
        df['IpAddress'] = "0.0.0.0"
        df['LogonType'] = np.nan

    elif re.search(r"^FileSystem-MFTECmd_\$MFT", d4pdf):

        df['D4_Plugin_']   = "FileSystem-MFTECmd_$MFT"
        df['D4_DataType_'] = "flist"

        # Rename columns
        df = df.rename(columns={'EntryNumber':              'Meta_', 
                                'SequenceNumber':           'NTFS-SeqNumber_',
                                'InUse':                    'Deleted_',
                                'ParentEntryNumber':        'ParentMeta_',
                                'ParentSequenceNumber':     'ParentSeqNumber_',
                                'ParentPath':               'ParentPath_',
                                'FileName':                 'FileName_',
                                'Extension_':               'FileExtension_',
                                'FileSize':                 'Size_',
                                'ReferenceCount':           'NTFS-ReferenceCount_',
                                'ReparseTarget':            'NTFS-ReparseTarget_',
                                'IsDirectory':              'IsDirectory_',
                                'Created0x10':              'BTime_',
                                'LastModified0x10':         'MTime_',
                                'LastRecordChange0x10':     'CTime_',
                                'LastAccess0x10':           'ATime_',
                                'HasAds':                   'NTFS-HasAds_',
                                'IsAds':                    'NTFS-IsAds_',
                                'SI<FN':                    'NTFS-SI<FN_',
                                'uSecZeros':                'NTFS-uSecZeros_',
                                'Copied':                   'NTFS-Copied_',
                                'SiFlags':                  'NTFS-SiFlags_',
                                'NameType':                 'NTFS-NameType_',
                                'Created0x30':              'NTFS-FN-BTime_',
                                'LastModified0x30':         'NTFS-FN-MTime_',
                                'LastRecordChange0x30':     'NTFS-FN-CTime_',
                                'LastAccess0x30':           'NTFS-FN-ATime_',
                                'UpdateSequenceNumber':     'NTFS-UpdateSequenceNumber_',
                                'LogfileSequenceNumber':    'NTFS-LogfileSequenceNumber_',
                                'SecurityId':               'NTFS-SecurityId_',
                                'ObjectIdFileDroid':        'NTFS-ObjectIdFileDroid_',
                                'LoggedUtilStream':         'NTFS-LoggedUtilStream_',
                                'ZoneIdContents':           'NTFS-ZoneIdContents_',     
                               })                           
                                                          
        # Reverse InUse -> Deleted                         
        df['Deleted_']    = ~df['Deleted_']
        df['FileType_']   = df['IsDirectory_']
        df['FileType_']   = np.where(df['FileType_'], 'd', 'f')
        df['ParentPath_'] = df['ParentPath_'] .str.replace(r"^\.","").str.replace(r"^\\\.$",".")
        df['FilePath_']   = df['ParentPath_'] + '\\' + df['FileName_']

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
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)


def get_source_options():
    return ['wrap_cols', 'beautify_cols']


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

    # SUB-FUNCTIONS ###########################################################
    def syntax(objtype=None):
        print('Syntax: analysis(obj, "analysis_type")\n')
        if objtype is None:
            d4list("str-help")
        else:
            d4list(objtype)
        return

    def d4list(objtype):
        # Analysis Modules Available for this objective
        anlav = False

        print("Available kape analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-kape", objtype):
            anlav = True
            print("- kape_files:  No.events kape file (Input: kpdfs)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')


    # FUNCTION BODY ###########################################################
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
        syntax(objtype)
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

    # kpdfs -------------------------------------------------------------------
    if   re.search("^dict-pandas_dataframe-kape", objtype):
        if anltype == "kape_files":
            return analysis_kape_files(*args, **kwargs)

    print("INFO: [d4kp] No analysis functions available for this data type ("+objtype+")")

def analysis_kape_files(*args, **kwargs):
    """ Analysis that gives kape files
        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """
    dfs = args[0]

    objtype = d4com.data_identify(dfs)

    if not objtype.startswith("dict-pandas_dataframe-kape"):
        print("ERROR: Invalid object for function: "+objtype)
        print("       Input object should be:      dict-pandas_dataframe-kape")
        return

    outdf = pd.DataFrame([],columns=['NEntries', 'KapeFile'])
    row = pd.Series()

    for key in dfs.keys():
        row['KapeFile']  = key
        row['NEntries'] = len(dfs[key])

        outdf = outdf.append(row,ignore_index=True).sort_values(by=['KapeFile']).reset_index(drop=True)

    #outdf.insert(0, 'Artifact/Tool', '')
    #outdf.insert(0, 'Category', '')
    outdf['Category'] = outdf['KapeFile'].str.replace('-.*','')
    outdf['Artifact/Tool'] = outdf['KapeFile'].str.replace('^[^-]*-','').str.replace('-[^-]*$','').str.replace('_Output$','').str.replace(r'.*\.dat$','')
    outdf['File']     = outdf['KapeFile'].str.replace('^[^-]*-','').str.replace(r'^([^-]*\.dat)','DUMMY-\\1').str.replace('^[^-]*$','').str.replace('^[^-]*-','')

    return outdf

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4kp")
class Ds4n6KpAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_kape")
class Ds4n6KapeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


