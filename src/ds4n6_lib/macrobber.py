# DS4N6
#
# Description: Library of functions to apply Data Science to forensics artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4mcrb

###############################################################################
# IMPORTS
###############################################################################

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
# bug: unix no exist error, replace by unx.
import ds4n6_lib.unx   as d4unx

###############################################################################
# FUNCTIONS
###############################################################################

# Hidden columns in simple() funcion
hiddencols = [ 'MTStampEpoch_', 'MTStamp_', 'ATStampEpoch_', 'ATStamp_', 'CTStampEpoch_', 'CTStamp_', 'Meta_', 'FileStem_', 'ParentPath_', 'ParentName_', 'PathSeparator_', 'FilePath-Hash_', 'FileName-Hash_', 'FileStem-Hash_', 'ParentPath-Hash_', 'ParentName-Hash_'] 

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
        print("DEBUG: [macrobber-read_data()]")

    header_names = ['MD5', 'path', 'inode', 'mode_as_string', 'UID', 'GID', 'size', 'atime', 'mtime', 'ctime', 'block_size'] 

    kwargs['header_names'] = header_names

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
    data_os           = kwargs.get('data_os',           None)
    generate_hashes   = kwargs.get('generate_hashes',   True)
    path_prefix       = kwargs.get('path_prefix',       None)

    # Specific Harmonization Pre-Processing ===================================
    def remove_prefix(df, prefixregex):
        if 'FilePath_' in df.columns:
            df['FilePath_'] = df['FilePath_'].str.replace(prefixregex,'')
        return df

    # Harmonize to File_List_HAM

    # PathSeparator is tool-dependent, not only OS-dependent
    pathsep = '/'

    df['MTStampEpoch_']  = df['mtime']
    df['MTStamp_']       = pd.to_datetime(df['mtime'], errors = 'coerce',  unit='s')
    df['MTStampDate_']   = df['MTStamp_'].dt.date
    df['MTStampTime_']   = df['MTStamp_'].dt.ceil(freq='s').dt.time
    df['MTStampDoW_']    = df['MTStamp_'].dt.day_name()
    df['ATStampEpoch_']  = df['atime']
    df['ATStamp_']       = pd.to_datetime(df['atime'], errors = 'coerce',  unit='s')
    df['ATStampDate_']   = df['ATStamp_'].dt.date
    df['ATStampTime_']   = df['ATStamp_'].dt.ceil(freq='s').dt.time
    df['ATStampDoW_']    = df['ATStamp_'].dt.day_name()
    df['CTStampEpoch_']  = df['ctime']
    df['CTStamp_']       = pd.to_datetime(df['ctime'], errors = 'coerce',  unit='s')
    df['CTStampDate_']   = df['CTStamp_'].dt.date
    df['CTStampTime_']   = df['CTStamp_'].dt.ceil(freq='s').dt.time
    df['CTStampDoW_']    = df['CTStamp_'].dt.day_name()
   #df['BTStampEpoch_']  = df['btime']
   #df['BTStamp_']       = pd.to_datetime(df['btime'], errors = 'coerce',  unit='s')
   #df['BTStampDate_']   = df['BTStamp_'].dt.date
   #df['BTStampTime_']   = df['BTStamp_'].dt.ceil(freq='s').dt.time
   #df['BTStampDoW_']    = df['BTStamp_'].dt.day_name()
    df['Size_']          = df['size'].astype('int64')
   #df['Mode_']          = None
    if not data_os == "windows":
        df['UID_']       = df['UID']
    if not data_os == "windows":
        df['GID_']       = df['GID']
    df['Meta_']          = df['inode']
   #df['File_Name']      = None
    df['Type_']          = df['mode_as_string'].str.extract('^(.)')
   #df['PrevType_']      = None
    if not data_os == "windows":
        df['Permissions_'] = df['mode_as_string'].str.replace('^.','').str.replace(r'\ .*$','')
   #df['Deleted_']       = None
   #df['Reallocated_']   = None
   #df['Hostname_']      = None
    if not df['MD5'].iloc[0] == 0:
        df['MD5_Hash_']  = df['MD5']
   #df['SHA256_Hash_']   = None
   #df['DriveLetter_']   = None
   #df['VSS_']           = None
   #df['EVOName_' ]      = None
   #df['EvidenceName_']  = None
   #df['Partition_']     = None
   #df['Tag_']           = None
    df['FilePath_']      = df['path']
    if path_prefix is not None:
        df = remove_prefix(df, path_prefix)
    df['FileName_']      = df['FilePath_'].str.replace('.*'+pathsep,'')
    df['FileStem_']      = df['FileName_'].str.replace(r'\.[^\.]*$','')
    df['FileExtension_'] = df['FileName_'].str.replace(r'^[^\.]*$', '').str.replace(r'.*\.','').str.lower()
    df['ParentPath_']    = df['FilePath_'].str.replace('(.*)'+pathsep+'.*','\\1')
    df['ParentName_']    = df['ParentPath_'].str.replace('.*'+pathsep,'')
    df['PathSeparator_'] = pathsep
   #df['FSType_']        = None
   #df['TSNTFSAttr_']    = None


    # Path-Hash Fields  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if generate_hashes:
        df['FilePath-Hash_']   = df['FilePath_'].str.lower().apply(hash)
        df['FileName-Hash_']   = df['FileName_'].str.lower().apply(hash)
        df['FileStem-Hash_']   = df['FileStem_'].str.lower().apply(hash)
        df['ParentPath-Hash_'] = df['ParentPath_'].str.lower().apply(hash)
        df['ParentName-Hash_'] = df['ParentName_'].str.lower().apply(hash)

    # Generic Harmonization ===================================================
    df = d4com.harmonize_common(df, datatype='flist', **kwargs)

    # Specific Harmonization Post-Processing ==================================

    # return ==================================================================

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
        print("DEBUG: [mcrb] [simple_func()]")

    # Artifact-specific argument parsing =======================================

    # Variables ================================================================
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
        print("Available macrobber analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-macrobber", objtype):
            anlav = True
            print("- macrobber_files:  No.events macrobber file (Input: macrobberdfs)")

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

    # mcrbdfs ------------------------------------------------------------------
    # if   re.search("^dict-pandas_dataframe-macrobber", objtype):
    #     if anltype == "macrobber_files":
    #         return analysis_macrobber_files(*args, **kwargs)

    print("INFO: [d4mcrb] No analysis functions available for this data type ("+objtype+")")

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4mcrb")
class Ds4n6McrbAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


@pd.api.extensions.register_dataframe_accessor("d4_macrobber")
class Ds4n6MacRobberAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)
