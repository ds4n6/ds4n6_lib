# DS4N6
#
# Description: Library of functions to apply Data Science to forensics artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4flst

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
# VARIABLES
###############################################################################
hiddencols = [ 'MTStampEpoch_', 'MTStamp_', 'ATStampEpoch_', 'ATStamp_', 'CTStampEpoch_', 'CTStamp_', 'Meta_', 'FileStem_', 'ParentName_', 'ParentPath_', 'ParentMeta_', 'PathSeparator_', 'FilePath-Hash_', 'FileName-Hash_', 'FileStem-Hash_', 'ParentPath-Hash_', 'ParentName-Hash_', 'NTFS-SeqNumber_', 'ParentSeqNumber_', 'ParentPath', 'NTFS-ReferenceCount_', 'NTFS-ReparseTarget_', 'IsDirectory_', 'NTFS-HasAds_', 'NTFS-IsAds_', 'NTFS-SI<FN_', 'NTFS-uSecZeros_', 'NTFS-Copied_', 'NTFS-SiFlags_', 'NTFS-NameType_', 'NTFS-FN-BTime_', 'NTFS-FN-MTime_', 'NTFS-FN-CTime_', 'NTFS-FN-ATime_', 'NTFS-UpdateSequenceNumber_', 'NTFS-LogfileSequenceNumber_', 'NTFS-SecurityId_', 'NTFS-ObjectIdFileDroid_', 'NTFS-LoggedUtilStream_', 'NTFS-ZoneIdContents_', ]

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

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

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
def analysis(*args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

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
        print("Available flist analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^pandas_dataframe-flist", objtype):
            anlav = True
            print("- size_top_n:  Top files by size (Input: flistdf)")
            print("- exefile:  Analysis of the multiple instances of a specific file on many hosts (Input: flistdf)")
            print("- unique_files_folder:  Find unique files (Input: flistdf)")
            print("- exefs:  Macro analysis of all EXEs in the exefs df as a whole (Input: flistdf)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')

    if d4.debug >= 2:
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

    # ANALYSIS FUNCTIONS ======================================================

    # flistdf ------------------------------------------------------------------
    if   re.search("^pandas_dataframe-flist", objtype):
        if anltype == "size_top_n":
            return analysis_size_top_n(obj)
        if anltype == "exefile":
            return analysis_exefile(obj)
        if anltype == "unique_files_folder":
            return analysis_unique_files_folder(obj)
        if anltype == "exefs":
            return analysis_exefs(obj)

    print("INFO: [d4flist] No analysis functions available for this data type ("+objtype+")")

# Specific Analysis Functions =================================================

def analysis_size_top_n(fstl, n=100):
    return fstl[~fstl['FileName_'].str.contains(r"\(\$FILE_NAME\)")][['Size_','FileName_']].sort_values(by='Size_', ascending=False).drop_duplicates().head(n)


def analysis_exefile(exefs, thisexef_path='notepad.exe'):
    # Description:
    #     Analysis of the multiple instances of a specific file on many hosts
    #     - ...

    # Variables
    exef_intg_max_occs = 3
    print("PARAMETERS --------------------------------------------------------\n")
    print("exe path: {}".format(thisexef_path))
    print("EXEFILE ANALYSIS --------------------------------------------------------\n")

    # Select instances of exef
    exefs[exefs['FilePath_'].str.contains(thisexef_path,case=False)].head()
    thisexefs = exefs[exefs['FilePath_'].str.contains(thisexef_path,case=False)]

    print("fsize   ANALYSIS - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    exefgrps = thisexefs.groupby('Size_')
    exefgrps_groups = exefgrps.groups
    nexefgrps = len(exefgrps_groups)
    print("No.groups: " + str(nexefgrps) + "\n")
    # exef_sizes = exefgrps.groups.keys()
    exef_sizes_occs = exefgrps.size()
    print("Groups (sorted by no. occurrances of 'fsize'):    ")
    print(exef_sizes_occs.sort_values(ascending=False))

    print("Interesting (no. occurrences <=" + str(exef_intg_max_occs) + "):    ")
    exef_intg=exefgrps.filter(lambda x: len(x) <= exef_intg_max_occs)
    print(exef_intg)
    return exef_intg

    
def analysis_unique_files_folder(exefs, thisexed_path="/Windows/Temp", exef_intg_max_occs="100", compop='==', recurse=False, prevdays=0, tsfield='m', verbose=False):
    print("PARAMETERS --------------------------------------------------------\n")
    print("exe path: {}".format(thisexed_path))
    print("max occs: {}".format(exef_intg_max_occs))
    print("compop: {}".format(compop))
    print("recurse: {}".format(recurse))
    print("prevdays: {}".format(prevdays))
    print("verbose: {}".format(verbose))
    print("-------------------------------------------------------------------\n")
    if compop not in ['>', '<', '>=', '==', '<=']:
        print("Invalid Comparison Operator: "+compop)
        return False

    regexrec=thisexed_path+"/"
    regexnorec=thisexed_path+"/[^/]*$"

    if recurse :
        thisexefsrec=exefs[exefs['FilePath_'].str.contains(regexrec,case=False,regex=True)]
        nexefsrec=len(thisexefsrec)
        thisexefs=thisexefsrec
        if verbose :
            print("No. files (recursive):     "+str(nexefsrec)+"\n")
    else:
        thisexefsnorec=exefs[exefs['FilePath_'].str.contains(regexnorec,case=False,regex=True)]
        nexefsnorec=len(thisexefsnorec)
        thisexefs=thisexefsnorec
        if verbose :
            print("No. files (non-recursive): "+str(nexefsnorec)+"\n")

    exefgrps = thisexefs.groupby('FilePath_')
    exefgrps_groups = exefgrps.groups
    nexefgrps = len(exefgrps_groups)
    # exef_sizes=exefgrps.groups.keys()
    # exef_sizes_occs=exefgrps.size()
    if verbose :
        print("phash ANALYSIS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
        print("RECURSION: "+str(recurse))
        print("No.groups: "+str(nexefgrps)+"\n")

    if prevdays == 0 :
        exef_intg = exefgrps.filter(lambda x: eval( str(len(x)) + compop + str(exef_intg_max_occs)) )
    else:
        print("No. Interesting (no. occurrences <=" + str(exef_intg_max_occs) + "): " + str(exef_intg) + "\n")
        lastmtime = exef_intg.sort_values(by="MTime_").tail(1)['MTime_']
        print("Last mtime: " + str(lastmtime))
        prevdate = lastmtime + pd.DateOffset(days=-prevdays)
        print("Previous Date: "+  str(prevdate))

    return exef_intg


def analysis_exefs(exefs,  thisexed_path="/Windows/Temp", date_from='2019-12-01'):

    # Description:
    #     Macro analysis of all EXEs in the exefs df as a whole
    #     - Files that appear only a few times, etc.
    print("PARAMETERS --------------------------------------------------------\n")
    print("date from: {}".format(date_from))
    # path-hash analysis  ===============================================================
    # rare_phash_occs = 3
    # thisexefilegrps = thisexefs.groupby('path-hash')
    # thisexefilegrps_groups = thisexefilegrps.groups
    # nthisexefilegrps = len(thisexefilegrps_groups)
    # thisexefile_phash = thisexefilegrps.groups.keys()
    # thisexefile_phash_occs = thisexefilegrps.size()
    # print("Groups (sorted by no. occurrances of 'path-hash'):    ")
    # print(thisexefile_phash_occs.sort_values(ascending=False))
    # thisexefile_phash_rare = thisexefile_phash_occs[thisexefile_phash_occs == 1].sort_values(ascending=False)

    # Rare Files Analysis ---------------------------------------------------------------
    # Files which appear only n times in the whole host set. 
    # Since we are grouping by phash, this means they appear once in each of the n hosts
    # If n=1 -> file appears only in 1 host
    # exef_intg = exefgrps.filter(lambda x: len(x) <= n)    

    # Creation time analysis ------------------------------------------------------------
    # Files created recently
    return exefs[exefs['ATime_'] > date_from]
# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4fsl")
class Ds4n6FslAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


@pd.api.extensions.register_dataframe_accessor("d4_flist")
class Ds4n6FSListAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)
