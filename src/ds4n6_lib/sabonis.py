# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4sbns

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

# DS IMPORTS -----------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
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
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    hostname     = kwargs.get('hostname',     None)


    # Specific Harmonization Pre-Processing =================================== 

    print("  + Applying tool-specific harmonization (pre)...")

    # TODO: Verify if the sabonis header is in place:
    # time,event_id,hostname,user,source_ip,source_hostname,logon_type,remote_user,remote_domain,source_artifact

    # Generic Harmonization ===================================================
    print("  + Applying common harmonization...")
    df = d4com.harmonize_common(df, datatype="sabonis", **kwargs)

    # Specific Harmonization Post-Processing ==================================

    print("  + Applying tool-specific harmonization (post)...")

    # See: https://github.com/jupyterj1s/sabonis/blob/main/evtxplayer.py
    #
    # sabonis logs:
    # - Security.evtx
    # - Microsoft-Windows-RemoteDesktopServices-rdpcorets%4operational.evtx
    # - Microsoft-Windows-TerminalServices-RemoteConnectionManager%4Operational.evtx
    # - Microsoft-Windows-TerminalServices-LocalSessionManager%4Operational.evtx
    # - Microsoft-Windows-TerminalServices-RDPClient%4operational.evtx
    # - Microsoft-Windows-SMBClient%4Security.evtx
    # - Microsoft-Windows-SMBServer%4Security.evtx

    # Security.evtx - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    # interesting_eventids = ["4624", "4625","4648","4778","4647","4634","4779"]
    #
    # computername     -> Computer         
    # user             -> UserID           
    # source_ip        -> IpAddress        
    # source_hostname  -> WorkstationName     
    # logon_type       -> LogonType        
    # remote_user      -> TargetUserName   
    # remote_domain    -> TargetDomainName  
    #
    # Microsoft-Windows-RemoteDesktopServices-rdpcorets%4operational.evtx - - - - - - - - - - -
    #
    # interesting_eventids = ["131"]
    #
    # computername     -> Computer
    # user             -> UserID
    # source_ip        -> ClientIP
    # source_hostname  -> -
    # logon_type       -> 10 (hardcoded)
    # remote_user      -> -
    # remote_domain    -> -
    #
    # Microsoft-Windows-TerminalServices-RemoteConnectionManager%4Operational.evtx  - - - - - -
    #
    # interesting_eventids = ["1149"]
    #
    # computername     -> Computer
    # user             -> UserID
    # source_ip        -> Param3
    # source_hostname  -> -
    # logon_type       -> 10 (hardcoded)
    # remote_user      -> Param1
    # remote_domain    -> Param2
    #
    # Microsoft-Windows-TerminalServices-LocalSessionManager%4Operational.evtx  - - - - - - - - 
    #
    # interesting_eventids = ["21","22","24","25"]
    #
    # computername     -> Computer
    # user             -> UserID
    # source_ip        -> Address
    # source_hostname  -> -
    # logon_type       -> 10 (hardcoded)
    # remote_user      -> User (suffix)
    # remote_domain    -> User (prefix)
    #
    # Microsoft-Windows-TerminalServices-RDPClient%4operational.evtx  - - - - - - - - - - - - -
    #
    # interesting_eventids = ["1024","1102"] 
    #
    # computername     -> Name
    # user             -> -
    # source_ip        -> Computer
    # source_hostname  -> Computer
    # logon_type       -> 3 (hardcoded) ???
    # remote_user      -> UserID
    # remote_domain    -> UserName
    #
    # Microsoft-Windows-SMBClient%4Security.evtx  - - - - - - - - - - - - - - - - - - - - - - -
    #
    # interesting_eventids = ["31001"]
    #
    # computername     -> ServerName
    # user             -> -
    # source_ip        -> Computer
    # source_hostname  -> Computer
    # logon_type       -> 3 (hardcoded)
    # remote_user      -> UserName
    # remote_domain    -> -
    #
    # Microsoft-Windows-SMBServer%4Security.evtx  - - - - - - - - - - - - - - - - - - - - - - -
    #
    # interesting_eventids = ["1009"]
    #
    # computername     -> Computer
    # user             -> UserID
    # source_ip        -> ClientName
    # source_hostname  -> -
    # logon_type       -> 3 (hardcoded)
    # remote_user      -> UserName
    # remote_domain    -> -

    # sabonis Notes:
    # - user 
    #   + empty for Security:4624
    #   + a sid for MSTS-RemoteConnMgrOp ; there actually is a username in that output
    # - source_hostname:
    #   + empty for MSTS-RemoteConnMgrOp

    # Harmonization ------------------------------------------------------------

    # Renaming Overview:
    # time		-> Timestamp_
    # event_id		-> EventID_
    # hostname		-> Computer_
    # user		-> UserID_
    # source_ip		-> SourceIP_
    # source_hostname	-> SourceComputer_
    # logon_type	-> LogonType_
    # remote_user	-> TargetUserName_
    # remote_domain	-> TargetComputer_
    # source_artifact	-> evtxFileName_

    # TODO: Verify if timezone can be different than utc and what happens then 
    print("    - Lowercasing: ", end='' )
    for field in ['hostname', 'user', 'source_hostname', 'remote_user', 'remote_domain', 'source_artifact']:
        print(field + " ", end='')
        df[field] = df[field].str.lower()
    print('')

    print("    - Renaming columns...")
    #df = df.rename(columns={"time": "Timestamp_", "event_id": "EventID_", "hostname": "Computer", "source_ip": "IpAddress_", "source_hostname": "WorkstationName_", "logon_type": "LogonType_", "remote_user": "TargetUserName_", "remote_domain": "TargetDomainName_", "source_artifact": "evtxFileName_"})
    df = df.rename(columns={"time": "Timestamp_", "event_id": "EventID_", "hostname": "Computer_", "user": "UserID_", "source_ip": "SourceIP_", "source_hostname": "SourceComputer_", "logon_type": "LogonType_", "remote_user": "TargetUserName_", "remote_domain": "TargetComputer_", "source_artifact": "evtxFileName_"})
    #df['Date_'] = df['Timestamp_'].dt.date
    #df['Time_'] = df['Timestamp_'].dt.ceil('s').dt.time

    print("    - Harmonization: ", end='')

    # time -> Timestamp_  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    print("Timestamp_ ", end='')
    df['Timestamp_'] = df['Timestamp_'].str.replace(' utc','')
    df['Timestamp_'] = pd.to_datetime(df['Timestamp_'])
    
    # event_id -> EventID_  - - - - - - - - - - - - - - - - - - - - - - - - - -

    # EventID_ does not require Harmonization

    # hostname -> Computer_ - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if not hostname == None:
        df['D4_Hostname_'] = hostname

    # Computer_
    print("Computer_ ", end='')
    df['Computer_'] = df['Computer_'].str.replace('^\\\\','')

    # ComputerName_ (derived from Computer)
    print("ComputerName_ ", end='')
    df.insert(8,'ComputerName_',df['Computer_'])
    df['ComputerName_'] = np.where(df['Computer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['Computer_'])
    df['ComputerName_'] = df['ComputerName_'].str.replace("\..*$","")
    df['ComputerName_'] = df['ComputerName_'].astype('string')

    # ComputerDomain_ (derived from Computer)
    print("ComputerDomain_ ", end='')
    df.insert(9,'ComputerDomain_',df['Computer_'])
    df['ComputerDomain_'] = np.where(df['ComputerDomain_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['ComputerDomain_'])
    df['ComputerDomain_'] = df['ComputerDomain_'].str.replace("^[^\.]*\.","")
    df['ComputerDomain_'] = df['ComputerDomain_'].str.replace("^[^\.]*$","d4_null")
    df['ComputerDomain_'] = df['ComputerDomain_'].astype('string')

    # ComputerIP_ (derived from Computer)
    print("ComputerIP_ ", end='')
    df.insert(10,'ComputerIP_',df['Computer_'])
    df['ComputerIP_'] = np.where(df['Computer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), df['Computer_'], "d4_null")
    df['ComputerIP_'] = df['ComputerIP_'].astype('string')

    # user -> UserID_ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are empty -> Replace by d4_null
    df['UserID_'] = df['UserID_'].str.replace("^\ *$","d4_null")

    # source_ip -> SourceIP_  - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are Hostnames
    # EvtIDs: 31001 1024 4624 1102 1149 24 21 22 25   
    # TODO: Separate between SouceIP_ & SourceComputer_

    # source_hostname -> SourceComputer_  - - - - - - - - - - - - - - - - - - - 

    # TODO: Some entries have and other don't have domain name -> Create additional cols (SourceDomainName)
    # Some entries are empty -> Replace by d4_null

    df['SourceComputer_'] = df['SourceComputer_'].str.replace("^\ *$","d4_null")

    # logon_type -> LogonType_  - - - - - - - - - - - - - - - - - - - - - - - - 

    print("LogonType ")
    # TODO: Review this vvvv
    # EventID 4648 has an empty LogonType. We will set it to -1
    df.loc[df['LogonType_'] == "", 'LogonType_'] = -1

    # Then we can set the LogonType_ column to int
    df = df.astype({'LogonType_': int})

    # remote_user -> TargetUserName_  - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are empty -> Replace by d4_null
    df['TargetUserName_'] = df['TargetUserName_'].str.replace("^\ *$","d4_null")

    # remote_domain -> TargetComputer_  - - - - - - - - - - - - - - - - - - - - 

    # TODO: remote_domain should be "TargetComputer_", right?

    # Some entries include a domain -> Created additional col (TargetDomainName_)
    # Some entries are empty -> Replace by d4_null
    df['TargetComputer_'] = df['TargetComputer_'].str.replace("^\ *$","d4_null")

    # TargetHostname_ 
    df['TargetHostname_'] = np.where(df['TargetComputer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['TargetComputer_'])
    df['TargetHostname_'] = df['TargetHostname_'].str.replace("\..*$","")
    df['TargetHostname_'] = df['TargetHostname_'].astype('string')

    # TargetDomain_ (derived from TargetHostname_)
    df['TargetDomain_'] = np.where(df['TargetComputer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['TargetComputer_'])
    df['TargetDomain_'] = df['TargetDomain_'].str.replace("^[^\.]*\.","")
    df['TargetDomain_'] = df['TargetDomain_'].str.replace("^[^\.]*$","d4_null")
    df['TargetDomain_'] = df['TargetDomain_'].astype('string')

    # TargetIP_ (derived from TargetHostname_)
    df['TargetIP_'] = np.where(df['TargetComputer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), df['TargetComputer_'], "d4_null")
    df['TargetIP_'] = df['TargetIP_'].astype('string')

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

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Variables ----------------------------------------------------------------
    hiddencols =  ['MD5','SHA-1','PESHA-1','PESHA-256','SHA-256','RunspaceId','IMP']

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Call to simple_common ----------------------------------------------------
    return d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

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

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    def syntax():
        print('Syntax: analysis(obj, "analysis_type")\n')
        d4list("str-help")
        return

    def d4list(objtype):
        print("Available sabonis analysis types:")
        print("- find_powershell: Analyze data and find powershell")
 
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

    if re.search("^pandas_dataframe-sabonis", objtype):
        if anltype == "find_powershell":
            return analysis_find_powershell(*args, **kwargs)
    else:
        print("ERROR: [sabonis] Unsupported input data.")
        return

def analysis_find_powershell(obj, *args, **kwargs):
    """ Analysis that finds poweshell in the DF

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    df = obj

    return df.xgrep("*", "powershell", "t" ).spl(out=True, ret=True)

# DATAFRAME ACCESSOR ##########################################################

@pd.api.extensions.register_dataframe_accessor("d4sbns")
class Ds4n6AtrsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_sabonis")
class Ds4n6AutorunsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df = self._obj
        return simple_func(df, *args, **kwargs)


