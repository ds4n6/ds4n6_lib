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
    print("    - Lowercasing: ", end='')
    str_fields = ['hostname', 'user', 'source_hostname', 'remote_user', 'remote_domain', 'source_artifact']
    df[str_fields] = df[str_fields].fillna('d4_null')  # New (solve error or all samples in a column==Null)
    for field in str_fields:
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
    df['Computer_'] = df['Computer_'].str.replace('^\\\\','',regex=True)

    # ComputerName_ (derived from Computer)
    print("ComputerName_ ", end='')
    df.insert(8,'ComputerName_',df['Computer_'])
    df['ComputerName_'] = np.where(df['Computer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['Computer_'])
    df['ComputerName_'] = df['ComputerName_'].str.replace("\..*$","",regex=True)
    df['ComputerName_'] = df['ComputerName_'].astype('string')

    # ComputerDomain_ (derived from Computer)
    print("ComputerDomain_ ", end='')
    df.insert(9,'ComputerDomain_',df['Computer_'])
    df['ComputerDomain_'] = np.where(df['ComputerDomain_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['ComputerDomain_'])
    df['ComputerDomain_'] = df['ComputerDomain_'].str.replace("^[^\.]*\.","",regex=True)
    df['ComputerDomain_'] = df['ComputerDomain_'].str.replace("^[^\.]*$","d4_null",regex=True)
    df['ComputerDomain_'] = df['ComputerDomain_'].astype('string')

    # ComputerIP_ (derived from Computer)
    print("ComputerIP_ ", end='')
    df.insert(10,'ComputerIP_',df['Computer_'])
    df['ComputerIP_'] = np.where(df['Computer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), df['Computer_'], "d4_null")
    df['ComputerIP_'] = df['ComputerIP_'].astype('string')

    # user -> UserID_ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are empty -> Replace by d4_null
    df['UserID_'] = df['UserID_'].str.replace("^\ *$","d4_null",regex=True)

    # source_ip -> SourceIP_  - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are Hostnames
    # EvtIDs: 31001 1024 4624 1102 1149 24 21 22 25   
    # TODO: Separate between SouceIP_ & SourceComputer_

    # source_hostname -> SourceComputer_  - - - - - - - - - - - - - - - - - - - 

    # TODO: Some entries have and other don't have domain name -> Create additional cols (SourceDomainName)
    # Some entries are empty -> Replace by d4_null

    df['SourceComputer_'] = df['SourceComputer_'].str.replace("^\ *$","d4_null",regex=True)

    # logon_type -> LogonType_  - - - - - - - - - - - - - - - - - - - - - - - - 

    print("LogonType ")
    # TODO: Review this vvvv
    # EventID 4648 has an empty LogonType. We will set it to -1
    df.loc[df['LogonType_'] == "", 'LogonType_'] = -1

    # Then we can set the LogonType_ column to int
    df = df.astype({'LogonType_': int})

    # remote_user -> TargetUserName_  - - - - - - - - - - - - - - - - - - - - - 

    # Some entries are empty -> Replace by d4_null
    df['TargetUserName_'] = df['TargetUserName_'].str.replace("^\ *$","d4_null",regex=True)

    # remote_domain -> TargetComputer_  - - - - - - - - - - - - - - - - - - - - 

    # TODO: remote_domain should be "TargetComputer_", right?

    # Some entries include a domain -> Created additional col (TargetDomainName_)
    # Some entries are empty -> Replace by d4_null
    df['TargetComputer_'] = df['TargetComputer_'].str.replace("^\ *$","d4_null",regex=True)

    # TargetHostname_ 
    df['TargetHostname_'] = np.where(df['TargetComputer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['TargetComputer_'])
    df['TargetHostname_'] = df['TargetHostname_'].str.replace("\..*$","",regex=True)
    df['TargetHostname_'] = df['TargetHostname_'].astype('string')

    # TargetDomain_ (derived from TargetHostname_)
    df['TargetDomain_'] = np.where(df['TargetComputer_'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), "d4_null", df['TargetComputer_'])
    df['TargetDomain_'] = df['TargetDomain_'].str.replace("^[^\.]*\.","",regex=True)
    df['TargetDomain_'] = df['TargetDomain_'].str.replace("^[^\.]*$","d4_null",regex=True)
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


def ml_logons_anomalies(df, model="simple_autoencoder", top_n=50, train_epochs = 5): # iforest | simple_autoencoder
    # 1) FEATURE ENGINEERING
    logondf = df[(df['EventID_'] == 4624) | (df['EventID_'] == 4625)].astype(str)
    filtered_logondf = logondf[['Timestamp_', 'TargetUserName_','LogonType_','EventID_']]  # Row selection for the analysis
    filtered_logondf.insert(1, 'LatMov_', "[" + logondf['SourceIP_'] + "]->[" + logondf['ComputerName_'] + "]")  # Combine two columns [src_ip]->[dst_hostname]
    
    # 2) DATA PROCESSING (Group samples by Timeslot, LatMov & User)
    timestamp, latmov, target_user, logon_type, ev_4624, ev_4625 = [],[],[],[],[],[]

    filtered_logondf.loc[:, 'Timestamp_'] = pd.to_datetime(filtered_logondf['Timestamp_'])
    for hour, df_hour in filtered_logondf.groupby(pd.Grouper(key='Timestamp_', freq='T')):  # Group samples by time-windows (T = 1 minute)
        movs = df_hour['LatMov_'].unique()
        for mov in movs:  # Group samples by Lateral Movement
            mov_df = df_hour[df_hour['LatMov_']==mov]
            usrs = mov_df['TargetUserName_'].unique()
            for usr in usrs:  # Group samples by User
                usr_df = mov_df[mov_df['TargetUserName_']==usr]
                types = usr_df['LogonType_'].unique()
                for type in types:
                    type_df = usr_df[usr_df['LogonType_']==type]
                    timestamp.append(hour)
                    latmov.append(type_df['LatMov_'].iloc[0])
                    target_user.append(usr)
                    logon_type.append(type)
                    ev_4624.append(type_df['EventID_'].str.count('4624').sum())
                    ev_4625.append(type_df['EventID_'].str.count('4625').sum())

    analysis_df = pd.DataFrame({
        'Timestamp_': timestamp,
        'LatMov_': latmov,
        'TargetUserName_': target_user,
        'LogonType_': logon_type,
        'EventID_4624_': ev_4624,
        'EventID_4625_': ev_4625
    })

    # 3) CONVERT CATEGORICAL FEATURES INTO NUMERICAL FEATURES (One-Hot Encoding)
    dumies = pd.get_dummies(analysis_df[['LogonType_','TargetUserName_','LatMov_']]).astype(int)
    train_df = pd.concat([dumies, analysis_df[['EventID_4624_','EventID_4625_']]], axis=1)

    # 4) ANOMALY DETECTION
    if model == 'iforest':
        from sklearn.ensemble import IsolationForest
        iforest = IsolationForest(
                        n_estimators  = 1000,
                        max_samples   ='auto',
                        contamination = 0.1,
                        random_state  = 1,
                    )

        iforest.fit(X=train_df.values)
        anomaly_scores = iforest.decision_function(train_df.values)
        anomaly_predictions = iforest.predict(train_df.values)
        anomaly_predictions[anomaly_predictions == 1] = 0
        anomaly_predictions[anomaly_predictions == -1] = 1

        anomaly_df = analysis_df[anomaly_predictions.astype(bool)].copy()
        a_score = list(anomaly_scores[anomaly_predictions.astype(bool)])
        anomaly_df['score'] = a_score
        output = anomaly_df.sort_values(by=['score']).reset_index(drop=True)
        return output[0:top_n]

    elif model == 'simple_autoencoder':
        import tensorflow as tf
        from tensorflow.keras import Model, Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))  
        x_train_scaled = min_max_scaler.fit_transform(train_df.copy())  # Normalization
        x_test_scaled = x_train_scaled

        class AutoEncoder(Model):
            def __init__(self, output_units):
                super().__init__()
                self.encoder = Sequential([  # ENCODER
                Dense(32, activation='relu'),
                Dropout(0.1),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(4, activation='relu')])

                self.decoder = Sequential([  # DECODER
                Dense(16, activation='relu'),
                Dense(32, activation='relu'),
                Dense(output_units, activation='sigmoid')])
            
            def call(self, inputs):
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                return decoded

        def get_predictions(model, x_test_scaled):
            predictions = model.predict(x_test_scaled)
            errors = tf.keras.losses.msle(predictions, x_test_scaled)
            return errors
        
        # TRAIN AUTOENCODER
        print("Training Autoencoder...")
        print("_____________________________________________________________________________________")
        model = AutoEncoder(output_units=x_train_scaled.shape[1])
        model.compile(loss='msle', metrics=['mse'], optimizer='adam')

        history = model.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=train_epochs,
            batch_size=10)

        errors = get_predictions(model, x_test_scaled)
        anomaly_df = analysis_df.copy()
        anomaly_df['score'] = errors
        output = anomaly_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        return output[0:top_n]

    else:
        raise ValueError("Error: model '" + model + "' not supported. Try 'iforest' or 'simple_autoencoder'")

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


