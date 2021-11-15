#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4evtx

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
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.utils  as d4utl
from ds4n6_lib.knowledge import evtid_desc, evtid_col_map, evtid_col_map_desc, evtx_simple_filters, evtid_groups, windows_builtin_accounts_regex, evilsigs, hidden_cols_evtx

###############################################################################
# FUNCTIONS
###############################################################################


# Convert XML into DF(s) ======================================================

def evtid_dfs_build_raw(df):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Input  (df) is a raw evtx DF
    # Output (df) is a dict of raw evtx DFs (one per eventid)

    dfs={}

    # EventID
    # - In order to be able to extract the evtid DFs we will need an EventID col.
    #   This is somewhat unnatural because this function receives a raw DF
    #   So we will create an EventID col and we will drop it later
    #
    # - Some events have the EventID under "System > EventID", 
    #   while others under "System > EventID > #text"
    #   What we will do is:
    #   + If both exist, we will merge both of them under the "EventID" column.
    #   + If only the "System > EventID" exists, then we will use that column.      
    if "EventID_" not in df.columns:        
        if "System > EventID" in df.columns:
            df['EventID_'] = df['System > EventID']
        else:
            df['EventID_'] = np.nan

        if "System > EventID > #text" in df.columns:            
            df['EventID_'] = df['EventID_'].fillna(df['System > EventID > #text'])
            df['EventID_'] = pd.to_numeric(df['EventID_'])   
            
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('EventID_')))
    df = df.reindex(columns= cols)

    evtids=df['EventID_'].drop_duplicates()
    evtids=evtids.astype(int).sort_values()

    for evtid in evtids:                
        #df[evtid]) #.query('EventID_ == @evtid'))
        dfs[evtid] = df.query('EventID_ == @evtid', engine="python")
        #dfs[evtid] = dfs[evtid].drop(columns=['EventID_'])
        #dfs[evtid] = dfs[evtid].d4evtx.consolidate_columns()
#    print(df.info())
    return dfs

def evtid_dfs_build(df):
    """ 
    Input:
        df: (df) is a raw evtx DF
    Return:
        return (df) is a dict of evtx DFs (one per eventid)
    """       
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    dfs={}

    dfnrows = len(df)

    if dfnrows == 0:
        print("EMPTY INPUT")
        return dfs
     
    # Get D4 Metadata
    orchestrator = df['D4_Orchestrator_'].iloc[0]
    tool         = df['D4_Tool_'].iloc[0]
    plugin       = df['D4_Plugin_'].iloc[0]
    hostname     = df['D4_Hostname_'].iloc[0]

    # EventID
    # - In order to be able to extract the evtid DFs we will need an EventID col.
    #   This is somewhat unnatural because this function receives a raw DF
    #   So we will create an EventID col and we will drop it later
    #
    # - Some events have the EventID under "System > EventID", 
    #   while others under "System > EventID > #text"
    #   What we will do is:
    #   + If both exist, we will merge both of them under the "EventID" column.
    #   + If only the "System > EventID" exists, then we will use that column.
    if "System > EventID" in df.columns:        
        df['EventID_'] = df['System > EventID']
    else:        
        df['EventID_'] = np.nan

    if "System > EventID > #text" in df.columns:
        df['EventID_'] = df['EventID_'].fillna(df['System > EventID > #text'])

    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('EventID_')))
    df = df.reindex(columns= cols)

    evtids=df['EventID_'].drop_duplicates()    
    evtids=evtids.astype(int).sort_values()

    alldf = pd.DataFrame([])

    print("- Parsing event IDs: ", end='')

    for evtid in evtids:
        print(str(evtid)+' ', end='')

        thisdf = df.query('EventID_ == @evtid', engine="python")
        thisdf = thisdf.d4evtx.consolidate_columns()
        thisdf = thisdf.dropna(axis=1, how='all')

        # Harmonize ========================================================
        thisdf = harmonize(thisdf, orchestrator=orchestrator, tool=tool, plugin=plugin, hostname=hostname)

        # Add df to dfs / alldf
        dfs[evtid] = thisdf
        alldf = pd.concat([alldf,thisdf])

    print("")
    
    # alldf / dfs['all']
    alldf = alldf.sort_values(by='EventRecordID')

    dfs['all'] = alldf

    return dfs

# HARMONIZATION FUNCTIONS #####################################################

def harmonize(df, **kwargs):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Specific Harmonization Pre-Processing ===================================

    # Generic -----------------------------------------------------------------
    if 'ProcessID' in df.columns:
        if len(df[df['ProcessID'].isna()]) == 0:
            df['ProcessID']=df['ProcessID'].astype(int)

    if 'ThreadID' in df.columns:
        if len(df[df['ThreadID'].isna()]) == 0:
            df['ThreadID']=df['ThreadID'].astype(int)

    if 'Version' in df.columns:
        if len(df[df['Version'].isna()]) == 0:
            df['Version']=df['Version'].astype(int)

    if 'Opcode' in df.columns:
        if len(df[df['Opcode'].isna()]) == 0:
            df['Opcode']=df['Opcode'].astype(int)
    
    # Security ----------------------------------------------------------------
    if 'LogonType' in df.columns:
        if len(df[df['LogonType'].isna()]) == 0:
            df['LogonType']=df['LogonType'].astype(int)
        else:
            df['LogonType']=df['LogonType'].astype(float)

    # Generic Harmonization ===================================================
    df = d4com.harmonize_common(df, datatype="evtx", **kwargs)

    # Specific Harmonization Post-Processing ==================================

    # return ==================================================================
    df['D4_DataType_'] = 'evtx'

    return df

# ENRICHMENT ##################################################################

# Give an enriched listing of evtid statistics
def evtid_stats(evt):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    counts = evt['EventID_'].value_counts()
    evtidssrv = evtidssr()
    evtidstats = pd.concat([counts, evtidssrv], axis=1, keys=['Count','Description']).dropna().astype({'Count': int})
    return evtidstats

    
# MACHINE LEARNING ############################################################

def ml(evtsdf, type="", **kwargs):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if type == "":
        print("Options: { access }")
        return 
    if type == "access":
        return ml_access_anomalies(evtsdf, **kwargs)

# ML - Access Anomalies =======================================================
def ml_access_anomalies(secevtxdf, **kwargs): 
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    argtsmin      = kwargs.get('argtsmin',      None)
    argtsmax      = kwargs.get('argtsmax',      None)
    mse_threshold = kwargs.get('mse_threshold', None)
    epochs        = kwargs.get('epochs',        40)
   
    dftsmin=secevtxdf.index.min()
    dftsmax=secevtxdf.index.max()

    if argtsmin is None:
        tsmin=dftsmin
    else:
        tsmin=argtsmin

    if argtsmax is None:
        tsmax=dftsmax
    else:
        tsmax=argtsmax

    evts4624 = secevtxdf.query('EventID_ == 4624', engine="python")

    np.random.seed(8)

    # DATA PREPARATION ------------------------------------
    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')].reset_index()
    useraccess           = evts4624_nonsysusers[["Timestamp","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('Timestamp')
    this_useraccess      = useraccess.loc[tsmin:tsmax]
    
    user_access_uwil = this_useraccess[['TargetUserName',"WorkstationName","IpAddress",'LogonType']].copy()
    
    # Lower-case WorkstationName col
    user_access_uwil['WorkstationName'] = user_access_uwil['WorkstationName'].str.lower().fillna("null_workstation")
    user_access_uwil['TargetUserName']  = user_access_uwil['TargetUserName'].str.lower()
    user_access_uwil['LogonType']       = user_access_uwil['LogonType'].astype(str)
    
    user_access_uwil_str = user_access_uwil.copy()
    
    user_access_uwil_str['TU-WN-IP-LT'] = "[" + user_access_uwil['TargetUserName'] + "]" + "[" + user_access_uwil['IpAddress'] + "][" + user_access_uwil['LogonType'] + "]"
    user_access_uwil_str.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)
    user_access_uwil_str = user_access_uwil_str.sort_values(by='TU-WN-IP-LT')

    df = user_access_uwil

    transform_dict = {}
    for col in df.columns:
        cats = pd.Categorical(df[col]).categories
        d = {}
        for i, cat in enumerate(cats):
            d[cat] = i
            transform_dict[col] = d
    
    inverse_transform_dict = {}
    for col, d in transform_dict.items():
           inverse_transform_dict[col] = {v:k for k, v in d.items()}
    
    df = df.replace(transform_dict)

    # SELECTING TRAINING / TEST DATA ----------------------
    X = df
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    print("- Splitting Input Data:")
    print("  + X         -> "+str(X.shape))
    print("    - X_train -> "+str(X_train.shape))
    print("    - X_test  -> "+str(X_test.shape))

    # CREATING THE NEURAL NETWORK ARCHITECTURE ------------
    # There are 4 different input features, and as we plan to use all the features in the autoencoder,
    # we define the number of input neurons to be 4.
    nfeatures    = 4

    input_dim    = X_train.shape[1]
    encoding_dim = nfeatures - 2

    print("- No. Features:    "+str(nfeatures))
    print("- Input Dimension: "+str(input_dim))

    # Input Layer
    input_layer  = Input(shape=(input_dim,))

    # We create an encoder and decoder. 
    # The ReLU function, which is a non-linear activation function, is used in the encoder. 
    # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern

    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(nfeatures, activation='linear')(encoded)

    # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
    # Next, the optimizer and loss function is defined using the compile method.
    # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
    # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
    # The loss is defined as mse, which is mean squared error

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')

    # TRAINING THE NETWORK --------------------------------
    # The training data, X_train, is fitted into the autoencoder. 
    # Let's train our autoencoder for 100 epochs with a batch_size of 4 and observe if it reaches a stable train or test loss value
    
    batch_size=4

    print("- Training:")
    print("  + epochs     = "+str(epochs))
    print("  + batch_size = "+str(batch_size))

    X_train = np.array(X_train)
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)

    # DOING PREDICTIONS -----------------------------------

    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:

    print("- Predictions:")
    predictions = autoencoder.predict(X_train)
    mse = np.mean(np.power(X_train - predictions, 2), axis=1)

    plt.plot(mse)

    # Auto-calculate mse_threshold - Top 5 values
    if mse_threshold is None:
       mse_threshold = int(np.sort(mse)[-15])
       autocalcmsg = " (Auto-calculated)"
    else:
       autocalcmsg = ""

    print("- MSE Threshold: "+str(mse_threshold)+autocalcmsg)

    # Select entries above the mse threshold
    xxx = X_train[mse >= mse_threshold]

    xxxdf = pd.DataFrame(xxx)
    xxxdf.columns = ['TargetUserName', 'WorkstationName', 'IpAddress', 'LogonType']
    anom = xxxdf.replace(inverse_transform_dict)
    print("- No.Anomalies: "+str(len(xxxdf)))
    display(pd.DataFrame(anom.groupby(['WorkstationName', 'IpAddress', 'TargetUserName', 'LogonType']).size()).rename(columns={0: 'Count'}))

    anom_uwil = anom.copy()

    anom_uwil['TU-WN-IP-LT'] = "["+anom_uwil['TargetUserName']+"]["+anom_uwil['IpAddress']+"]["+anom_uwil['LogonType']+"]"
    #anom_uwil.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)

    anom_uwil_uniq_df = pd.DataFrame(anom_uwil['TU-WN-IP-LT'].unique(),columns=['TU-WN-IP-LT']).sort_values(by='TU-WN-IP-LT')
    display(anom_uwil_uniq_df)

    anom_uwil_uniq_df_ts = user_access_uwil_str[user_access_uwil_str['TU-WN-IP-LT'].isin(anom_uwil_uniq_df['TU-WN-IP-LT'])]
    #anom_uwil_uniq_df_ts.head(2)

    # OVERPLOT ANOMALOUS DATA OVER ORIGINAL DATA ------------------------------
    col='TU-WN-IP-LT'
    data=user_access_uwil_str

    plt.figure(figsize=(20,10))

    # Plot original data (green)
    frame = data
    plt.grid(color='g', linestyle='-', linewidth=0.1)
    plt.plot(frame.index, data[col], 'g.')

    # Over-Plot anomalous data (red)
    frame = anom_uwil_uniq_df_ts
    plt.plot(frame.index, frame[col], 'r.')

    plt.show()

    return X_train, mse

# KNOWLEDGE ===================================================================
evtids={}  #patch to fix lint error (there should be a variable renamed)

def evtidsdf():
    evtidssr = pd.Series(evtids)
    evtidsdf = evtidssr.to_frame('Description')
    return evtidsdf

def evtidssr():
    evtidssr = pd.Series(evtids)
    evtidssr.index.astype('int64')

    return evtidssr

# Enrich an evt ID, providing its long description
def evtid_enrich(evtid):
    return evtids['evtid']

# duplicated
# Give an enriched listing of evtid statistics
# def evtid_stats(evt):
#     counts = evt['EventID_'].value_counts()
#     evtidssrv = evtidssr()
#     evtidstats = pd.concat([counts, evtidssrv], axis=1, keys=['Count','Description']).dropna().astype({'Count': int})
#     return evtidstats


# ANALYSIS FUNCTIONS ##########################################################

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

    def syntax_simple():
        print('Syntax: analysis(obj, "analysis_type")\n')

    def syntax():
        syntax_simple()
        d4list("str-help")
        return

    def d4list(objtype):

        # Analysis Modules Available for this objective
        anlav = False
        print("Available evtx analysis types:")
        if objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-evtx-raw", objtype):
            anlav = True
            print("- evtx_files:  No.events & first/last event per evtx file (Input: evtxdfs)")
        if objtype == "str-help" or objtype == "str-list" or re.search("^pandas_dataframe-evtx-ham", objtype):
            anlav = True
            print("- evtid_stats: No.events & first/last event per evtid     (Input: evtxdf)")
        # EXPERIMENTAL. Not enabled by default.
        #if objtype == "mystr-help" or objtype == "str-list" or re.search("^dict-pandas_dataframe-evtx-ham", objtype):
        #    anlav = True
        #    print("- findevil:    Analyze data and find malicious patterns   (Input: evtxdfs)")
        if objtype == "str-help" or objtype == "str-list" or re.search("^pandas_dataframe-evtx-ham", objtype):
            anlav = True
            print("- access:      Logon info summary (4624,...)              (Input: secevtxdf)")
        if objtype == "str-help" or objtype == "str-list" or re.search("^pandas_dataframe-evtx-ham", objtype):
            anlav = True
            print("- failed logons:      Failed Logons info (4625,...)              (Input: secevtxdf)")

        if anlav == False:
            print('- No analysis modules available for this object ('+objtype+').')

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    thisdatatype = "evtx"

    nargs = len(args)

    if nargs == 0:
        syntax()
        return

    obj = args[0]

    objtype = d4com.data_identify(obj)

    if d4.debug >= 4:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] objtype: "+objtype)

    if isinstance(obj, str):
        if obj == "list":
            d4list(objtype)
            return
        if obj == "help":
            syntax()
            return

    if nargs == 1:
        if re.search("^dict-pandas_dataframe-"+thisdatatype, objtype) or re.search("^pandas_dataframe-"+thisdatatype, objtype):
            d4list(objtype)
        else:
            syntax()
        return

    anltype = args[1]

    if not isinstance(anltype, str):
        syntax()
        return

    if anltype == "help":
        syntax_simple()
        d4list(objtype)
        return
    elif anltype == "list":
        d4list(objtype)
        return

    # If object is a dict of dfs
    if   re.search("^dict-pandas_dataframe-evtx-raw", objtype):
        if anltype == "evtx_files":
            return analysis_evtxfiles(*args, **kwargs)
        elif anltype == "findevil":  
            return analysis_findevil(obj, anltype)
        else:
            print("ERROR: [evtx] Unknown analysis type: "+anltype)
            return
    elif re.search("^pandas_dataframe-evtx-ham", objtype):
        if anltype == "access":  
            return analysis_access(*args, **kwargs)
        elif anltype == "evtid_stats":
            return analysis_evtids_stats(*args, **kwargs)
        elif anltype == "failed logons":
            return failed_logons_info(*args, **kwargs)
        else:
            print("ERROR: [evtx] Unknown analysis type: "+anltype)
            return
    else:
        if anltype:  
            print("ERROR: [evtx] The requested analysis type is not available for the provided input data: "+objtype)
        else:
            print("ERROR: [evtx] No analysis modules available for this data type: "+objtype)
        return

# analysis_evtxfiles() ---------------------------------------------------------------

def analysis_evtxfiles(dfs, type="", argtsmin="", argtsmax=""):

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    outdf = pd.DataFrame([],columns=['File','NEvts','TSMin','TSMax'])
    row = pd.Series()

    for evtxf in dfs.keys():
        if len(dfs[evtxf]) != 0:
            row['TSMin'] = dfs[evtxf]['System > TimeCreated > @SystemTime'].min()
            row['TSMax'] = dfs[evtxf]['System > TimeCreated > @SystemTime'].max()
        else:
            row['TSMin'] = ""
            row['TSMax'] = ""
            
        row['File'] = re.sub('.*\\\\','',evtxf)
        row['NEvts'] = len(dfs[evtxf])
        outdf = outdf.append(row,ignore_index=True)

    return outdf


# analysis_evtids_stats() ---------------------------------------------------------------
def analysis_evtids_stats(indf, type="", argtsmin="", argtsmax=""):

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    df = pd.DataFrame(indf['EventID_'].value_counts()).rename(columns={'EventID_': 'Count'})
    df.index.names = ['EventID_']
    df = df.sort_values(by='EventID_')

    evtxlog = indf['evtxFileName_'].iloc[0]

    if evtxlog in evtid_desc.keys():
        edesc = pd.DataFrame(evtid_desc[evtxlog], index=[0]).T
        thisevtids = indf['EventID_'].unique()
        df['Description'] = edesc.query('index in @thisevtids', engine="python")

    # firstseen = pd.DataFrame(indf['EventID_'].drop_duplicates(keep="first")).sort_values(by='EventID_').reset_index().set_index('EventID_')
    # lastseen  = pd.DataFrame(indf['EventID_'].drop_duplicates(keep="last" )).sort_values(by='EventID_').reset_index().set_index('EventID_')
    # df['FirstSeen'] = firstseen['Timestamp']
    # df['LastSeen']  = lastseen['Timestamp']
    df = df.sort_values(by='Count', ascending=False)

    #display(Markdown("**"+str(evtxlog)+"**"))

    return df

# failed_logons_info() ---------------------------------------------------------------

def failed_logons_info(evtsdf, type="", argtsmin="", argtsmax="", graph=True):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
    if evtsdf['EventID_'][0] == 4625:
        failedlogonslist = evtsdf['EventID_']
        failedlogonsdf = pd.DataFrame(failedlogonslist)
        failedlogonsdf['Timestamp'] = failedlogonslist.index
        stats4625 = pd.Series(failedlogonsdf['Timestamp']).resample('D').nunique()
        stats4625df = pd.DataFrame(stats4625)
        stats4625_df = pd.DataFrame()
        stats4625_df['Failed Logons'] = stats4625df['Timestamp']

        if graph :
            failed_logons_graph(evtsdf)
    
        return stats4625_df
    else:
        print("Event ID not supported. This analysis is meant to be used with 4625 Security events.")
        
def failed_logons_graph(obj):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
    
    secevtxdf = obj

    failedlogonslist = secevtxdf['EventID_']
    failedlogonsdf = pd.DataFrame(failedlogonslist)
    failedlogonsdf['Timestamp'] = failedlogonslist.index
    pd.Series(failedlogonsdf['Timestamp']).resample('H').nunique().plot.bar()
    plt.show()
    

# analysis_access() ---------------------------------------------------------------

def analysis_access(evtsdf, type="", argtsmin="", argtsmax="", freq="D", stats=True, graph=True, detail=True):

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if stats :
        statsdf = analysis_access_stats(evtsdf, "stats", argtsmin, argtsmax, freq, detail)

    if graph :
        analysis_access_graph(evtsdf, "graph", argtsmin, argtsmax)

    return statsdf

def analysis_access_stats(obj, type="", firstdate="", lastdate="", freq="D", detail=True):
    # Input  (df) is a raw Security.evtx DF

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    secevtxdf = obj

    evts4624 = secevtxdf.query('EventID_ == 4624', engine="python")

    if firstdate == "":
        firstdate = evts4624.index.min()

    if lastdate == "":
        lastdate = evts4624.index.max()

    evts4624_users = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    evts4624_users['WorkstationName'] = evts4624_users['WorkstationName'].fillna("-")
    userstatsdf = evts4624_users['TargetUserName'].value_counts().reset_index().rename(columns={'TargetUserName': 'Count', 'index': 'TargetUserName'}).sort_values(by='TargetUserName')

    usersfirstdf = pd.DataFrame(evts4624_users['TargetUserName'].drop_duplicates(keep="first").reset_index().sort_values(by='TargetUserName'))
    userslastdf  = pd.DataFrame(evts4624_users['TargetUserName'].drop_duplicates(keep="last").reset_index().sort_values(by='TargetUserName'))
    userssiddf   = pd.DataFrame(evts4624_users[['TargetUserName','TargetUserSid']].drop_duplicates().reset_index(drop=True).sort_values(by='TargetUserName'))
    
    wkndf=evts4624_users['WorkstationName'].value_counts().reset_index().rename(columns={'WorkstationName': 'Count', 'index': 'WorkstationName'})

    ipdf=evts4624_users['IpAddress'].value_counts().reset_index().rename(columns={'IpAddress': 'Count', 'index': 'IpAddress'})

    d4utl.display_side_by_side([wkndf,ipdf,usersfirstdf,userslastdf,userstatsdf,userssiddf] , ['WORKSTATION NAME', 'IP ADDRESS','FIRST', 'LAST', 'STATS','SID'])

    #d4utl.display_side_by_side([usersfirstdf.sort_values(by='TargetUserName'), userslastdf.sort_values(by='TargetUserName'), userstatsdf.sort_values(by='TargetUserName'), userssiddf ] , ['FIRST', 'LAST', 'STATS','SID'])

    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    evts4624_nonsysusers['WorkstationName'] = evts4624_nonsysusers['WorkstationName'].fillna("-")
    useraccess=evts4624_nonsysusers.reset_index()[["Timestamp","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('Timestamp')
    x = useraccess.groupby([pd.Grouper(freq=freq), "WorkstationName", "IpAddress",'TargetUserName','LogonType']).size()
    # try:
    #     x = useraccess.loc[firstdate:lastdate].groupby([pd.Grouper(freq=freq), "WorkstationName", "IpAddress",'TargetUserName','LogonType']).size()
    # except:
    #     x = useraccess.iloc[firstdate:lastdate].groupby([pd.Grouper(freq=freq), "WorkstationName", "IpAddress",'TargetUserName','LogonType']).size()

    if detail :
        # Convert multi-Index to DF
        y = pd.DataFrame(x)
        y.reset_index(inplace=True)
        y['WorkstationName'] = y['WorkstationName'].str.lower()
        y.columns = ['Timestamp','WorkstationName','IpAddress','TargetUserName','LogonType','Count']

        return y

def analysis_access_graph(obj, type="", firstdate="", lastdate=""):
    # Input  (df) is a raw Security.evtx DF

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    secevtxdf = obj

    evts4624 = secevtxdf.query('EventID_ == 4624', engine="python")

    if firstdate == "":
        firstdate=evts4624.index.min()

    if lastdate == "":
        lastdate=evts4624.index.max()

    evts4624_nonsysusers=evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    useraccess=evts4624_nonsysusers.reset_index()[["Timestamp","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('Timestamp')
    user_access_uwil=useraccess[["WorkstationName", "IpAddress",'TargetUserName','LogonType']].copy()
    # try:
    #     user_access_uwil=useraccess[["WorkstationName", "IpAddress",'TargetUserName','LogonType']].loc[firstdate:lastdate].copy()
    # except:
    #     user_access_uwil=useraccess[["WorkstationName", "IpAddress",'TargetUserName','LogonType']].iloc[firstdate:lastdate].copy()

    user_access_uwil['WorkstationName'] = user_access_uwil['WorkstationName'].str.lower()
    user_access_uwil['TargetUserName'] = user_access_uwil['TargetUserName'].str.lower()
    user_access_uwil['LogonType'] = user_access_uwil['LogonType'].astype(str)
    user_access_uwil['IP-WN-TU-LT'] = "["+user_access_uwil['TargetUserName']+"]["+user_access_uwil['WorkstationName']+"]["+user_access_uwil['IpAddress']+"]["+user_access_uwil['LogonType']+"]"
    user_access_uwil.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)

    ## Let's do some graphing
    _fig, ax0 = plt.subplots()

    label = 'IP-WN-TU-LT'
    ihtl = user_access_uwil['IP-WN-TU-LT'].astype("object")
    ihtl = ihtl.dropna().sort_values()
    # Labels
    ax0.set_xlabel('Date')
    ax0.set_ylabel(label, color='g')
    ax0.yaxis.set_label_position("left")
    ax0.plot(ihtl.index, ihtl, 'r.')
    ax0.yaxis.tick_left()
    
    # Create a duplicate of the original xaxis, giving you an additional axis object
    ax1 = ax0.twinx()
    # Set the limits of the new axis from the original axis limits
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_ylabel(label, color='g')
    ax1.yaxis.set_label_position("right")
    ax1.plot(ihtl.index, ihtl, 'r.')
    ax1.yaxis.tick_right()
    
    # Show
    plt.rcParams['figure.figsize'] = 20,10
    
    plt.show()

# analysis_findevil() ---------------------------------------------------------

def analysis_findevil(evtxdfs, anltype="help"):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    evtid_groupsdf = pd.DataFrame.from_dict(evtid_groups)
    avanltypes = list(pd.DataFrame.from_dict(evtid_groups).index)

    findevilexecd = False

    if anltype in avanltypes:
        atledf = pd.DataFrame.from_dict(evtid_groups).loc[anltype][evtid_groupsdf.loc[anltype].notna()]

        evtxlogs = atledf.index

        for evtxlog in evtxlogs:
            # FIND EVIL
            if evtxlog in evilsigs.keys():
                findevilexecd = True
                eseids = pd.Series(evilsigs[evtxlog].keys())
                atleids = atledf.loc[evtxlog] # set
                thiseseids = pd.Series(list(set(eseids) & set(atleids)))
                for thiseseid in thiseseids:
                    for es in evilsigs[evtxlog][thiseseid]:
                        fequery = evilsigs[evtxlog][thiseseid][es]
                        evldf = evtxdfs[evtxlog].d4evtx.simple(str(thiseseid), ret=True, out=False).query(fequery, engine="python")
                        if len(evldf) > 0:
                            display(Markdown("**WARNING: EVIL SIGNATURES FOUND!**"))
                            display(Markdown("  * **Name:** "+es+"\n"+"  * **Signature:** "+fequery))
                            display(evldf)

    if not findevilexecd :
        print("- No findevil rules for the provided evtx file(s)")
    else:
        print("- findevil done.")

# DATAFRAME ACCESSOR ##########################################################

# FUNCTIONS ===================================================================
def column_types_set_func(df):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Timestamp
    if 'System > TimeCreated > @SystemTime' in df.columns:
        df['System > TimeCreated > @SystemTime'] = df['System > TimeCreated > @SystemTime'].astype('datetime64[ns]')

    # Generic - Raw / Processed
    genrawintcols  = ['System > EventID', 'System > Version', 'System > Level', 'System > Task', 'System > Opcode', 'System > EventRecordID', 'System > Execution > @ProcessID', 'System > Execution > @ThreadID', 'System > EventID > #text'] #, 'System > EventID > @Qualifiers'
    genprocintcols = ['ProcessID', 'ThreadID', 'Version', 'Opcode']

    # Security - Raw / Processed
    secrawintcols  = []
    secprocintcols = ['LogonType','KeyLength']

    # Some columns cannot be int because in some cases they can have NaNs, 
    # and in pandas an int column cannot have NaNs (a float column can, though)
    for colset in [ genrawintcols, genprocintcols, secrawintcols, secprocintcols ]:
        for intcol in colset:
            if intcol in df.columns:
                if len(df[df[intcol].isna()]) == 0:                    
                    df[intcol]=df[intcol].astype(int)
                else:
                    df[intcol]=df[intcol].astype(float)

    # EvtID 4697
    if 'ServiceStartType' in df.columns:
         if len(df[df['ServiceStartType'].isna()]) == 0:
             df['ServiceStartType'] = df['ServiceStartType'].astype(int)
 
    return df
        
def consolidate_columns_func(df):
    # A evtx DF consolidation can be applied at any time: on a raw evtx file or on a DF subset of evtids
    # It's useful because some columns EventData > Data > N > @Name|#text columns are variable even
    # within the same evtid
    # So if you have a subset of entries of a specific evtid, it can be interesting to run this function
    # in order to produce a cleaner output

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if len(df) == 0:
        print("INFO: Empty DataFrame")
        return

    evtxf=df.iloc[0]['evtxFileName_']

    outdf=pd.DataFrame([])

    # Drop duplicates
    df = df.drop_duplicates()

    # Define Timestamp (if not defined yet)
    if df.index.name != 'Timestamp':
        df = df.rename(columns={'System > TimeCreated > @SystemTime': 'Timestamp'}).set_index('Timestamp')
        df.index = df.index.astype('datetime64[ns]').round('1s')

    # Drop unnecessary cols
    cols2drop=df.columns[df.columns.str.contains('@xmlns')]
    df = df.drop(columns=cols2drop)

    # EventID
    # - Some events have the EventID under "System > EventID", while others under "System > EventID > #text"
    # - What we will do is:
    #   + If both exist, we will merge both of them under the "EventID" column.
    #   + If only the "System > EventID" exists, then we will use that column.
    # - The EventID_ column may have already been introduced in previous steps, 
    #   so we need to check if that's the case so as not to create a second similar col
    if 'EventID_' not in df.columns:
        if "System > EventID" in df.columns:
            df = df.rename(columns={'System > EventID': 'EventID_'})
        else:
            df['EventID_'] = np.nan

        if "System > EventID > #text" in df.columns:
            df['EventID_'] = df['EventID_'].fillna(df['System > EventID > #text'])
            df = df.drop(columns=['System > EventID > #text'])

    # Rename some columns manually
    df = df.rename(columns={'System > Provider > @Name': 'ProviderName', 'System > Provider > @Guid': 'ProviderGuid', 'System > Version': 'Version', 'System > Level': 'Level', 'System > Task': 'Task', 'System > Opcode': 'Opcode', 'System > Keywords': 'Keywords', 'System > EventRecordID': 'EventRecordID', 'System > Execution > @ProcessID': 'ProcessID', 'System > Execution > @ThreadID': 'ThreadID', 'System > Channel': 'Channel', 'System > Computer': 'Computer','System > Correlation > @ActivityID': '@CorrelationActivityID', 'System > Security > @UserID': '@UserID', 'EventData > @Name': '@Name' })

    # Adjust data types
    df['EventID_'] = df['EventID_'].astype(int) 

    # If there are multiple evtids we sort them by number
    if len(df['EventID_'].drop_duplicates()) != 1:
        evtids=df['EventID_'].drop_duplicates().sort_values()
    else:
        evtids=df['EventID_'].drop_duplicates()

    nevtids=len(evtids)

    if nevtids != 1:
        print("Parsing evtids: ["+evtxf+"] ",end='')

    for evtid in evtids:
        if nevtids != 1:
            print(str(evtid)+" ",end="")
            
        evtiddf = df.query('EventID_ == @evtid', engine="python")

        # If we filter by evtid some cols will not belong to this evtid, so we better drop NaN cols
        evtiddf = evtiddf.dropna(axis=1, how='all')

        evtdatacols = evtiddf.columns[evtiddf.columns.str.contains('EventData > Data > [0-9][0-9]* > @')]

        evtdatacolsmaxnum = evtdatacols.str.replace(".* ([0-9][0-9]*) .*","\\1").drop_duplicates().astype(int).max()

        if evtdatacolsmaxnum >= 0:
            for n in range(0,evtdatacolsmaxnum+1):
                name='EventData > Data > '+str(n)+' > @Name'
                text='EventData > Data > '+str(n)+' > #text'

                # Most of the times there is only one @Name value per eventID, 
                # but sometimes there may be several, and we will need to iterate over them
                if len(evtiddf[name].drop_duplicates()) == 1:
                    colname=str(evtiddf[name].drop_duplicates().iloc[0])
                    evtiddf=evtiddf.drop(columns=name)
                    if colname in evtiddf.columns:
                        colname=colname+"_"
                    evtiddf=evtiddf.rename(columns={ text: colname })
                
        # Consolidate EventData > Data > @Name
        if 'EventData > Data > @Name' in evtiddf.columns:
            name='EventData > Data > @Name'
            text='EventData > Data > #text'

            if len(evtiddf[name].drop_duplicates()) == 1:
                colname=str(evtiddf[name].drop_duplicates().iloc[0])
                evtiddf=evtiddf.drop(columns=name)
                if colname in evtiddf.columns:
                    colname=colname+"_"
                evtiddf=evtiddf.rename(columns={ text: colname })

        # Drop columns with all rows at NaN
        evtiddf = evtiddf.dropna(axis=1, how='all')

        # Re-arrange columns
        # We must check that the specific cols are in the col list, in case they are constant cols
        if 'EventID_' in evtiddf.columns:
            cols = evtiddf.columns.tolist()
            cols.insert(0, cols.pop(cols.index('EventID_')))
            evtiddf = evtiddf.reindex(columns= cols)

        if 'EventRecordID' in evtiddf.columns:
            cols = evtiddf.columns.tolist()
            cols.insert(0, cols.pop(cols.index('EventRecordID')))
            evtiddf = evtiddf.reindex(columns= cols)

        # Concat evtid DFs 
        outdf=pd.concat([evtiddf, outdf])

    # Adjust EventID columns
    if 'System > EventID' in outdf.columns:
        outdf.drop(columns='System > EventID')

    # Sort by EventRecordID
    # - If there is only one record we will not sort
    if len(outdf) > 1:
        outdf=outdf.sort_values('EventRecordID')

    # Round TS to 1 second
    outdf.index = outdf.index.astype('datetime64[ns]').round('1s')

    return outdf

# simple_*_func() ====================================================================

def simple_func(obj, *args, **kwargs):

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    df=obj

    if 'evtxFileName_' in df.iloc[0].keys() and '@xmlns' in df.iloc[0].keys():
        # DF is an evtx file        
        return simple_evtx_file_raw_func(df, *args, **kwargs)
    elif 'evtxFileName_' in df.iloc[0].keys():
        # DF is an evtx file
        return simple_evtx_file_func(df, *args, **kwargs)
    else:
        print("ERROR: Unrecognized input data.")
        return pd.DataFrame([])

# simple ======================================================================
def simple_evtx_file_raw_func(df, *args, **kwargs):
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

    # Artifact-specific argument parsing ---------------------------------------

    if isinstance(args[0], str):
        evtidspar = args[0]
    else:
        if len(args[0]) == 0:
            evtidspar = ""
        else:
            evtidspar = args[0]
            if len(evtidspar) == 0:
                evtidspar = ""
            else:
                evtidspar = str(evtidspar[0])

    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Pre-Processing ----------------------------------------------------------

    # If specific evtids are defined, filter for them
    if evtidspar != "":
        if "," in evtidspar:
            evtidssr = pd.Series(evtidspar.split(",")).astype(int)
        else:
            evtidssr = pd.Series(int(evtidspar))

        if df.index.name != 'Timestamp':
            if "System > EventID" in df.columns:
                df1 = df.query('`System > EventID` in @evtidssr', engine="python")
            else:
                df1 = pd.DataFrame([])

            if "System > EventID > #text" in df.columns:
                df2 = df[df['System > EventID > #text'].isin(evtidssr)]
            else:
                df2 = pd.DataFrame([])

            df = pd.concat([df1,df2]).dropna(how='all')

    # Call to simple_common ----------------------------------------------------
    dfout = d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

    # Post-Processing ---------------------------------------------------------

    # Return ------------------------------------------------------------------
    return dfout


def simple_evtx_file_func(df, *args, **kwargs):

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    apply_filters = kwargs.get('apply_filters', True)
    out_meta_opt  = kwargs.get('out_meta', True)
    out_meta      = out_meta_opt

    # Arg. Parsing -------------------------------------------------------------
    # Show whatever selected on the screen or don't show anything   
    out_opt                = kwargs.get('out',                    True)
    # Show the resulting DF on the screen
    # out_df_opt             = kwargs.get('out_df',                 True)
    # Show the resulting Metadata (Constant/Hidden cols, etc.) on the screen
    out_meta_opt           = kwargs.get('out_meta',               True)
    # Return the resulting DataFrame
    ret_opt                = kwargs.get('ret',                    False)
    # Return the resulting DataFrame as shown in the screen
    # (without hidden cols, collapsed cols, etc.; not in full)
    ret_out_opt            = kwargs.get('ret_out',                False)

    # Check if out is set (will be needed later)
    outargset = 'out' in kwargs.keys()

    # Initialize vars
    # out      = out_opt
    # out_df   = out_df_opt
    out_meta = out_meta_opt
    ret      = ret_opt
    ret_out  = ret_out_opt

    # ret implies no out, unless out is set specifically
    if ('ret' in kwargs.keys() or 'ret_out' in kwargs.keys()) and not outargset and (ret or ret_out):
        # If ret is set to True by the user we will not provide stdout output,
        # unless the user has specifically set out = True
        # out      = False # Do not output to the screen at all
        # out_df   = False # Do not output to the screen the DF, only the headers
        out_meta = False # Do not output to the screen the Metadata

    if 'out' in kwargs.keys():
        if out_opt == False:
            # out_df   = False
            out_meta = False

    # if 'out_df' in kwargs.keys():
    #     if out_df_opt :
    #         out    = True
    #         out_df = True

    if 'out_meta' in kwargs.keys():
        if out_meta_opt :
            # out      = True
            out_meta = True

    if 'ret_out' in kwargs.keys():
        if ret_out :
            ret = True

    # Artifact-specific argument parsing ---------------------------------------

    if isinstance(args[0], str):
        evtidspar = args[0]
    else:
        if len(args[0]) == 0:
            evtidspar = ""
        else:
            evtidspar = args[0]
            if len(evtidspar) == 0:
                evtidspar = ""
            else:
                evtidspar = str(evtidspar[0])


    # Variables ----------------------------------------------------------------
    hiddencols =  []

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 20

    # Pre-Processing ----------------------------------------------------------

    # Identify evtx evtx FileName
    evtxfname=df.iloc[0]['evtxFileName_']

    # If specific evtids are defined, filter for them
    if evtidspar != "":
        if "," in evtidspar:
            evtidssr = pd.Series(evtidspar.split(",")).astype(int)
        else:
            evtidssr = pd.Series(int(evtidspar))

        if df.index.name != 'Timestamp':
            if "System > EventID" in df.columns:
                df1 = df.query('`System > EventID` in @evtidssr', engine="python")
            else:
                df1 = pd.DataFrame([])

            if "System > EventID > #text" in df.columns:
                df2 = df[df['System > EventID > #text'].isin(evtidssr)]
            else:
                df2 = pd.DataFrame([])

            df = pd.concat([df1,df2]).dropna(how='all')

    # If specific evtids (either by numbers or macro) are defined, filter for them
    if evtidspar == "help" or evtidspar == "list":
        macrosdf = pd.DataFrame.from_dict(evtid_groups[evtxfname], orient='index').fillna("-")
        display(macrosdf)
        return
    elif evtidspar == "":
        evtidssr = df['EventID_'].drop_duplicates()
    elif re.search(r'^[0-9,\ ]*$', evtidspar):
        evtids = evtidspar
        if "," in evtids:
            evtidssr = pd.Series(evtids.split(",")).astype(int)
        else:
            evtidssr = pd.Series(int(evtids))
        df = df.query('EventID_ in @evtidssr', engine="python").dropna(axis=1,how='all')
    elif re.search('[0-9A-Za-z_]*$', evtidspar):
        if evtidspar in  evtid_groups[evtxfname]:
            evtidssr = pd.Series(list(evtid_groups[evtxfname][evtidspar]))
            df = df.query('EventID_ in @evtidssr', engine="python").dropna(axis=1,how='all')
        else:
            print('ERROR: EvtID Group not found: ' + evtidspar)
            return
    else:
        print("ERROR: Invalid evtid expression: "+evtids)
        return

    # Knowledge Enrichment - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if evtxfname in evtid_col_map:
        for evtid in evtidssr:
            if evtid in evtid_col_map[evtxfname]:
                for col in df.columns:
                    if col in evtid_col_map[evtxfname][evtid]:
                        newcol=col+"_K_"
                        colloc = df.columns.get_loc(col)
                        newcolloc = colloc + 1
                        if newcol not in df.columns:
                            df.insert(newcolloc,newcol,"-")
                            df[newcol] = df[col].map(evtid_col_map[evtxfname][evtid][col])

    # evtid-specific Filters  - - - - - - - - - - - - - - - - - - - - - - - - -
    if apply_filters :
        filtersdf = pd.DataFrame([], columns=['LogFile', 'EvtID', 'Filter', 'Expresion'])
        if evtxfname in evtx_simple_filters:
            for evtid in evtidssr:
                if evtid in evtx_simple_filters[evtxfname]:
                    df_noevtid = df[df['EventID_'] != evtid]
                    df_evtid   = df[df['EventID_'] == evtid]

                    for flt in evtx_simple_filters[evtxfname][evtid]:
                        fltexpr = evtx_simple_filters[evtxfname][evtid][flt]
                        fltexprb = re.sub(r"\^","\\^\\\\",fltexpr)
                        fltexprb = fltexprb.replace("$", r"\$")
                        row = pd.DataFrame([[evtxfname, evtid, flt, fltexprb]], columns=['LogFile', 'EvtID', 'Filter', 'Expresion'])
                        filtersdf = filtersdf.append(row, ignore_index=True)
                        #print(filter.values())
                        df_evtid = df_evtid.query(fltexpr,engine="python")

                    df = df_noevtid.append(df_evtid, ignore_index=True)


    if apply_filters  and out_meta:
        if len(filtersdf) > 0:
            display(Markdown("**Filters Applied:**"))
            display(filtersdf)

    # evtid Statistics / Descriptions - - - - - - - - - - - - - - - - - - - - -
    if len(evtidssr) > 1 and out_meta:
        display(Markdown("**EventID Statistics / Descriptions:**\n<br>"))
        statsdf = pd.DataFrame(df['EventID_'].value_counts()).reset_index().rename(columns={'EventID_': 'Count', 'index': 'EventID_'})
        if evtxfname in evtid_desc.keys():
            evtiddscstr=""
            for evtid in statsdf['EventID_']:
                if evtid in evtid_desc[evtxfname].keys():                    
                    evtiddesc = evtid_desc[evtxfname][evtid]
                    statsdf.loc[(statsdf.EventID_ == evtid),'Description']= evtiddesc
                    evtidblt = "\n  * "+str(evtid)+": "+str(evtiddesc)
                    evtiddscstr = evtiddscstr + evtidblt
                    # anydescs = True
                # else:
                #     missingdescs = True
        display(statsdf)

    # Call to simple_common ----------------------------------------------------
    dfout = d4com.simple_common(df, *args, **kwargs, hiddencols=hiddencols, maxdfbprintlines=maxdfbprintlines)

    # Post-Processing ---------------------------------------------------------

    # Return ------------------------------------------------------------------
    return dfout


def get_source_options():
    return ['consolidate_cols', 'apply_filters', 'beautify_cols']

# MACHINE LEARNING ============================================================

def find_anomalies_evtx(indf):
    if 'D4_DataType_' in indf.columns:
        if indf['evtxFileName_'][0]=='Microsoft-Windows-TaskScheduler%4Operational.evtx':
            hml_df = convert_scheduled_tasks_ham_to_hml(indf)
            return hml_df
        elif indf['D4_DataType_'][0]=='evtx-hml':
            hml_df = indf
        else:
            print("Event type not supported")
            return None, None
    
        
def convert_scheduled_tasks_ham_to_hml(evtx_ham_df):
    
    if evtx_ham_df['evtxFileName_'][0]!='Microsoft-Windows-TaskScheduler%4Operational.evtx':
        return None
    
    hml_df = evtx_ham_df[['EventID_', 'Computer', '@Name', 'TaskName',  '@UserID', 'UserName', 'UserContext', 'ResultCode', 'ActionName']]
    hml_df.reset_index(inplace=True)
    # Rename Columns
    hml_df = hml_df.rename(columns={"Timestamp":   'Timestamp_'})
    hml_df = hml_df.rename(columns={'Computer':    'Computer_'})
    hml_df = hml_df.rename(columns={'@Name':       'AtName_'})
    hml_df = hml_df.rename(columns={'TaskName':    'TaskName_'})
    hml_df = hml_df.rename(columns={'UserName':    'UserName_'})
    hml_df = hml_df.rename(columns={'UserContext': 'UserContext_'})
    hml_df = hml_df.rename(columns={'@UserID':     'AtUserID_'})
    hml_df = hml_df.rename(columns={'ActionName':  'ActionName_'})
    hml_df = hml_df.rename(columns={'ResultCode':  'ResultCode_'})

    # Computer_ -----------------------------------------------------------------------------------
    hml_df['Computer_'] = hml_df['Computer_'].str.lower()

    # UserName / UserContext ----------------------------------------------------------------------
    hml_df['UserName_']    = hml_df['UserName_'].str.lower()
    hml_df['UserContext_'] = hml_df['UserContext_'].str.lower()

    # Combine UserName + UserContext Cols
    hml_df['UserNC_'] = hml_df['UserName_']
    hml_df['UserNC_'] = hml_df['UserNC_'].fillna(hml_df['UserContext_'])
    hml_df = hml_df.drop(columns=['UserName_', 'UserContext_'])

    # Add Domain d4_null to entries with no domain
    hml_df['UserNC_'] = hml_df['UserNC_'].str.replace('^([^\\\\]*)$', 'd4_null\\\\\\1')

    # Fill NaNs with default values
    hml_df['UserNC_']       = hml_df['UserNC_'].fillna('d4_null')

    # Hostname_ -----------------------------------------------------------------------------------
    # This column is needed to merge tskevtx + tskflist. We will drop it when running the model 
    hml_df['Hostname_'] = hml_df['Computer_'].str.replace('\..*','')
    hml_df = hml_df.drop(columns=['Computer_'])
    # Fill NaNs with default values ================================================================
    hml_df['ResultCode_']   = hml_df['ResultCode_'].fillna("-64646464")
    hml_df['ActionName_']   = hml_df['ActionName_'].fillna('d4_null')

    hml_df['D4_DataType_'] = 'evtx-hml'

    hml_df = hml_df.reset_index(drop=True)
    return hml_df

# ACCESSOR ====================================================================

@pd.api.extensions.register_dataframe_accessor("d4evtx")
class Ds4n6evtxAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def column_types_set(self):
        df = self._obj
        return column_types_set_func(df)

    def consolidate_columns(self):
        df=self._obj
        return consolidate_columns_func(df)

    def simple(self, *args, **kwargs):
        obj=self._obj
        return simple_func(obj, args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("d4_evtx")
class Ds4n6_evtxAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def column_types_set(self):
        df = self._obj
        return column_types_set_func(df)

    def consolidate_columns(self):
        df=self._obj
        return consolidate_columns_func(df)

    def simple(self, *args, **kwargs):
        obj=self._obj
        return simple_func(obj, args, **kwargs)
