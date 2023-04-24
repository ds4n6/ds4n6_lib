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
import ds4n6_lib.ml     as d4ml
import ds4n6_lib.utils  as d4utl
from ds4n6_lib.knowledge import evtid_desc, evtid_col_map, evtid_col_map_desc, evtx_simple_filters, evtid_groups, windows_builtin_accounts_regex, evilsigs, hidden_cols_evtx

###############################################################################
# VARIABLES
###############################################################################
# Lateral Movement Files
evtx_lm_files = ['Security.evtx', 'Microsoft-Windows-RemoteDesktopServices-rdpcorets%4operational.evtx', 'Microsoft-Windows-TerminalServices-RemoteConnectionManager%4Operational.evtx', 'Microsoft-Windows-TerminalServices-LocalSessionManager%4Operational.evtx', 'Microsoft-Windows-TerminalServices-RDPClient%4operational.evtx', 'Microsoft-Windows-SMBClient%4Security.evtx', 'Microsoft-Windows-SMBServer%4Security.evtx']


###############################################################################
# FUNCTIONS
###############################################################################


# Convert XML into DF(s) ======================================================

def evtid_dfs_build_raw(df):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Input  (df) is a raw evtx DF
    # Output (df) is a dict of raw evtx DFs (one per eventid)

    dfs = {}

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

def evtid_dfs_build(df, **kwargs):
    """ 
    Input:
        df: (df) is a raw evtx DF
    Return:
        return (df) is a dict of evtx DFs (one per eventid)
    """       
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    evtids2parse = kwargs.get('evtids2parse',       [])
    verbose      = kwargs.get('verbose',             0)

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

    if evtids2parse != []:
        evtids2process = evtids2parse
    else:
        evtids2process = evtids

    if verbose: print("- Parsing event IDs: ", end='')

    for evtid in evtids2process:
        if verbose: print(str(evtid)+' ', end='')

        thisdf = df.query('EventID_ == @evtid', engine="python")
        if len(thisdf) == 0:
            print("- No " + str(evtid) + " events found. Skipping.")
            continue
        thisdf = thisdf.d4evtx.consolidate_columns()
        thisdf = thisdf.dropna(axis=1, how='all')

        # Harmonize ========================================================
        thisdf = harmonize(thisdf, orchestrator=orchestrator, tool=tool, plugin=plugin, hostname=hostname)

        # Add df to dfs / alldf
        dfs[evtid] = thisdf
        alldf = pd.concat([alldf,thisdf])

    print("")
    
    if len(dfs.keys()) == 0:
        print("- WARNING: No events found. Returning an empty dictionary.")
        return dfs
    else:
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
# TODO: DEPRECATED. REMOVE.
#     def find_anomalies_local(mlfulldf, mltraindf, mlpredsdf, odalg="simple_autoencoder", fitloops=1, epochs=40, loss_threshold=None, verbose=1, kerasverbose=1):
#         # TODO: This local function is to be replaced by the more generic find_anomalies
#         #       When that happens, this function needs to be removed from here
#     
#         # OD Algorithm-specific Data Preparation  - - - - - - - - - - - - - - - - -
#         if verbose != 0:
#             print("  + Performing Outlier Detection Algorithm Data Preparation for " + odalg + "..." )
#     
#         if odalg == "simple_autoencoder":
#             # If we use a simple autoencoder 
#             # Drop duplicates
#             print('    - Dropping Duplicates')
#             mlfulldf  = mlfulldf.drop_duplicates()
#             mltraindf = mltraindf.drop_duplicates()
#             mlpredsdf = mlpredsdf.drop_duplicates()
#     
#         # Keep the orginal DFs
#         mlfullorigdf  = mlfulldf
#         mltrainorigdf = mltraindf
#         mlpredsorigdf = mlpredsdf
#     
#         # DATA TRANSFORM (Categorical, Normalization, ...)
#         if verbose != 0:
#             print("  + Building Transform dictionary: ", end='')
#     
#         transform_dict = {}
#         for col in mlfulldf.columns:
#             if verbose != 0:
#                 print(col + " ", end='')
#             cats = pd.Categorical(mlfulldf[col]).categories
#             d = {}
#             for i, cat in enumerate(cats):
#                 d[cat] = i
#                 transform_dict[col] = d
#         if verbose != 0:
#             print("")
#         
#         if verbose != 0:
#             print("  + Building Inverse Transform dictionary: ", end='')
#         inverse_transform_dict = {}
#         for col, d in transform_dict.items():
#             if verbose != 0:
#                 print(col + " ", end='')
#             inverse_transform_dict[col] = {v:k for k, v in d.items()}
#         if verbose != 0:
#             print("")
#         
#         if verbose != 0:
#             print("  + Applying Transform dictionary to DF... ")
#         mltraindf = mltraindf.replace(transform_dict)
#         mlpredsdf = mlpredsdf.replace(transform_dict)
#     
#         # Resulting Data
#         if verbose != 0:
#             print("  + Resulting ML Training    DF Shape: " + str(mltraindf.shape))
#             print(mltraindf.head(1))
#             print("  + Resulting ML Predictions DF Shape: " + str(mlpredsdf.shape))
#             print(mlpredsdf.head(1))
#             print("")
#     
#         if verbose == 0:
#             print("Done.\n")
#     
#         # SELECTING TRAINING / TEST DATA ------------------------------------------
#     
#         # N/A
#         
#         # CREATING THE NEURAL NETWORK ARCHITECTURE --------------------------------
#         models   = {}
#         mdlnames = []
#         losses   = []
#         tlcnt    = 1
#         while tlcnt <= fitloops:
#             # There are 4 different input features, and as we plan to use all the features in the simple autoencoder,
#             # we define the number of input neurons to be 4.
#             nfeatures    = mlfulldf.shape[1]
#     
#             input_dim    = mlfulldf.shape[1]
#             encoding_dim = nfeatures - 2
#     
#             if verbose != 0:
#                 print("NETWORK ARCHITECTURE:")
#                 print("- No. Features:    "+str(nfeatures))
#                 print("- Input Dimension: "+str(input_dim))
#                 print("")
#     
#             # Input Layer
#             input_layer  = Input(shape=(input_dim,))
#     
#             # We create an encoder and decoder. 
#             # The ReLU function, which is a non-linear activation function, is used in the encoder. 
#             # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern
#             encoded = Dense(encoding_dim, activation='relu')(input_layer)
#             decoded = Dense(nfeatures, activation='linear')(encoded)
#     
#             # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
#             # Next, the optimizer and loss function is defined using the compile method.
#             # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
#             # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
#             # The loss is defined as mse, which is mean squared error
#             autoencoder = Model(input_layer, decoded)
#             autoencoder.compile(optimizer='adadelta', loss='mse')
#             modelname = autoencoder.name
#     
#             # Save autoencoder name to list
#             mdlnames.append(autoencoder.name)
#             # Save autoencoder model to dict
#             models[modelname] = autoencoder
#     
#             print("Autoencoder Summary:")
#             print("")
#             autoencoder.summary()
#             print("")
#     
#             # TRAINING THE NETWORK --------------------------------
#             # The training data, mldf, is fitted into the autoencoder. 
#             # Let's train our autoencoder for 100 epochs with a batch_size of 4 and observe if it reaches a stable train or test loss value
#             batch_size=4
#     
#             if verbose != 0:
#                 print("TRAINING:")
#                 print("- Configuration:")
#                 print("  + epochs     = "+str(epochs))
#                 print("  + batch_size = "+str(batch_size))
#                 print("  + fitloops   = "+str(fitloops))
#                 print("")
#             else:
#                 print("TRAINING... ", end='')
#     
#             # Here we convert the DF to a nunpy array
#             aeinarr = np.array(mltraindf).astype('float32') 
#             if verbose != 0:
#                 print("- Fitting model... ", end='')
#     
#             print("")
#             print("[LOOP ITERATION: "+str(tlcnt)+"/"+str(fitloops)+"]")
#             print("")
#     
#             losshist = autoencoder.fit(aeinarr, aeinarr, epochs=epochs, batch_size=batch_size, verbose=kerasverbose)
#             losses.append(losshist.history['loss'][-1])
#             tlcnt += 1
#     
#         minloss        = min(losses)
#         minlossidx     = losses.index(minloss)
#         minlossmdlname = mdlnames[minlossidx]
#         minlossmodel   = models[minlossmdlname]
#     
#         if verbose != 0:
#             print("")
#             print("- Models:          "+str(mdlnames))
#             print("- Losses:          "+str(losses))
#             print("- Min. Loss:       "+str(minloss))
#             print("- Min. Loss Model: "+str(minlossmdlname))
#             print("")
#     
#         # TODO: Save the Min. Loss info to file, for future reference
#         #       - Maybe not needed... can you obtain the loss from the model file?
#     
#         # TODO: Save model
#         #       - We will also have to modify the code above so the training will
#         #         not happen if there is saved model, and the model will be loaded instead
#         #
#         #   # Save model, if model file does not exist
#         #   if autosave_minloss_model:
#         #       if autosave_minloss_model:
#         #           minlossautoencoder = minlossmodel
#         #           if model_filename_root is not None:
#         #               model_filename_root = model_filename_root
#         #           else:
#         #               model_filename_root = ""
#     
#         #           model_filename_save = model_filename_root+"-"+model_type+"-"+runid+"-loss_"+str(minloss)+".h5"
#     
#         #           if not os.path.exists(model_filename_save):
#         #               print("- Saving Model to:")
#         #               print("    "+model_filename_save)
#         #               print("")
#         #               autoencoder.save(model_filename_save)
#         #       else:
#         #           if not os.path.exists(model_filename):
#         #               print("- Saving Model to:")
#         #               print("    "+model_filename)
#         #               autoencoder.save(model_filename)
#         #               print("")
#     
#         # DOING PREDICTIONS -----------------------------------
#     
#         # Once the model is fitted, we predict the input values by passing the same mldf dataset to the autoencoder's predict method.
#         # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:
#     
#         print("PREDICTIONS:")
#         # Here we convert the DF to a nunpy array
#         aeinarr = np.array(mlpredsdf).astype('float32') 
#     
#         predsraw = autoencoder.predict(aeinarr)
#         mse = np.mean(np.power(aeinarr - predsraw, 2), axis=1)
#     
#         # Un-normalize predictions (predraw) and DF-ize all
#         # WARNING: mldf is actually not a DF, it's a numpy.ndarray
#         # TODO: We should change mldf to something else. It's confusing
#         predsraw = aeinarr
#     
#         predsdf = pd.DataFrame(predsraw)
#         predsdf.columns = mlpredsdf.columns
#         predsdf = predsdf.replace(inverse_transform_dict)
#     
#         for x in predsdf.columns:
#             predsdf[x]=predsdf[x].astype(mlpredsorigdf[x].dtypes.name)
#     
#         msedf = pd.DataFrame(mse)
#     
#         # Auto-calculate loss_threshold - Top 5 values
#         if loss_threshold is None:
#            loss_threshold = int(np.sort(mse)[-15])
#            autocalcmsg = " (Auto-calculated)"
#         else:
#            autocalcmsg = ""
#     
#         print("- MSE Threshold: "+str(loss_threshold)+autocalcmsg)
#     
#         # Select entries above the loss threshold
#         xxx = aeinarr[mse >= loss_threshold]
#     
#         xxxdf = pd.DataFrame(xxx)
#         xxxdf.columns = mlfulldf.columns
#         anom = xxxdf.replace(inverse_transform_dict)
#         print("- No. Anomalies - Total:    "+str(len(xxxdf)))
#         print("- No. Anomalies - Red Zone: "+str(len(xxxdf)))
#     
#         return anom, predsdf, msedf
#     
#     def host_user_access_analysis(indf, **kwargs): 
#     
#         # Arguments Parsing -------------------------------------------------------
#         redzone     = kwargs.get('redzone',       None)
#         odanalysis  = kwargs.get('odanalysis',  True)
#     
#         # Input Data & Arguments Validation - - - - - - - - - - - - - - - - - - - -
#     
#         # indf  · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
#         # TODO: Verify that indf contains all the required columns
#         
#         # Verify that indf contains just 1 Computer
#         comps  = indf['Computer_'].drop_duplicates().reset_index(drop=True)
#         ncomps = indf['Computer_'].nunique()
#     
#         opdf = indf.query('EventID_ == 4624')
#     
#         # TODO: This code should be a standalone function
#         if ncomps > 2:
#             print("- Multiple Computer_ values on input DF. Aborting.")
#             display(str(comps))
#             return
#         elif ncomps == 2:
#             comp1 = comps.loc[0]
#             comp2 = comps.loc[1]
#             comp1base = re.sub("\..*$", "", comp1)
#             comp2base = re.sub("\..*$", "", comp2)
#             if comp1base == comp2base:
#                 if len(comp1) > len(comp2):
#                     compname = comp1
#                 else:
#                     compname = comp2
#                 print("- Computer Name: " + compname)
#                 opdf['Computer_'] = compname
#             else:
#                 print("- 2 Computer_ values on input DF. Different base name. Aborting.")
#                 display(comps)
#                 return
#     
#         # redzone · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
#         # TODO: This validation should be a function in d4.common (rz_arg_healthcheck)
#         if redzone is not None:
#             if type(redzone) != list:
#                 print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
#                 return
#             else:
#                 if len(redzone) != 2:
#                     print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
#                     return
#                 else:
#                     rzmin = redzone.loc[0]
#                     rzmax = redzone.loc[1]
#     
#         # Common Analysis ---------------------------------------------------------
#     
#         # Identify accounts that have never logged in before the Red Zone
#         # We will use event Security:4624
#     
#         # Red Zone 
#         # TODO: To Be Completed
#         # if redzone[0] is not None:
#         #     tmpdf = hostsbmldf[['Timestamp_','TargetUserName_']].drop_duplicates()
#         #     tmpdf[tmpdf['Timestamp_'] >= redzone[0]]
#         #     rzdfidx = tmpdf[tmpdf['Timestamp_'] >= redzone[0]].index
#         #     hostsbmldf.loc[rzdfidx]
#         #
#         # if redzone[0] is not None:
#         #     ...
#         #
#         # e4624dfrzdf =
#         
#         # Outlier / Anomaly Analysis ----------------------------------------------
#         if odanalysis == True:
#             ml_access_anomalies(indf, **kwargs)

def ml_access_anomalies(indf, **kwargs): 
    # If you provide the redzone arg the training will be carried out with the
    # data before the redzone and the predictions will be made in the redzone

    # Configuration 
    supportedodalgs = ['simple_autoencoder', 'lstm_autoencoder']

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    argtsmin       = kwargs.get('argtsmin',      None)
    argtsmax       = kwargs.get('argtsmax',      None)
    loss_threshold = kwargs.get('loss_threshold', None)
    epochs         = kwargs.get('epochs',        40)
    excsysusers    = kwargs.get('excsysusers',   False)                  # Exclude System Users (non S-1-5-21-)
    exchostusers   = kwargs.get('exchostusers',  False)                  # Exclude Hostname Users (hostname$)
    kerasverbose   = kwargs.get('kerasverbose',  2)
    verbose        = kwargs.get('verbose',       1)
    redzone        = kwargs.get('redzone',       None)
    odalg          = kwargs.get('odalg',         "simple_autoencoder")   # Outlier Detection Algorithm
    excusers       = kwargs.get('excusers',      None)                   # Excluded Users (to avoid false positives)
    # Data Preparation 
    ipaddrasnum    = kwargs.get('ipaddrasnum',   True)                   # Convert IP Address to Integers to improve ML accuracy
    # Anomaly threshold
    ntop           = kwargs.get('ntop',          20)                     # No. Top anomalies to consider
    # Show output
    showoutputs    = kwargs.get('showoutputs',   False)                  # Enable/disable output to screen,
                                                                         # so this function can be used in jupyter/ipython or 
                                                                         # be called as a function from another python script
    display        = kwargs.get('display',       False)                  # Display (for Jupyter notebooks)
    # Plots
    plots          = kwargs.get('lossplot',      True)                   # Generate plots
    lossplot       = kwargs.get('lossplot',      True)                   # Generate loss plot
    showplots      = kwargs.get('showplots',     True)                   # Show plots
    saveplots      = kwargs.get('saveplots',     True)                   # Save plots
    plotsavepath   = kwargs.get('plotsavepath',  '')                     # plot save path
    # ML Feature Selection / Engineering
    usedow         = kwargs.get('usedow',        False)                  # Use Day of the Week as a ML Feature
    usetimeofday   = kwargs.get('usetimeofday',  False)                  # Use Time of the Day (4h intervals) as a ML Feature
    # ML Parameters
    epochs         = kwargs.get('epochs',        40)
    fitloops       = kwargs.get('fitloops',       3)
    lstm_timesteps = kwargs.get('lstm_timesteps', 200)

    # Arguments Validation  - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # redzone
    # TODO: This validation should be a function in d4.common (rz_arg_healthcheck)
    if redzone is not None:
        if type(redzone) != list:
            print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
            redzone = None
        else:
            if len(redzone) != 2:
                print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
                redzone = None

    # Arguments Processing  - - - - - - - - - - - - - - - - - - - - - - - - - - 

    if verbose == 0 and 'kerasverbose' not in kwargs.keys():
        kerasverbose=0

    if verbose == 0 and 'lossplot' not in kwargs.keys():
        lossplot = False

    # If plots is False we disable lossplot
    if plots == False and 'lossplot' not in kwargs.keys():
        lossplot = False

    # NOTE: verbose > 0 for INFO prints
    if verbose:
        print("- INFO:")
        print("  + lossplot:      " + str(lossplot))
        print("  + plots:        " + str(plots))
        print("  + showplots:    " + str(showplots))
        print("  + saveplots:    " + str(saveplots))
        print("  + plotsavepath: " + str(plotsavepath))

    # HAM -> HML CONVERSION ---------------------------------------------------
    indf = convert_logons_ham_to_hml(indf, verbose)
    
    # tsmin/tsmax · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
    dftsmin = indf['Timestamp_'].min()
    dftsmax = indf['Timestamp_'].max()

    if argtsmin is None:
        tsmin=dftsmin
    else:
        tsmin=argtsmin

    if argtsmax is None:
        tsmax=dftsmax
    else:
        tsmax=argtsmax

    # redzone · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
    if redzone is not None:
        rzmin = redzone[0]
        rzmax = redzone[1]

    # Arguments Supported Options - - - - - - - - - - - - - - - - - - - - - - - 
    if odalg not in supportedodalgs:
        print("ERROR: Unsupported Anomaly Detection Algorithm: " + str(odalg))
        return

    # HEALTH CHECKS -----------------------------------------------------------

    # [HC] Verify that the fields that we need are present in the df
    # It is possible to run this function on event logs that may have been 
    # obtained from other means other than the original evtx. We will allow that,
    # but we need to ensure that the fields required to make this function work are
    # present
    if 'SourceIP_' in indf.columns:
        if verbose:
            print("INFO: Renaming SourceIP column -> SourceIP_")
        indf = indf.rename(columns={"SourceIP": "SourceIP_"})

    reqcols = ['Timestamp_','Computer_','EventID_','WorkstationName_','SourceIP_','TargetUserName_','LogonType_']
    optcols = [] # Not used currently

    missingcols = ""
    for reqcol in reqcols:
        if reqcol not in indf.columns:
            missingcols = missingcols + " " + reqcol
    if missingcols != "":
        print("ERROR: Missing columns on input DF: " + str(missingcols))
        return 

    # DEFINITIONS ------------------------------------------------------------

    # Columns of Interest
    # - CoIs that must always be present
    colsoibase = ['Timestamp_', 'EventID_', 'Computer_', 'TargetUserName_', 'WorkstationName_', 'SourceIP_', 'LogonType_']
    # - Additional CoIs that may or may not be present
    colsoiopt  = ['LogoutTimestamp_', 'SessionLength__']

    # START ------------------------------------------------------------------

    # Round TS to seconds
    #indf['Timestamp_'] = indf['Timestamp_'].dt.ceil('s')

    evts4624 = indf.query('EventID_ == 4624', engine="python")

    np.random.seed(8)

    evts4624_final = evts4624

    if excsysusers == True:
        if 'TargetUserSid_' in evts4624.keys():
            evts4624_final = evts4624_final[evts4624_final['TargetUserSid_'].str.contains('S-1-5-21-')].reset_index()

     
    if exchostusers == True:
        if 'TargetUserName_' in evts4624.columns:
            print("- WARNING: Excluding Host Users (hostname$). This could be exploited by attackers.")
            evts4624_final = evts4624_final.query('~TargetUserName_.str.contains("\$$")', engine="python").reset_index()

    # Merge Columns of Interest Base + Optional
    if verbose:
        print("- Adding optional columns to Columns of Interest:")
    colsoi = colsoibase
    for col in colsoiopt:
        if col in evts4624_final:
            if verbose: print("  + " + col)
            colsoi = colsoi + [col]

    if verbose: print("- Columns of Interest: ", end='')
    if verbose: print(colsoi)

    # TODO: LEGACY. REMOVE.
    #tuwilcols = ['Timestamp_', 'TargetUserName_', "WorkstationName_", "SourceIP_", 'LogonType_']
    #uwilcols  = ['TargetUserName_', 'WorkstationName_', 'SourceIP_', 'LogonType_']

    tuwilcols = colsoi.copy()
    tuwilcols.remove('EventID_')
    tuwilcols.remove('Computer_')
    tuwilcols.remove('LogoutTimestamp_')
        
    uwilcols = tuwilcols.copy()
    uwilcols.remove('Timestamp_')

    hostnamelst = evts4624_final['Computer_'].drop_duplicates().tolist()

    if verbose == 5:
        display(indf.head(1))
        display(evts4624_final.head(1))

    if len(hostnamelst) == 0:
       print("WARNING: No. hostnames found for this host. Aborting.")
       return
    elif len(hostnamelst) == 1:
       hostname = hostnamelst[0]
    else:
       hostname = hostnamelst[-1]
       print("WARNING: Multiple hostnames found for this host: " + str(hostnamelst))
       print("         Selecting the last one: " + str(hostname))

    if verbose:
        print("INFO:")
        print("- Hostname:        " + str(hostname))
        print("- Timestamp Range: " + str(tsmin) + " -> " + str(tsmax))
        print("- No. events:      " + str(len(evts4624_final)))
    if redzone != None:
        if verbose: print("- Red Zone:        " + str(redzone[0]) + ' -> ' + str(redzone[1]))
    print("")

    if len(evts4624_final) == 0:
        print("INFO: No events to process. Aborting.")
        return

    if odalg == "lstm_autoencoder":
        if len(evts4624_final) <= lstm_timesteps * 2:
            print("INFO: Not enough events for ML. Aborting.")
            return
    else:
        # Why 50? It seems a fair value, but this should be fine tuned.
        if len(evts4624_final) <= 50:
            print("INFO: Not enough events for ML. Aborting.")
            return

    # DATA PREPARATION --------------------------------------------------------
    if verbose:
        print("DATA PREPARATION:")

    useraccess = evts4624_final[colsoi]
  
    if verbose: print(useraccess.query('TargetUserName_.str.contains("\$$")', engine="python").head(10))

    this_useraccess = useraccess[useraccess['Timestamp_'] >= tsmin]
    this_useraccess = this_useraccess[useraccess['Timestamp_'] <= tsmax]
    
    user_access_tuwil = this_useraccess[tuwilcols].copy()
    
    # Lower-case cols
    if verbose != 0:
        print("  + Columns: ", end='')
        print("WorkstationName_ ", end='')
    user_access_tuwil['WorkstationName_'] = user_access_tuwil['WorkstationName_'].str.lower().fillna("null_workstation")
    if verbose != 0:
        print("TargetUserName_ ", end='')
    user_access_tuwil['TargetUserName_']  = user_access_tuwil['TargetUserName_'].str.lower()
    if verbose != 0:
        print("LogonType_ ", end='')
    user_access_tuwil['LogonType_']       = user_access_tuwil['LogonType_'].astype('string')
    if verbose != 0:
        print("")
    
    # Timestamp_ Feature Engineering  - - - - - - - - - - - - - - - - - - - - -
    if usedow == True:
        user_access_tuwil.insert(1,'TSDoW_',0)
        user_access_tuwil['TSDoW_'] = user_access_tuwil['Timestamp_'].dt.dayofweek
    if usetimeofday == True:
        user_access_tuwil.insert(2,'TS4H',0)
        user_access_tuwil['TS4H'] = user_access_tuwil['Timestamp_'].dt.ceil('H').dt.hour.div(6).round(0).astype(int)

    # Convert IP Addresses from octets to integers. This should improve ML accuracy.
    # TODO: This code is disabled. This code works, but the one that reverses the process does not. 
    ipaddrasnum = False      
    if ipaddrasnum == True:
        if verbose:
            print("- Converting IP Addresses to Integer")

        import ipaddress

        # Some entries are "-". We will replace them by "0.0.0.1" 
        # (which is an invalid IP Address and therefore never appear in our data)
        user_access_tuwil['SourceIP_'] = user_access_tuwil['SourceIP_'].str.replace('-','0.0.1.0')

        # Convert IP addresses to integer
        ipv6mask = user_access_tuwil['SourceIP_'].str.contains(":")
        user_access_tuwil['SourceIPType_'] = ''  # We create a new column to "remember" the IP Address type 
                                                 # so we can reverse the process
        # - IPv4
        user_access_tuwil['SourceIP_'][~ipv6mask] = user_access_tuwil['SourceIP_'][~ipv6mask].apply(lambda x: str(int(ipaddress.IPv4Address(x))))
        user_access_tuwil['SourceIPType_'][~ipv6mask] = "IPv4"
        # - IPv6
        user_access_tuwil['SourceIP_'][ipv6mask] = user_access_tuwil['SourceIP_'][ipv6mask].apply(lambda x: str(int(ipaddress.IPv6Address(x))))
        user_access_tuwil['SourceIP_'].astype(int)

    # user_access_uwil_str -> Compact string version of user_access_uwil
    if verbose != 0:
        print("  + Building uwil string df...")

    user_access_uwil_str = user_access_tuwil.copy()
    user_access_uwil_str['TU-WN-IP-LT'] = "[" + user_access_uwil_str['TargetUserName_'] + "]" + "[" + user_access_uwil_str['SourceIP_'] + "][" + user_access_uwil_str['LogonType_'] + "]"
    user_access_uwil_str.drop(columns=uwilcols, inplace=True)
    user_access_uwil_str = user_access_uwil_str.sort_values(by='TU-WN-IP-LT')

    mlfulltsdf = user_access_tuwil.copy()

    # Lowercase everything
    for column in mlfulltsdf.columns:
        if type(mlfulltsdf[column]) == "str" or type(mlfulltsdf[column]) == "string":
            mlfulltsdf[column] = mlfulltsdf[column].str.lower() 

    # Exclusions (to avoid FPs) - - - - - - - - - - - - - - - - - - - - - - - -
    # Excluded Users
    if excusers != None:
        print('  + Excluding Users: ' + str(excusers))
        mlfulltsdf = mlfulltsdf.query('TargetUserName_ not in @excusers')

    # TODO: Other exclusions (IPs, ...)

    # Red Zone  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TODO: Verify if there actually is data in the Benchmark Zone (< RZ).
    #       If there is no data then we shouldn't divide in train/preds, but
    #       act as if there was no redzone
    #       (WARNING message: "redzone define but no data outside the redzone. Ignoring redzone.)
    if redzone is not None:
        mltraintsdf = mlfulltsdf.query('Timestamp_ < @rzmin').reset_index(drop=True)
        mlpredstsdf = mlfulltsdf.query('Timestamp_ >= @rzmin & Timestamp_ <= @rzmax').reset_index(drop=True)
        if len(mltraintsdf) <= lstm_timesteps * 2:
            print("WARNING: Not enough training data (" + str(len(mltraintsdf)) + ").")
            print("         Training will be done with the full dataset (train+pred).")
            print("         Results will probably be less accurate.")
            print("")
            mltraintsdf = mlfulltsdf
            mlpredstsdf = mlfulltsdf
    else:
        mltraintsdf = mlfulltsdf
        mlpredstsdf = mlfulltsdf
    
    for col in ['Timestamp_', 'FTNS-SourceHostname__']:
        if col in mlfulltsdf.columns:
            mlfulldf  = mlfulltsdf.drop(columns= [col])
            mltraindf = mltraintsdf.drop(columns=[col])
            mlpredsdf = mlpredstsdf.drop(columns=[col])

    if verbose:
        print("- Calling anomaly detection algorithm with:")
        print("  + traindf: " + str(mltraindf.shape) + " -> " + str(mltraintsdf['Timestamp_'].min()) + " - " + str(mltraintsdf['Timestamp_'].max()))
        print("  + predsdf: " + str(mlpredsdf.shape)  + " -> " + str(mlpredstsdf['Timestamp_'].min()) + " - " + str(mlpredstsdf['Timestamp_'].max()))
        print("")

    # TODO: WorkstationName_ is dropped because it's unreliable.
    predslossdf = d4ml.ml_model_execution(traindf=mltraindf, preddf=mlpredsdf, model_type=odalg, fitloops=fitloops, epochss=[epochs], lstm_timesteps=lstm_timesteps, error_threshold=loss_threshold, verbose=verbose, kerasverbose=kerasverbose, cols2drop=[])

    if predslossdf is None:
        print("ERROR: No ML data returned. Something went wrong when calling: d4ml.ml_model_execution.")
        return
    
    lossdf = predslossdf['Loss_']

    # Convert IP Addresses back from integers to octets. This was done to improve ML accuracy.
    # TODO: This code should work, but the SourceIPType_ column is lost when we call d4ml.ml_model_execution
    #       We need to modify that function so the cols2drop are restored before returning anomlossdf
    if ipaddrasnum == True:
        if verbose:
            print("- Converting IP Addresses to Integer")
        import ipaddress
        # Some entries are "-". We will replace them by "0.0.0.1" 
        # (which is an invalid IP Address and therefore never appear in our data)
        predslossdf['SourceIP_'] = predslossdf['SourceIP_'].str.replace('-','0.0.1.0')
        # Convert IP addresses to integer
        # - IPv4
        mask = ~predslossdf['SourceIPType_'].str.contains("IPv4")
        predslossdf['SourceIP_'][mask] = predslossdf['SourceIP_'][mask].apply(lambda x: ipaddress.IPv4Address(x))
        # - IPv6
        mask = predslossdf['SourceIPType_'].str.contains("IPv6")
        predslossdf['SourceIP_'][mask] = predslossdf['SourceIP_'][mask].apply(lambda x: ipaddress.IPv6Address(x))
        predslossdf['SourceIP_'].astype(int)

    if odalg == "lstm_autoencoder":
        # If we use a lstm autoencoder we will add the Timestamp col
        predslossextdf = predslossdf.copy()
        predslossextdf.insert(0,'Timestamp_',0)
        predslossextdf['Timestamp_'] = predslossextdf['Timestamp_'].astype('datetime64[ns]') 
        predslossextdf['Timestamp_'] = mlpredstsdf.loc[lstm_timesteps-1:]['Timestamp_']
    else:
        predslossextdf = predslossdf

    anomdf = predslossextdf.copy().sort_values(by="Loss_", ascending=False)
    for col in anomdf.columns:
        if re.search("^Loss", col):
            anomdf = anomdf.drop(columns=col)

    ###############################################################################
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    
    # Extract DFs of interest from anomdf dict
    anomuniqdf = anomdf.copy()
    if 'Timestamp_' in anomuniqdf.columns:
        anomuniqdf = anomuniqdf.drop(columns=['Timestamp_'])

    # Anomalies Uniq (ts not taken into account, but without losing the sequence)
    anomuniqdf = anomuniqdf.drop_duplicates().reset_index().rename(columns={'index': 'OrigIndex'})
    topanomuniqdf = anomuniqdf.head(ntop)
    last_topanomuniqdf_index = int(topanomuniqdf.tail(1)['OrigIndex'])

    if exchostusers == True:
        if verbose:
            print("INFO: Excluding Host Users (hostname$) (exchostusers = True).\n\n")
        topanomdf = anomdf.query('~TargetUserName_.str.contains("\$$")', engine="python")
    else:
        topanomdf = anomdf

    # We select all the entries from the first anomaly till the <topn> anomaly, considered as <topn> "different" entries
    # This is to avoid a single entry repeated many times in the time realm consumes the <topn> positions
    # We obtain this number in "last_topanomuniqdf_index"
    topanomdf = topanomdf.loc[:last_topanomuniqdf_index]

    top_uwil_groupby_tabledf = pd.DataFrame(topanomdf.groupby(['SourceIP_', 'TargetUserName_', 'LogonType_']).size()).rename(columns={0: 'Count'})

    uwil_groupby_tabledf = pd.DataFrame(anomdf.groupby(['SourceIP_', 'TargetUserName_', 'LogonType_']).size()).rename(columns={0: 'Count'})

    if showoutputs == True:
        print('Top ' + str(ntop) + ' Anomalies - UWIL Unique')
        print(topanomuniqdf)
        print('')

        print('TOP ' + str(ntop) + ' UWIL GROUPBY TABLE')
        print(top_uwil_groupby_tabledf)
        print('')

        print('Top Anomalies derived from Top ' + str(ntop) + ' UWIL Unique Anomalies', 3)
        print(topanomdf)
        print('')

    # TODO: This is typically too big to print. We will just export it. 
    #print('FULL UWIL GROUPBY TABLE')
    #print(uwil_groupby_tabledf)
    #print('')

    # PLOTS -------------------------------------------------------------------

    # Plot Loss  - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - -
    if plots == True and lossplot == True:
        if showplots == True:
            plt.plot(lossdf)

        if saveplots == True and plotsavepath != '':
            pltf = plotsavepath + "/loss.png"
            print("- Saving MSE plot: " + pltf)
            lossdf.plot().get_figure().savefig(pltf)

    # OVERPLOT ANOMALOUS DATA OVER ORIGINAL DATA  - - - - - - - - - - - - - - -
    if plots == True:
        uwilcol='TU-WN-IP-LT'
        # topanomdf - topanomuwilcoldf
        #data = user_access_uwil_str
        #data = topanomdf.copy()
        #data = data.set_index('Timestamp_')

        #plt.figure(figsize=(20,10))
        plt.figure(figsize=(14,7))

        # TODO: This is only necessary if we don't use LSTM.
        #       BUT... if this is not LSTM, the timestamp does not make sense.
        #       Commented for now...
        #
        #    # Add Timestamp_ to original data df and set it as index in order to plot it
        #    dataidx = data.index
        #    data['Timestamp_'] = useraccess.loc[dataidx]['Timestamp_']
        #    # We will set Timestamp_ as index for plotting purposes
        #
        #    # data_tsindex = data.set_index('Timestamp_')
        #
        #    # TODO: If this is "unique", adding the Timestamp does not make any sense
        #    #       since plitting will be misleading (it will give you all ocurrences
        #    #       of that combination, not only the suspicious one.
        #    #       We should replace this by plottin the top X anomalies (without uniq'ing)
        #    # Add Timestamp_ to anomalies df and set it as index in order to plot it
        #    anomidx = top_anom_uwil_uniq_df_ts.index
        #    top_anom_uwil_uniq_df_ts['Timestamp_'] = useraccess.loc[anomidx]['Timestamp_']
        #    # We will set Timestamp_ as index for plotting purposes
        #    top_anom_uwil_uniq_df_ts_tsindex = top_anom_uwil_uniq_df_ts.set_index('Timestamp_')

        # Plot anomalous data only firest (red) - - - - - - - - - - - - - - - - - -
        frame = topanomdf.copy()
        
        # Create TU-WN-IP-LT column & set timestamp index, and drop all the other cols
        frame[uwilcol] = "["+frame['TargetUserName_']+"]["+frame['SourceIP_']+"]["+frame['LogonType_']+"]"
        frame[uwilcol] = frame[uwilcol].astype(str)
        frame = frame.set_index('Timestamp_')
        # Drop all columns except uwilcol
        for col in frame.columns:
            if col != uwilcol:
                frame = frame.drop(columns=col)

        # Get first/last timestamps
        ftsmin = frame.index.min()
        ftsmax = frame.index.max()
        if verbose: print(str(ftsmin)+' -> '+str(ftsmax))
       
        plt.title(str(hostname))
        plt.xlabel('Date')
        plt.ylabel('UWIL')
        plt.xticks(rotation=45)
        plt.grid(color='g', linestyle='-', linewidth=0.1)
        plt.tick_params(axis='y', which='major', labelsize=6)
        
        
        import datetime
        frame[uwilcol] = frame[uwilcol].astype(str)
        plt.plot(frame.index, frame[uwilcol], 'r.')
        plt.tight_layout()

        if showplots == True:
            plt.show()

        if saveplots == True and plotsavepath != '':
            pltf1 = plotsavepath + "/uwil-red.png"
            print("- Saving red (anomalies-only) plot: " + pltf1)
            plt.savefig(pltf1)

        # Plot original data (green) with overplotted anomalies (red) - - - - - - -

        # We will save 2 versions: Low-Res & Hi-Res
        for res in ['lores','hires']:
            if res == "lores":
                fgszx=14
                fgszy=7
            elif res == "hires":
                fgszx=40
                fgszy=20

            # Plot original data (green)  · · · · · · · · · · · · · · · · · · · · · · · 
            frame = mlfulltsdf.copy()
            
            # Create TU-WN-IP-LT column & set timestamp index, and drop all the other cols
            frame[uwilcol] = "["+frame['TargetUserName_']+"]["+frame['SourceIP_']+"]["+frame['LogonType_']+"]"
            frame[uwilcol] = frame[uwilcol].astype(str)
            frame = frame.set_index('Timestamp_')
            # Drop all columns except uwilcol
            for col in frame.columns:
                if col != uwilcol:
                    frame = frame.drop(columns=col)

            plt.figure(figsize=(fgszx,fgszy))  # Middle Resolution Image (40x20 is high res)
            plt.title(str(hostname))
            plt.xlabel('Date')
            plt.ylabel('UWIL')
            plt.xticks(rotation=45)
            plt.tick_params(axis='y', which='major', labelsize=8)
            plt.grid(color='g', linestyle='-', linewidth=0.1)

            plt.plot(frame.index, frame[uwilcol], 'g.')
            
            # Over-Plot anomalous data (red)  · · · · · · · · · · · · · · · · · · · · · 
            frame = topanomdf.copy()
            
            # Create TU-WN-IP-LT column & set timestamp index, and drop all the other cols
            frame[uwilcol] = "["+frame['TargetUserName_']+"]["+frame['SourceIP_']+"]["+frame['LogonType_']+"]"
            frame[uwilcol] = frame[uwilcol].astype(str)
            frame = frame.set_index('Timestamp_')
            # Drop all columns except uwilcol
            for col in frame.columns:
                if col != uwilcol:
                    frame = frame.drop(columns=col)

            plt.plot(frame.index, frame[uwilcol], 'r.')
            plt.tight_layout()

            if showplots == True:
                plt.show()

            if saveplots == True and plotsavepath != '':
                pltf2 = plotsavepath + "/uwil-green_red-" + res + ".png"
                print("- Saving UWIL red+green (normal+anomalies) plot: " + pltf2)
                plt.savefig(pltf2)
            
    # Return ------------------------------------------------------------------
    predslossextdf = predslossextdf.sort_values(by="Loss_", ascending=False)
    for col in predslossextdf.columns:
        if re.search("^Loss", col):
            predslossextdf = predslossextdf.drop(columns=col)
    predslossextdf = predslossextdf.reset_index(drop=False)
    
    return predslossextdf

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

    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    evts4624_nonsysusers['WorkstationName'] = evts4624_nonsysusers['WorkstationName'].fillna("-")
    useraccess=evts4624_nonsysusers.reset_index()[["Timestamp","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('Timestamp')
    x = useraccess.groupby([pd.Grouper(freq=freq), "WorkstationName", "IpAddress",'TargetUserName','LogonType']).size()

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

# NOTE: This function is incomplete. It does not invoke the corresponding ML func
def find_anomalies_evtx(indf, type=None):
    if 'D4_DataType_' in indf.columns:
        if indf['evtxFileName_'][0]=='Microsoft-Windows-TaskScheduler%4Operational.evtx':
            hml_df = convert_scheduled_tasks_ham_to_hml(indf)
            return hml_df
        elif indf['evtxFileName_'][0]=='Security.evtx' and type == 'logon':
            hml_df = convert_logon_events_ham_to_hml(indf)
            return hml_df
        elif indf['D4_DataType_'][0]=='evtx-hml':
            hml_df = indf
            return hml_df
        else:
            print("Event type not supported")
            return None, None
        

def convert_logons_ham_to_hml(secevtxrawdf, verbose=0):
    secevtxdfs = evtid_dfs_build(secevtxrawdf, evtids2parse=[4624, 4634], verbose=verbose)
    
    # Read 4634 events and add session length information
    log_e4624df = secevtxdfs[4624]
    log_e4624df.insert(0,'Timestamp_',None)
    log_e4624df['Timestamp_'] = log_e4624df.index
    log_e4624df = log_e4624df.sort_values(by="Timestamp_").reset_index(drop=True)

    log_e4634df = secevtxdfs[4634]

    # Merge 4624 + 4634
    if verbose: print("- Adding session length via evtID 4634 TargetLogonId")
    logoutdf = log_e4634df.reset_index()[['Timestamp','TargetLogonId']]

    log_e4624df = pd.merge(log_e4624df, logoutdf, left_on='TargetLogonId', right_on='TargetLogonId').rename(columns={'Timestamp': 'LogoutTimestamp_'})
    log_e4624df['SessionLength__'] = log_e4624df['LogoutTimestamp_'] - log_e4624df['Timestamp_']
    log_e4624df['SessionLength__'] = log_e4624df['SessionLength__'] / np.timedelta64(1, 's')

    log_e4624df = log_e4624df.rename(columns={"TargetUserName": "TargetUserName_",
                                              "Computer": "Computer_",
                                              "LogonType": "LogonType_",
                                              "IpAddress": "SourceIP_",
                                              "WorkstationName": "WorkstationName_"})
    log_e4624df['TargetHostname_'] = 'd4_null'

    # Lowercase everything
    if verbose: print("- Lowercasing columns: ", end='')
    for column in log_e4624df.columns:
        if isinstance(log_e4624df[column], str) or str(log_e4624df[column].dtype) == "string":
            #print(column + " ", end='')
            log_e4624df[column] = log_e4624df[column].str.lower()

    if verbose: print("\n")

    if verbose: print('- No. 4624 events: '+str(len(log_e4624df)))
    return log_e4624df

    
def convert_scheduled_tasks_ham_to_hml(evtx_ham_df):    
    
    hml_df = evtx_ham_df[['EventID_', 'Computer', '@Name',  '@UserID', 'UserContext', 'UserName', 'ResultCode', 'TaskName']]
    
    # Rename Columns
    hml_df = hml_df.rename(columns={'Computer':    'Computer_'})
    hml_df = hml_df.rename(columns={'@Name':       'AtName_'})
    hml_df = hml_df.rename(columns={'TaskName':    'TaskName_'})
    hml_df = hml_df.rename(columns={'UserName':    'UserName_'})
    hml_df = hml_df.rename(columns={'UserContext': 'UserContext_'})
    hml_df = hml_df.rename(columns={'@UserID':     'AtUserID_'})
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

     
def ml_scheduled_evtx_anomalies(indf, **kwargs): 
    # If you provide the redzone arg the training will be carried out with the
    # data before the redzone and the predictions will be made in the redzone

    # Configuration 
    supportedodalgs = ['simple_autoencoder', 'lstm_autoencoder']

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    argtsmin       = kwargs.get('argtsmin',      None)
    argtsmax       = kwargs.get('argtsmax',      None)
    loss_threshold = kwargs.get('loss_threshold', None)
    epochs         = kwargs.get('epochs',        40)
    kerasverbose   = kwargs.get('kerasverbose',  2)
    verbose        = kwargs.get('verbose',       1)
    redzone        = kwargs.get('redzone',       None)
    odalg          = kwargs.get('odalg',         "simple_autoencoder")   # Outlier Detection Algorithm
    excusers       = kwargs.get('excusers',      None)                   # Excluded Users (to avoid false positives)
    # Anomaly threshold
    ntop           = kwargs.get('ntop',          20)                     # No. Top anomalies to consider
    # Show output
    showoutputs    = kwargs.get('showoutputs',   False)                  # Enable/disable output to screen,
                                                                         # so this function can be used in jupyter/ipython or 
                                                                         # be called as a function from another python script
    display        = kwargs.get('display',       False)                  # Display (for Jupyter notebooks)
    # Plots
    plots          = kwargs.get('lossplot',      True)                   # Generate plots
    lossplot       = kwargs.get('lossplot',      True)                   # Generate loss plot
    showplots      = kwargs.get('showplots',     True)                   # Show plots
    saveplots      = kwargs.get('saveplots',     True)                   # Save plots
    plotsavepath   = kwargs.get('plotsavepath',  '')                     # plot save path
    # ML Feature Selection / Engineering
    usedow         = kwargs.get('usedow',        False)                  # Use Day of the Week as a ML Feature
    usetimeofday   = kwargs.get('usetimeofday',  False)                  # Use Time of the Day (4h intervals) as a ML Feature
    # ML Parameters
    epochs         = kwargs.get('epochs',        40)
    fitloops       = kwargs.get('fitloops',       3)
    lstm_timesteps = kwargs.get('lstm_timesteps', 200)

    # Arguments Validation  - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # redzone
    # TODO: This validation should be a function in d4.common (rz_arg_healthcheck)
    if redzone is not None:
        if type(redzone) != list:
            print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
            redzone = None
        else:
            if len(redzone) != 2:
                print("ERROR: redzone should be a 2 element list: [rzmin, rzmax]")
                redzone = None

    # Arguments Processing  - - - - - - - - - - - - - - - - - - - - - - - - - - 

    if verbose == 0 and 'kerasverbose' not in kwargs.keys():
        kerasverbose=0

    if verbose == 0 and 'lossplot' not in kwargs.keys():
        lossplot = False

    # If plots is False we disable lossplot
    if plots == False and 'lossplot' not in kwargs.keys():
        lossplot = False

    print("- INFO:")
    print("  + lossplot:      " + str(lossplot))
    print("  + plots:        " + str(plots))
    print("  + showplots:    " + str(showplots))
    print("  + saveplots:    " + str(saveplots))
    print("  + plotsavepath: " + str(plotsavepath))

    # redzone · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
    if redzone is not None:
        rzmin = redzone[0]
        rzmax = redzone[1]

    # Arguments Supported Options - - - - - - - - - - - - - - - - - - - - - - - 
    if odalg not in supportedodalgs:
        print("ERROR: Unsupported Anomaly Detection Algorithm: " + str(odalg))
        return

    # HEALTH CHECKS -----------------------------------------------------------

    # [HC] Verify that the fields that we need are present in the df
    # It is possible to run this function on event logs that may have been 
    # obtained from other means other than the original evtx. We will allow that,
    # but we need to ensure that the fields required to make this function work are
    # present

    reqcols = ['EventID_','AtName_','TaskName_','AtUserID_', 'ResultCode_', 'UserNC_', 'Hostname_']
    optcols = [] # Not used currently
    
    missingcols = ""
    for reqcol in reqcols:
        if reqcol not in indf.columns:
            missingcols = missingcols + " " + reqcol
    if missingcols != "":
        print("ERROR: Missing columns on input DF: " + str(missingcols))
        return 

    # DEFINITIONS ------------------------------------------------------------

    # Columns of Interest
    # - CoIs that must always be present
    colsoibase = ['EventID_','AtName_','TaskName_','AtUserID_','ActionName_','ResultCode_', 'UserNC_', 'Hostname_']

    # - Additional CoIs that may or may not be present
    colsoiopt  = []

    # START ------------------------------------------------------------------

    if redzone != None:
        print("- Red Zone:        " + str(redzone[0]) + ' -> ' + str(redzone[1]))
    print("")

    if odalg == "lstm_autoencoder":
        if len(indf) < lstm_timesteps * 2:
            minimum = lstm_timesteps * 2
            print(f"INFO: Not enough events for ML (Minimum = {minimum}). Aborting.")
            return
    else:
        if len(indf) < 50:
            print("INFO: Not enough events for ML (Minimum = 50). Aborting.")
            return

    # DATA PREPARATION --------------------------------------------------------
    if verbose:
        print("DATA PREPARATION:")
      
    
    # Lowercase everything
    for column in indf.columns:
        if type(indf[column]) == "str" or type(indf[column]) == "string":
            indf[column] = indf[column].str.lower() 

    # Red Zone  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TODO: Verify if there actually is data in the Benchmark Zone (< RZ).
    #       If there is no data then we shouldn't divide in train/preds, but
    #       act as if there was no redzone
    #       (WARNING message: "redzone define but no data outside the redzone. Ignoring redzone.)
    if redzone is not None:
        mltraintsdf = indf.query('Timestamp_ < @rzmin').reset_index(drop=True)
        mlpredstsdf = indf.query('Timestamp_ >= @rzmin & Timestamp_ <= @rzmax').reset_index(drop=True)
        if len(mltraintsdf) <= lstm_timesteps * 2:
            print("WARNING: Not enough training data (" + str(len(mltraintsdf)) + ").")
            print("         Training will be done with the full dataset (train+pred).")
            print("         Results will probably be less accurate.")
            print("")
            mltraintsdf = indf
            mlpredstsdf = indf
    else:
        mltraintsdf = indf
        mlpredstsdf = indf
    
    
    #print("- Calling anomaly detection algorithm with:")
    #print("  + traindf: " + str(mltraintsdf.shape) + " -> " + str(mltraintsdf['Timestamp_'].min()) + " - " + str(mltraintsdf['Timestamp_'].max()))
    #print("  + predsdf: " + str(mlpredstsdf.shape)  + " -> " + str(mlpredstsdf['Timestamp_'].min()) + " - " + str(mlpredstsdf['Timestamp_'].max()))
    #print("")

    predslossdf = d4ml.ml_model_execution(traindf=mltraintsdf, preddf=mlpredstsdf, model_type=odalg, fitloops=fitloops, epochss=[epochs], lstm_timesteps=lstm_timesteps, error_threshold=loss_threshold, verbose=verbose, kerasverbose=kerasverbose, cols2drop=[])

    if predslossdf is None:
        print("ERROR: No ML data returned. Something went wrong when calling: d4ml.ml_model_execution.")
        return
    
    lossdf = predslossdf['Loss_']

    if odalg == "lstm_autoencoder":
        # If we use a lstm autoencoder we will add the Timestamp col
        predslossextdf = predslossdf.copy()
        #predslossextdf.insert(0,'Timestamp_',0)
        #predslossextdf['Timestamp_'] = predslossextdf['Timestamp_'].astype('datetime64[ns]') 
        #predslossextdf['Timestamp_'] = mlpredstsdf.loc[lstm_timesteps-1:]['Timestamp_']
    else:
        predslossextdf = predslossdf

    anomdf = predslossextdf.copy().sort_values(by="Loss_", ascending=False)
    for col in anomdf.columns:
        if re.search("^Loss", col):
            anomdf = anomdf.drop(columns=col)
            
    predslossextdf = predslossextdf.sort_values(by="Loss_", ascending=False)
    for col in predslossextdf.columns:
        if re.search("^Loss", col):
            predslossextdf = predslossextdf.drop(columns=col)
    predslossextdf = predslossextdf.reset_index(drop=False)

    ###############################################################################
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    
    # Extract DFs of interest from anomdf dict
    anomuniqdf = anomdf.copy()
    #if 'Timestamp_' in anomuniqdf.columns:
    #    anomuniqdf = anomuniqdf.drop(columns=['Timestamp_'])

    # Anomalies Uniq (ts not taken into account, but without losing the sequence)
    anomuniqdf = anomuniqdf.drop_duplicates().reset_index().rename(columns={'index': 'OrigIndex'})
    topanomuniqdf = anomuniqdf.head(ntop)
    last_topanomuniqdf_index = int(topanomuniqdf.tail(1)['OrigIndex'])

    topanomdf = anomdf

    # We select all the entries from the first anomaly till the <topn> anomaly, considered as <topn> "different" entries
    # This is to avoid a single entry repeated many times in the time realm consumes the <topn> positions
    # We obtain this number in "last_topanomuniqdf_index"
    topanomdf = topanomdf.loc[:last_topanomuniqdf_index]

    if showoutputs == True:
        print('Top ' + str(ntop) + ' Anomalies - UWIL Unique')
        print(topanomuniqdf)
        print('')

        print('TOP ' + str(ntop) + ' UWIL GROUPBY TABLE')
        print(top_uwil_groupby_tabledf)
        print('')

        print('Top Anomalies derived from Top ' + str(ntop) + ' UWIL Unique Anomalies', 3)
        print(topanomdf)
        print('')

    # TODO: This is typically too big to print. We will just export it. 
    #print('FULL UWIL GROUPBY TABLE')
    #print(uwil_groupby_tabledf)
    #print('')

    # PLOTS -------------------------------------------------------------------

    # Plot Loss  - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - -

    return predslossextdf
    