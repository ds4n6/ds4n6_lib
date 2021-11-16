#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4ml

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import time
import datetime
import inspect

import json
import pickle

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

# tensorflow-keras (for GPU use)
from tensorflow.keras.models            import Model, Sequential, load_model
from tensorflow.keras.layers            import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks         import History 


# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.common as d4com
import ds4n6_lib.utils  as d4utl
import ds4n6_lib.evtx  as d4evtx


###############################################################################
# FUNCTIONS
###############################################################################

def find_anomalies(indf, model_type="simple_autoencoder", **kwargs):
    if 'D4_DataType_' in indf.columns:
        if indf['D4_DataType_'][0]=='evtx':
            hml_df = d4evtx.find_anomalies_evtx(indf)
        #elif indf['D4_DataType_'][0]=='flist':
        #    hml_df = convert_flist_ham_to_hml(indf)
        elif indf['D4_DataType_'][0]=='evtx-hml': #or indf['D4_DataType_'][0]=='flist-hml':
            d4evtx.find_anomalies_evtx(indf)
        else:
            print("DataFrame type not supported")
            return None, None
    else:
        print("DataFrame type not supported")
        return None, None
    
    hml_df = hml_df.drop(columns=['D4_DataType_',])

    if model_type == "simple_autoencoder":
        topanomdf, anomdf, lossdf  = ml_model_execution_quick_case(traindf=hml_df, preddf=hml_df, model_type=model_type, **kwargs)
        return anomdf, lossdf
    elif model_type == "lstm_autoencoder":
        topanomdf, anomdf, lossdf, lossftdf = ml_model_execution_quick_case(traindf=hml_df, preddf=hml_df, model_type=model_type, **kwargs)
        return anomdf, lossdf

def ml_model_execution_quick_case(**kwargs):

    # traindf                = kwargs.get('traindf',                None)
    # trainarr               = kwargs.get('trainarr',               None)
    # preddf                 = kwargs.get('preddf',                 None)
    # model_type             = kwargs.get('model_type',             None)
    # lstm_units             = kwargs.get('lstm_units',             None)
    # lstm_time_steps        = kwargs.get('lstm_time_steps',        None)
    # activation_function    = kwargs.get('activation_function',    None)
    # model_filename         = kwargs.get('model_filename',         None)
    # loops                  = kwargs.get('loops',                  1)
    # epochss                = kwargs.get('epochss',                [2])
    # error_ntop             = kwargs.get('error_ntop',             5000)
    # verbose                = kwargs.get('verbose',                0)
    # cols2drop              = kwargs.get('cols2drop',              [])
    # transform_method       = kwargs.get('transform:_method',      "label_encoder")
    # batch_size             = kwargs.get('batch_size',             8)
    # ntop_anom              = kwargs.get('ntop_anom',              500)
    # autosave_miloss_model  = kwargs.get('autosave_minloss_model', True)
    # maxcnt                 = kwargs.get('maxcnt',                 1)
    # activation_function    = kwargs.get('activation_function',    "tanh")
    
    return model_execution(**kwargs)

def model_execution(**kwargs):

    verbose                = kwargs.get('verbose',                0)
    traindf                = kwargs.get('traindf',                None)
    predindf               = kwargs.get('preddf',                 None)
    evilquerystring        = kwargs.get('evilquerystring',        None)
    cols2drop              = kwargs.get('cols2drop',              [])
    model_type             = kwargs.get('model_type',             "simple_autoencoder")
    epochss                = kwargs.get('epochss',                [10])
    ntop_anom              = kwargs.get('ntop_anom',               200)
    maxcnt                 = kwargs.get('maxcnt',                 1)
    # unused parameters
    # model_filename_root    = kwargs.get('model_filename_root',    None)
    # model_filename         = kwargs.get('model_filename',         None)
    evilqueryfield         = kwargs.get('evilqueryfield',         None)
    # loops                  = kwargs.get('loops',                  1)
    # batch_size             = kwargs.get('batch_size',             8)
    # error_ntop             = kwargs.get('error_ntop',             None)
    # autosave_minloss_model = kwargs.get('autosave_minloss_model', False)

    # Model-specific Arguments
    lstm_time_steps        = kwargs.get('lstm_time_steps',        200)

    if d4.debug != 0:
        verbose = d4.debug

    # Drop columns - During the Model Definition we can try dropping columns see if that improves our detection
    if traindf is not None:
        traindf = traindf.drop(columns=cols2drop)
    if predindf   is not None:
        predindf  = predindf.drop(columns=cols2drop)

    for epochs in epochss:
        display(Markdown("----"))
        display(Markdown("**"+str(epochs)+" epochs**"))
        evilentryidxs = []
        cnt=1

        while cnt <= maxcnt:
            if model_type == "lstm_autoencoder":
                # In the LSTM case we are also interested in the per-feature loss,
                # since the global loss is calculated as the mean of the losses of
                # the features
                if d4.debug == 5:
                     predinarr, predoutarr, anomdf, loss, lossft = ml_autoencoder(**kwargs, epochs=epochs)
                else:
                     anomdf, loss, lossft = ml_autoencoder(**kwargs, epochs=epochs)

                # We need to shift the predindf by lstm_time_steps-1 
                predinshifteddf = predindf.iloc[lstm_time_steps-1:]
                predinshifteddf.index -= lstm_time_steps-1

                if d4.debug >= 3:
                    display(predinshifteddf.head(3))
                    display(predinshifteddf.tail(3))

                # Sort anomdf by loss
                erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
                anomsorteddf = predinshifteddf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})
                if d4.debug >= 3:
                    print("DEBUG: erroranomidx -> "+str(type(erroranomidx))+str(len(erroranomidx)))
                    print("DEBUG: predinshifteddf -> "+str(type(predinshifteddf))+" -> "+str(predindf.iloc[lstm_time_steps-1:].shape))
                    print("DEBUG: anomsorteddf -> "+str(type(anomsorteddf))+str(anomsorteddf.shape))
                    display(anomsorteddf.head(3))
                    display(anomsorteddf.tail(3))
            else:
                anomdf, loss  = ml_autoencoder(**kwargs, epochs=epochs)

                # Sort anomdf by loss
                erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
                anomsorteddf = predindf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})

            if d4.debug >= 3:
                print("")
                print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
                print("DEBUG: anomdf.dtypes: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                print(anomdf.dtypes)
                print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print("")


            #erroranomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index']).values)
            #anomsorteddf = predindf.loc[erroranomidx].reset_index().rename(columns={"index": "Orig_Index"})

            # Find & Print our evil entries
            if (evilqueryfield is not None) and (evilquerystring is not None):
                query         = eval("evilqueryfield")+'.str.contains("'+evilquerystring+'")'
                evilentriesdf = anomsorteddf.query(query, engine="python")

                print("")
                display(Markdown("**Evil Entries:**"))
                display(evilentriesdf)
                print("")

                if evilentriesdf.shape[0] == 0:
                    print("ERROR: Evil Entry not found. Verify Injection.")
                    print("")
                else:
                    evilentryorigidx = evilentriesdf.head(1).index

                    if model_type == "lstm_autoencoder":
                        # The LSTM model drops the first time_steps values from the data,
                        # so the index needs to be corrected
                        evilentrylstmidx = evilentryorigidx 
                        evilentryidx = anomsorteddf.query('index == @evilentrylstmidx').index.values[0]
                    else:
                        evilentryidx = anomsorteddf.query('index == @evilentryorigidx').index.values[0]

                    # Save in list (multiple cnt-runs)
                    evilentryidxs.append(evilentryidx)

                    display(Markdown("**Best Evil Entry Error Order: "+str(evilentryidxs)+"**"))
                    print("")

            # Print Top Anomalies
            if ntop_anom > anomdf.shape[0]:
                ntop_anom = anomdf.shape[0]

            errortopanomidx = list(pd.Series(pd.DataFrame(loss.sort_values(ascending=False)).reset_index()['index'].head(ntop_anom)).values)
            if model_type == "lstm_autoencoder":
                # The LSTM model drops the first time_steps values from the data,
                # so the index needs to be corrected
                errortopanomidx = [x + lstm_time_steps - 1 for x in errortopanomidx]

            if d4.debug >= 3:
                print("DEBUG: [DBG"+str(d4.debug)+"] errortopanomidx: ", end='')
                print(errortopanomidx)

            topanomdf = anomdf.loc[errortopanomidx].reset_index()
            losssr    = pd.Series(loss)

            #anomdf = anomdf.loc[errortopanomidx].reset_index()
            if verbose >= 1:
                display(Markdown("**TOP 10 ANOMALIES**"))
                display(topanomdf.head(10))

            cnt += 1

    if model_type == "lstm_autoencoder":
        # Convert np.array to DF - LSTM-specific
        lossftdf  = pd.DataFrame(lossft, columns=predindf.columns)

        if d4.debug == 5:
            return predinarr, predoutarr, topanomdf, anomsorteddf, losssr, lossftdf
        else:
            return topanomdf, anomsorteddf, losssr, lossftdf
    else:
        return topanomdf, anomsorteddf, losssr

# ML MODELS ###################################################################

def ml_autoencoder(**kwargs): 
    '''
    # GENERAL COMMENTS ========================================================
    #
    # TRAIN / PREDICTION DATA =================================================
    #
    # APPLICABILITY:
    #
    # - This function has been tested at the moment in the following scenarios:
    #   + The prediction dataset includes the train dataset
    #
    # IMPORTANT NOTES:
    #
    # - This function requires 2 dataset inputs: train and prediction
    # - Since this function is an anomaly-oriented implementation test data is 
    #   not valuable, so there will be no test dataset (we will not split the
    #   input training data in the traditional train/test datasets)
    # - The train and prediction datasets can be the same or different.
    #   This is so because you may have the chance to have a clean dataset
    #   without the anomalies you are looking for (e.g. previous to the 
    #   intrusion) or not
    # 
    # NAMING CONVENTIONS: 
    # 
    # For predictions we will use the predindf / predoutdf convention:
    # - predindf  -> Input  data to the model
    # - predoutdf -> Output data from the model (predictions)
    #
    # This in/out differentation is not necessary for the train data, since 
    # there will be no output data from the model after the training process. 
    # So for the train data we will just use: 
    # - traindf   -> Input data to the model
    #
    # =========================================================================
    '''

    # FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create_lstm_dataset_train(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])

        if d4.debug >= 3:
            print("")
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
            print("DEBUG: - X  -> "+str(type(X))+" -> "+str(X.shape))
            print("DEBUG: - y  -> "+str(type(y))+" -> "+str(y.shape))
            print("DEBUG: - Xs -> "+str(type(np.array(Xs)))+" -> "+str(np.array(Xs).shape))
            print("DEBUG: - ys -> "+str(type(np.array(ys)))+" -> "+str(np.array(ys).shape))
            print("")

        #return np.array(Xs), np.array(ys)
        return np.array(Xs)

    def create_lstm_dataset_predict(X, time_steps=1):
        Xs = []
        for i in range(len(X) - time_steps + 1):
        #for i in range(0, len(X), time_steps):
            v = X.iloc[i:(i + time_steps)].values
            #print("XXXX: "+str(i)+" - "+str(v))
            Xs.append(v)        
        return np.array(Xs)

    def flatten_3d_array(X):
        '''
        # Input:  X           ->  3D array for lstm, sample x timesteps x features.
        # Output: flattened_X ->  2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)

    def prediction_data_preparation(predindf, transform_method, data_scaling_method, verbose):
        print("- Transforming prediction columns ("+transform_method+")")

        if verbose >= 1:
            print("\n[PRED] Before Transform:")
            display(predindf.head(4))

        # This mechanism transforms everything as categoricals
        if transform_method == "categorical_basic":

            transform_dict = {}
            for col in predindf.columns:
                cats = pd.Categorical(predindf[col]).categories
                d = {}
                for i, cat in enumerate(cats):
                    d[cat] = i
                    transform_dict[col] = d
            
            inverse_transform_dict = {}
            for col, d in transform_dict.items():
                   inverse_transform_dict[col] = {v:k for k, v in d.items()}

            predindf = predindf.replace(transform_dict)

        elif transform_method == "label_encoder":
            transform_dict = {}
            for col in predindf.columns:
                transform_dict[col] = LabelEncoder()
                predindf[col] = transform_dict[col].fit_transform(predindf[col].astype(str))

        else:
            print("ERROR: Invalid transform code: "+transform_method)
            return
        
        if verbose >= 1:
            print("[PRED] After Transform:")
            display(predindf.head(4))
            print("")

        # Scaling - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        scaler = None

        predinarr = np.array(predindf)

        if data_scaling_method != "none":
            if data_scaling_method == "normalize":
                print("- Scaling Data -> Normalize")
                scaler   = MinMaxScaler()
                scaler   = scaler.fit(predinarr)
                predinarr = scaler.transform(predinarr)

            elif data_scaling_method == "standardize":
                print("- Scaling Data -> Standardize")
                print("  + WARNING: standardization is not implemented yet. Skipping.")
            
            if verbose >= 1:
                print("\n[PRED] After Scaling -> "+data_scaling_method)
                display(pd.DataFrame(predinarr, columns=predindf.columns).head(4))

        # LSTM  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if model_type == "lstm_autoencoder":
            print("- [PRED] Creating 3D pred numpy array from predinarr (this can take a long time)...")
            predinarr2d = predinarr
            predinarr3d = create_lstm_dataset_predict(pd.DataFrame(predinarr2d), lstm_time_steps)
            predinarr   = predinarr3d
            print("  + "+str(predinarr2d.shape)+" -> "+str(predinarr3d.shape))
            
        return predinarr, transform_dict, scaler

    def train_data_preparation(traindf, transform_method, transform_dict, data_scaling_method, scaler, lstm_time_steps=None, verbose=0):

        # COMMON PREPARATION FOR ALL MODELS - - - - - - - - - - - - - - - - - - - - 
        print("- [TRAIN] Transforming columns ("+transform_method+")")

        if verbose >= 1:
            print("\n[TRAIN] Before Transform:")
            display(traindf.head(4))

        if transform_method == "categorical_basic":
            traindf = traindf.replace(transform_dict)

        elif transform_method == "label_encoder":
            for col in traindf.columns:
                traindf[col] = transform_dict[col].fit_transform(traindf[col].astype(str))
        else:
            print("ERROR: Invalid transform code: "+transform_method)
            return
        
        if verbose >= 1:
            print("\n[TRAIN] After Transform:")
            display(traindf.head(4))

        # Scaling - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
        trainarr = np.array(traindf)

        if d4.debug >= 3:
            print("\n[TRAIN] After Transform (np.array):")
            display(trainarr[:4])

        if data_scaling_method != "none":
            if data_scaling_method == "normalize":
                # Scale the train dataset
                trainarr = scaler.transform(trainarr)

                if d4.debug >= 3:
                    print("")
                    print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
                    print("DEBUG: Per-feature Scaling Min-Max: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                    for i in range(predindf.shape[1]):
                        print('%d, train: min=%.3f, max=%.3f' % (i, trainarr[:, i].min(), trainarr[:, i].max()))
                    print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print("")

            elif data_scaling_method == "standardize":
                print("- Scaling Data -> Standardize")
                print("  + WARNING: standardization is not implemented yet. Skipping.")

            if verbose >= 1:
                print("\n[TRAIN] After Scaling -> "+data_scaling_method)
                display(pd.DataFrame(trainarr, columns=traindf.columns).head(4))

        # Model-specific Data Preparation - - - - - - - - - - - - - - - - - - - - - 
        if model_type == "lstm_autoencoder":
            print("- [TRAIN] Creating 3D pred numpy array from predinarr (this can take a long time)...")
            trainarr2d = trainarr
            trainarr3d = create_lstm_dataset_train(pd.DataFrame(trainarr2d), pd.DataFrame(trainarr2d), lstm_time_steps)
            trainarr   = trainarr3d
            print("  + "+str(trainarr2d.shape)+" -> "+str(trainarr3d.shape))

        return trainarr

    # ARGUMENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Generic - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    verbose                = kwargs.get('verbose',                0)
    # Hardware  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    use_gpu                = kwargs.get('use_gpu',                False)
    # Input Data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    trainorigdf            = kwargs.get('traindf',                None)
    predinorigdf           = kwargs.get('preddf',                 None)
    # Input Data Filtering - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # argtsmin               = kwargs.get('argtsmin',               None)
    # argtsmax               = kwargs.get('argtsmax',               None)
    # Data Preparation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    transform_method       = kwargs.get('transform_method',       "label_encoder")
    data_scaling_method    = kwargs.get('data_scaling_method',    "normalize")     # { none | normalize | (standardize) }
    # Model Definition - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model_type             = kwargs.get('model_type',             "simple_autoencoder")
    encoding_dim           = kwargs.get('encoding_dim',           3)
    activation_function    = kwargs.get('activation_function',    "relu")
    optimizer              = kwargs.get('optimizer',              None)
    loss                   = kwargs.get('loss',                   None)
    lstm_time_steps        = kwargs.get('lstm_time_steps',        200)
    lstm_units             = kwargs.get('lstm_units',             50)
    # Model Training - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    epochs                 = kwargs.get('epochs',                 40)
    batch_size             = kwargs.get('batch_size',             32)
    loops                  = kwargs.get('loops',                  1)
    # Model Saving - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model_filename         = kwargs.get('model_filename',         None)
    model_filename_root    = kwargs.get('model_filename_root',    None)
    autosave_minloss_model = kwargs.get('autosave_minloss_model', False)
    # Predictions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    error_threshold        = kwargs.get('error_threshold',          None)
    error_ntop             = kwargs.get('error_ntop',               None)

    # BODY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # HEALTH CHECKS -----------------------------------------------------------
    if model_type is None:
        print("ERROR: model_type not selected.")
        print("Options: { simple_autoencoder | multilayer_autoencoder }")
        return

    # CREATE A COPY OF ORIGINAL DATA FOR MANIPULATION -------------------------
    #
    # Both traindf and predindf will be modified through the Data Preparation
    # phases (encoding, normalizing, etc.), this is because we save the original
    # data as:
    # - trainorigdf  -> Input train data       (as provided to the function)
    # - predinorigdf -> Input predictions data (as provided to the function)

    traindf   = trainorigdf.copy()
    predindf  = predinorigdf.copy()

    # ARGUMENT PROCESSING -----------------------------------------------------
    if traindf is not None:
        traindf_shape = traindf.shape
    else:
        traindf_shape = None

    if predindf is not None:
        predindf_shape = predindf.shape
    else:
        predindf_shape = None

    # INFO --------------------------------------------------------------------
    print("- General:")
    print("  + Verbosity:               "+str(verbose))
    print("- Input Data:")       
    print("  + Train DF:                "+str(traindf_shape))
    print("  + Prediction DF:           "+str(predindf_shape))
    print("- Data Preparation:")
    print("  + Transform Method:        "+transform_method)
    print("  + Data Scaling Method:     "+str(data_scaling_method))
    print("- Model Parameters:")
    print("  + Model Type:              "+model_type)
    print("  + Encoding Dimension:      "+str(encoding_dim))
    print("  + Activation Function:     "+str(activation_function))
    # Model-specific Arguments
    if model_type == "lstm_autoencoder":
        print("  + lstm_units:              "+str(lstm_units))
        print("  + lstm_time_steps:         "+str(lstm_time_steps))
    print("- Training Parameters:") 
    print("  + Training Loops:          "+str(loops))
    print("  + epochs:                  "+str(epochs))
    print("  + batch_size:              "+str(batch_size))
    print("- Model Saving:")         
    print("  + autosave_minloss_model:  "+str(autosave_minloss_model))
    print("  + model_filename:          "+str(model_filename))
    print("  + model_filename_root:     "+str(model_filename_root))
    print("- Predictions:")        
    print("  + error_threshold:         "+str(error_threshold))
    print("  + error_ntop:              "+str(error_ntop))
    print("")

    # INITIALIZATION ----------------------------------------------------------
    np.random.seed(8)

    runid = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    # DATA PREPARATION --------------------------------------------------------
    # We will do Data Preparation for the prediction data first, in order to 
    # make sure there are no outliers which may break the process later
    # In the current implementation of this function, the train dataset is
    # included in the prediction dataset, so the right dataset to define the 
    # transform matrixes (transform, standardization, normalization, etc.) is 
    # the preddf. And then, we will apply that to the traindf (which will be a
    # subset of the preddf as we say)
  
    # DATA SUBSETS  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if model_type == "lstm_autoencoder":
        predinoriglstmdf = predinorigdf.iloc[lstm_time_steps-1:]

    # PREDICTION  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    predinarr, transform_dict, scaler = prediction_data_preparation(predindf, transform_method, data_scaling_method, verbose)
    #print("XXXX: "+str(predinarr.shape))


    # CREATING THE NEURAL NETWORK ARCHITECTURE --------------------------------
    # Load model, if model file exists
    if model_filename is not None and os.path.exists(model_filename):
            model_loaded_from_file = True

            print("- Loading Model from:")
            print("    "+model_filename)
            autoencoder = load_model(model_filename)

            print("")
            print("Autoencoder Summary:")
            print("")
            autoencoder.summary()
            print("")

    else:
        # TRAIN - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        trainarr = train_data_preparation(traindf, transform_method, transform_dict, data_scaling_method, scaler, lstm_time_steps, verbose)

        model_loaded_from_file = False

        # DATA PREPARATION --------------------------------------------------------

        # SELECTING TRAINING / TEST DATA ------------------------------------------
        # Not useful in unsupervised anomaly detection. Skipping.

        # ML MODEL CREATION -------------------------------------------------------
        nfeatures    = traindf.shape[1]

        models   = {}
        mdlnames = []
        losses   = []
        tlcnt    = 1
        while tlcnt <= loops:
            print("")
            print("[LOOP ITERATION: "+str(tlcnt)+"/"+str(loops)+"]")
            print("")

            print("MODEL CREATION")
            print("")


            # We create an encoder and decoder. 
            # The ReLU function, which is a non-linear activation function, is used in the encoder. 
            # The encoded layer is passed on to the decoder, where it tries to reconstruct the input data pattern

            if model_type == "simple_autoencoder":
                input_dim    = nfeatures
                hidden_dim   = encoding_dim

                print("- Creating Model")
                print("  + No. Features:          "+str(nfeatures))
                print("  + Input Array Dimension: "+str(input_dim))
                print("")

                input_layer = Input(shape=(input_dim,))
                print(input_layer)
                encoded     = Dense(encoding_dim, activation='relu'  )(input_layer)
                decoded     = Dense(input_dim,    activation='linear')(encoded)

                autoencoder = Model(input_layer, decoded)

                optimizer_deflt = 'adadelta'
                loss_deflt      = 'mse'

            elif model_type == "multilayer_autoencoder":
                input_dim    = nfeatures
                hidden_dim   = encoding_dim

                print("- Creating Model")
                print("  + No. Features:    "+str(nfeatures))
                print("  + Input Array Dimension: "+str(input_dim))
                print("")

                input_layer = Input(shape=(input_dim,))
                encoded     = Dense(encoding_dim, activation="relu"  )(input_layer)
                encoded     = Dense(hidden_dim,   activation="relu"  )(encoded)
                decoded     = Dense(hidden_dim,   activation="relu"  )(encoded)
                decoded     = Dense(encoding_dim, activation="relu"  )(decoded)
                decoded     = Dense(input_dim,    activation="linear")(decoded)

                autoencoder = Model(input_layer, decoded)

                optimizer_deflt = 'adadelta'
                loss_deflt      = 'mse'

            elif model_type == "lstm_autoencoder":
                # Refs: - https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
                #       - https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/

                print("- Creating Model")
                print("  + No. Features:          "+str(nfeatures))
                print("  + Input Array Dimension: "+str(trainarr.shape))
                print("")

                # SINGLE LAYER - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
                #
                # For LSTM use with GPUs, see:
                # - https://keras.io/api/layers/recurrent_layers/lstm/
                # - https://stackoverflow.com/questions/62044838/using-cudnn-kernel-for-lstm

                autoencoder = Sequential()

                if use_gpu :
                    autoencoder.add(LSTM(lstm_units, activation=activation_function, input_shape=(lstm_time_steps, nfeatures),
                                         recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False, use_bias=True))
                else:
                    autoencoder.add(LSTM(lstm_units, activation=activation_function, input_shape=(lstm_time_steps, nfeatures)))
                    
                autoencoder.add(RepeatVector(n=lstm_time_steps))
                autoencoder.add(LSTM(lstm_units, return_sequences=True))
                autoencoder.add(TimeDistributed(Dense(nfeatures)))

                # MULTILAYER - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #  autoencoder = keras.Sequential()
                #  # Encoder
                #  autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
                #  autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
                #  autoencoder.add(RepeatVector(timesteps))
                #  # Decoder
                #  autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
                #  autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
                #  autoencoder.add(TimeDistributed(Dense(input_dim)))

                # With Dropout Layers  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #  model = keras.Sequential()
                #  model.add(keras.layers.LSTM( units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
                #  model.add(keras.layers.Dropout(rate=0.2))
                #  model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
                #  model.add(keras.layers.LSTM(units=64, return_sequences=True))
                #  model.add(keras.layers.Dropout(rate=0.2))
                #  model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
                #  model.compile(loss='mae', optimizer='adam')

                optimizer_deflt = 'adam'
                loss_deflt      = 'mae'
           
            if optimizer is None:
                optimizer = optimizer_deflt

            if loss is None:
                loss = loss_deflt

            # The following model maps the input to its reconstruction, which is done in the decoder layer, decoded.
            # Next, the optimizer and loss function is defined using the compile method.
            # The adadelta optimizer uses exponentially-decaying gradient averages and is a highly-adaptive learning rate method.
            # The reconstruction is a linear process and is defined in the decoder using the linear activation function.
            # The loss is defined as mse, which is mean squared error

            print("- Compiling Model")
            print("")
            autoencoder.compile(optimizer=optimizer, loss=loss)
            modelname = autoencoder.name

            # Save autoencoder name to list
            mdlnames.append(autoencoder.name)
            # Save autoencoder model to dict
            models[modelname] = autoencoder

            display(Markdown("**Autoencoder Summary:**"))
            print("")
            autoencoder.summary()
            print("")

            # TRAINING THE NETWORK --------------------------------
            print("TRAINING")
            print("")
            print("- Training Info:")
            print("  + epochs     = "+str(epochs))
            print("  + batch_size = "+str(batch_size))
            print("")

            
            now = datetime.datetime.now()
            print("- Training Start: "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
            print("")

            losshist = autoencoder.fit(trainarr, trainarr, epochs=epochs, batch_size=batch_size)
            losses.append(losshist.history['loss'][-1])
            tlcnt += 1

            now = datetime.datetime.now()
            print("")
            print("- Training End:   "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
            print("")

        minloss        = min(losses)
        minlossidx     = losses.index(minloss)
        minlossmdlname = mdlnames[minlossidx]
        minlossmodel   = models[minlossmdlname]

        print("")
        print("- Models:          "+str(mdlnames))
        print("- Losses:          "+str(losses))
        print("- Min. Loss:       "+str(minloss))
        print("- Min. Loss Model: "+str(minlossmdlname))
        print("")
    
        # Save model, if model file does not exist
        if autosave_minloss_model:
            if autosave_minloss_model:
                minlossautoencoder = minlossmodel
                if model_filename_root is not None:
                    model_filename_root = model_filename_root
                else:
                    model_filename_root = ""

                model_filename_save = model_filename_root+"-"+model_type+"-"+runid+"-loss_"+str(minloss)+".h5"

                if not os.path.exists(model_filename_save):
                    print("- Saving Model to:")
                    print("    "+model_filename_save)
                    print("")
                    autoencoder.save(model_filename_save)
            else:
                if not os.path.exists(model_filename):
                    print("- Saving Model to:")
                    print("    "+model_filename)
                    autoencoder.save(model_filename)
                    print("")

    # DO PREDICTIONS / FIND ANOMALIES =========================================

    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:

    print("PREDICTIONS")
    print("")

    # RUN MODEL ---------------------------------------------------------------
    if not model_loaded_from_file:
        # Set autoencoder model to the one with minimal loss
        print("- Setting autoencoder to min loss model: "+minlossmdlname)
        autoencoder = minlossmodel
        
    print("- Input DF Shape: "+str(predindf.shape)+" -> "+str(predinarr.shape))

    print("- Running predictions...")
    now = datetime.datetime.now()
    print("  + Start: "+str(now.strftime("%Y-%m-%d %H:%M:%S")))
    predoutarr = autoencoder.predict(predinarr)
    now = datetime.datetime.now()
    print("  + End:   "+str(now.strftime("%Y-%m-%d %H:%M:%S")))

    # Calculate Loss  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if   model_type == "simple_autoencoder":
        loss = np.mean(np.power(predinarr - predoutarr, 2), axis=1)
        predinfindf = predindf
    elif model_type == "multilayer_autoencoder":
        loss = np.mean(np.power(predinarr - predoutarr, 2), axis=1)
        predinfindf = predindf
    elif model_type == "lstm_autoencoder":

        predinarr_flat  = flatten_3d_array(predinarr)
        predoutarr_flat = flatten_3d_array(predoutarr)

        loss_ft   = np.power(predinarr_flat - predoutarr_flat, 2)
        loss_full = np.mean(np.power(predinarr_flat - predoutarr_flat, 2), axis=1)

        loss      = loss_full

        predinfindf = predindf.iloc[lstm_time_steps:]

        if d4.debug >= 3:
            print("")
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
            print("DEBUG: - predindf:        "+str(predindf.shape))
            print("DEBUG: - predinfindf:     "+str(predinfindf.shape))
            print("DEBUG: - predinarr:       "+str(predinarr.shape))
            print("DEBUG: - predinarr_flat:  "+str(predinarr_flat.shape))
            print("DEBUG: - predoutarr:      "+str(predoutarr.shape))
            print("DEBUG: - predoutarr_flat: "+str(predoutarr_flat.shape))
            print("DEBUG: - loss :           "+str(loss.shape))
            print("")

    plt.plot(loss)

    # OBTAIN ANOMALIES --------------------------------------------------------

#test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
#test_score_df['loss'] = test_mae_loss

    # Auto-calculate error_threshold - Top N values
    if error_threshold is None:
        if error_ntop is None:
            ntoperror = 15
        else:
            # Ensure that the number of error rows specified are <= DF rows
            if error_ntop >= predinfindf.shape[0]:
                ntoperror = predinfindf.shape[0]
            else:
                ntoperror = error_ntop

        error_threshold = int(np.sort(loss)[-ntoperror])
        autocalcmsg = " (Auto-calculated - Top "+str(ntoperror)+")"
    else:
        autocalcmsg = ""

    print("- Error Threshold: "+str(error_threshold)+autocalcmsg)


    # Select entries above the error threshold
    # Instead of using predoutdf we will use predinfindf, which has been adapted
    # to be the same dimension as predoutdf and is good enough to identify the
    # anomalous entries as predoutdf, since they both have corresponding entries
    # (original vs predicted) in a 1-to-1 way
    if   model_type == "simple_autoencoder":
        anomdf = predinorigdf[loss >= error_threshold]
    elif model_type == "multilayer_autoencoder":
        anomdf = predinorigdf[loss >= error_threshold]
    elif model_type == "lstm_autoencoder":
        anomdf = predinoriglstmdf[loss >= error_threshold]

    if d4.debug >= 3:
        print("")
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
        print("DEBUG: anomdf shape: "+str(anomdf.shape))
        print("DEBUG: anomdf dtype & head: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        display(anomdf.dtypes)
        display(anomdf.head(4))
        print("DEBUG: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("")

    print("- No.Anomalies: "+str(len(anomdf)))
    print("- RUN ID:       "+str(runid))

    # RETURN ------------------------------------------------------------------
    losssr = pd.Series(loss)

    if model_type == "lstm_autoencoder":
        if d4.debug == 5:
            return predinarr_flat, predoutarr_flat, anomdf, losssr, loss_ft
        else:
            return anomdf, losssr, loss_ft
    else:
        return anomdf, losssr
