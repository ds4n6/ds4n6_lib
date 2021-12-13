# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4pl

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
from tqdm import tqdm
import xml.etree.ElementTree as et
import subprocess
import datetime

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython         import get_ipython
from IPython.display import display, Markdown, HTML, Javascript

# ML  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from sklearn.model_selection import train_test_split

# DS4N6 IMPORTS ---------------------------------------------------------------
import ds4n6_lib.d4     as d4
import ds4n6_lib.utils  as d4utl
import ds4n6_lib.evtx   as d4evtx
import ds4n6_lib.common as d4com
import ds4n6_lib.gui    as d4gui

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
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if   bool(re.search(r"\.csv$", evdl, re.IGNORECASE)):
        return read_plaso_l2tcsv(evdl, **kwargs)
    
    elif bool(re.search(r"\.json$", evdl, re.IGNORECASE)):
        return read_plaso_json(evdl, **kwargs)
    else:
        print("ERROR: Unable to read input file. Unsupported file extension.")
        return

def read_plaso_l2tcsv(plasof, **kwargs):
    """ Read plaso data from from a l2tcsv file

        Args: 
            plasof (str): path to file source
            kwargs: read options
        Returns: 
            dictionary of pandas.DataFrame
    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    
    dfs = {}
    # When creating the .csv file with psort, make sure to filter out events 
    # with absurd timestamps (e.g. < 1970 or > the current year). 
    # plaso sometimes includes "broken" timestamps and the will break pandas
    # psort.py -z UTC -o l2tcsv -w plaso_file.csv plaso_file.dump \ 
    #     "date > '1960-01-01 00:00:00' AND date < '2020-12-31 23:59:59'"
    plines=pd.read_csv(plasof)
    
    if 'date' in plines.columns:
        # Now, we need to clean the data. There may be wrong values.
        plines=plines[~plines['date'].str.contains('/00/')]
        # Let's include:
        # - a timestamp column which combines the date and time cols, and is 
        #   of type datetime64
        # - a column "sourcetype_clean" similar to "sourcetype" but "cleaner",
        #   i.e. easier to use
        plines.insert(0, 'timestamp', 0)
        plines.insert(7, 'sourcetype_clean', '')
        plines['sourcetype_clean']=plines['sourcetype'].str.replace('UNKNOWN : ','').str.replace(' ','_')
        plines['timestamp']=pd.to_datetime(plines['date']+" "+plines['time'])

        # Let's create a DF for each source (pe_compilation_time, winlogon, ...),
        # It will be easier for analysis
        srctypes = plines['sourcetype_clean'].unique()
        for srctype in srctypes:
            srctypevar = str(srctype).lower()
            print('Reading source_type %-40s into dataframe ->  %-40s' % (srctype,srctypevar))
            dfs[srctypevar] = plines[plines['sourcetype_clean'] == srctype ].copy()
    
    else:
        dfs = plines
    
    
    return dfs

def read_plaso_json(evdl, **kwargs):
    """ Read plaso data from from a json file

        Args: 
            evdl (str): path to file source
            kwargs: read options
        Returns: 
            dictionary of pandas.DataFrame
    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    #
    # Generate your plaso json file with the following command:
    #     psort.py -z UTC -o json -w plaso_file.json plaso_file.dump "date > '1960-01-01 00:00:00' AND date < '2021-01-01 00:00:00'"

    # Parse Arguments
    tool                     = kwargs.get('tool',                    '')
    hostname                 = kwargs.get('hostname',                '')
    do_harmonize             = kwargs.get('harmonize',               True)
    build_all_df             = kwargs.get('build_all_df'            , False)
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
    
        # Read Data - - - - - - - - - - - - - - - - - - - - - - - - - -
        if os.path.exists(pklrawf) and use_pickle :
            print("- Saved RAW pickle file found:")
            print("      "+pklrawf)
            print("- Reading data from RAW pickle file...")
            dfs = pickle.load(open(pklrawf, "rb"))
            print("- Done.")
            print("")
        else:
            if not os.path.exists(evdl):
                print("ERROR: File does not exist:")
                print("         "+evdl)
                print("")
                return

            dfs = {}
            print("    " + evdl)
            print('- Reading json file lines: ',end='')
            cnt = 0
            with open(evdl) as infile:
                evtsdict = {}
                for line in infile:
                    if line.strip() != "}":
                        lineclean = re.sub(r'^[,{]\ *','', line)
                        datatype = lineclean.split('"data_type": ', maxsplit=1)[-1].split(maxsplit=1)[0].replace('"', '').replace(",", "").replace(":","_")
            
                        if datatype not in evtsdict.keys():
                            evtsdict[datatype]={}
                        linecleanjson = json.loads("{"+lineclean+"}")
                        evtsdict[datatype].update(linecleanjson)

                    # Print progress counter
                    if cnt % 50000 == 0:
                        print("[" + str(cnt) + "] ", end='')
                    cnt=cnt+1
            
            print("\n")
            print("- Generating pandas dataframes: ")
            keys = evtsdict.keys()
            for key in keys:
                # Convert json string to pandas DF
                nevtskey = len(evtsdict[key])
                print('  + %-45s ... ' % (key), end='')
                df = pd.DataFrame(evtsdict[key]).T

                dfs[key] = df

                print(' [%s]' % (str(nevtskey)))

        # Harmonize - - - - - - - - - - - - - - - - - - - - - - - - - -
        if do_harmonize :
            print(" ")
            print("- Harmonizing pandas dataframes: ")
            keys = dfs.keys()
            for key in keys:
                print('  + %-45s ... ' % (key), end='')
                if  len(dfs[key]) > 0:
                    dfs[key] = harmonize(dfs[key], tool=tool, plugin=key, hostname=hostname)
                    print(' [OK]')
                else:
                    print(' [EMPTY]')

            if build_all_df :
                dfs['all'] = pd.concat([dfs['all'], dfs[key]], ignore_index=True)

        print("- Done Harmonizing.\n")

        # Save to pickle - - - - - - - - - - - - - - - - - - - - - - - - - -
        d4com.save_pickle(evdl, dfs, use_pickle, do_harmonize)
        

    print("\nNOTE: Now you can use the syntax <yourvar>['<datatype>'] to access your specific dataframe")

    return dfs

# HARMONIZATION FUNCTIONS #####################################################

# Tool Output Harmonization -> HTM
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

    # Specific Harmonization Pre-Processing =================================== 

    # Name index as pevtnum (plaso evt number) and reset it (as a std col)
    df.index = df.index.set_names("pevtnum")
    df = df.reset_index()

    # Resort columns
    cols = df.columns.tolist()
    if 'message' in df.columns:
        cols.insert(len(cols), cols.pop(cols.index('message')))
    cols.insert(len(cols), cols.pop(cols.index('pevtnum')))
    cols.insert(0, cols.pop(cols.index('timestamp_desc')))
    df = df.reindex(columns=cols)

    df['timestamp_desc'] = df['timestamp_desc'].str.replace(' Time$','').str.replace('Modification','Modif.')

    # Generic Harmonization ===================================================
    df = d4com.harmonize_common(df, **kwargs)

    # Specific Harmonization Post-Processing ==================================

    # Convert timestamp to datetime
    nowus = int(time.time()*1000000)
    df.insert(5, 'Timestamp_', None)
    df['Timestamp_'] = df['timestamp']
    df['Timestamp_'] = df['Timestamp_'].mask(df['Timestamp_'].lt(0),0)
    df['Timestamp_'] = df['Timestamp_'].mask(df['Timestamp_'].gt(nowus),0)
    df['Timestamp_'] = pd.to_datetime(df['Timestamp_'], unit="us")

    # Resort columns
    cols = df.columns.tolist()

    # return ==================================================================
    return df

def harmonize_plaso_artifact_common(df):
    """ Common part of the proccess Convert plaso DF in HAM format

        Args: 
            df (pandas.DataFrame): DF to harmonize
            kwargs(dict): harmonize options
        Returns: 
            pandas.DataFrame in HAM Format
    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")
    cols = ['timestamp_desc', '__container_type__', '__type__', 'data_type', 'inode', 'parser', 'pathspec', 'sha256_hash', 'message', 'pevtnum'] 
    df = df.drop(columns=[col for col in cols if col in df.columns])
    return df

# ARTIFACT EXTRACTION FUNCTIONS ###############################################

# Index:
# - evtx
# - windows_registry_key_value
# - windows_registry_service
# - amcache 
#   + windows_registry_amcache 
#   + windows_registry_amcache_programs

# evtx ========================================================================
def plaso_extract_single_evtx(plevtxdf, evtxf, nwf=d4.main_nwf, recovered=False, save_xml=False, xml_filename=""):

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if recovered :
        thisplevtxdf = plevtxdf.query('LogFile == @evtxf')
    else:
        thisplevtxdf = plevtxdf.query('LogFile == @evtxf and recovered == False')

    # Extract xml_string field
    print("    - Extracting & concatenating plaso xml_string entries")
    xmlstr = thisplevtxdf['xml_string'].str.cat()
    xmlstr = xmlstr.replace('\\n',' ')

    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # nwf - XML not well-formed
    if nwf != "":
        print("    - Escaping/fixing not well-formed records")
        for rule in nwf:
            next = False
            
            if "file" in rule:
                if rule.get('file') != evtxf:
                    next = True
                else:
                    print("  Rule for this evtx: %s" % evtxf)
            
            if next == False:
                thisstart = len(xmlstr)
                if rule.get('type') == "re":
                    xmlstr = re.sub(rule.get('find'), rule.get('replace'), xmlstr)
                else:
                    xmlstr = xmlstr.replace(rule.get('find'),rule.get('replace'))

                if thisstart != len(xmlstr):
                    print("      + Rule executed: %s" % rule.get('replace'))
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    xmlstr='<Events>'+xmlstr+'</Events>'

    if save_xml :
        if xml_filename == "":
            xml_filename = "plaso_extracted_evtx_events.xml"

        print("    - Saving xml to file:")
        print("        "+xml_filename)

        xmlfile = open(xml_filename, "w")
        xmlfile.write(xmlstr)
        xmlfile.close()

    # We no longer need plevtxdf. Let's clear it to save memory
    plevtxdf = pd.DataFrame([])

    try:
        print("    - Converting XML -> DF")
        outdf = d4utl.xml_to_df(xmlstr, sep=" > ").d4evtx.column_types_set()
        print("")
    except:
        print("    - ERROR: Could not convert XML -> DF")
        outdf = pd.DataFrame()
        print("")

    return outdf

def plaso_extract_evtx(plobj, nwf=d4.main_nwf, recovered=False, save_xml=False, xml_filename=""):
    # Input can be either a full plaso DFs dict or a DF of plaso evtx

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    evtxdfs={}

    if isinstance(plobj, dict):
        plevtxdf = plobj['windows_evtx_record']
    elif isinstance(plobj, pd.DataFrame):
        plevtxdf = plobj
    else:
        print("ERROR: Unsupported Input Data")
        return
 
    # Remove the path / preserve the file name only
    # Windows path (if plaso has been run on a disk image or live Windows System)
    plevtxdf['LogFile'] = plevtxdf['pathspec'].apply(lambda x: x['location']).str.replace(".*\\\\","").copy()
    # Linux path (if plaso has been run on a Linux System, e.g. on extracted Windows artifacts)
    plevtxdf['LogFile'] = plevtxdf['pathspec'].apply(lambda x: x['location']).str.replace(".*/","").copy()

    #plevtxdf['LogFile'] = plevtxdf['pathspec'].apply(lambda x: "["+str(x['parent']['location'])+"]"+str(x['location']))
    evtxfs = plevtxdf['LogFile'].drop_duplicates() 
    #evtxfs=plevtxdf['pathspec'].apply(lambda x: "["+str(x['parent']['location'])+"]"+str(x['location'])).str.replace("/","",1).drop_duplicates().str.replace(".*\\\\","")

    # Event Log files present in the plaso evtx DF
    #evtxfs=plevtxdf['pathspec'].apply(lambda x: "["+str(x['parent']['location'])+"]"+str(x['location'])).str.replace("/","",1).drop_duplicates().str.replace(".*\\\\","")

    # Include the "recovered" column from plaso
    plevtxdf['recovered'] = plevtxdf['recovered'].astype(bool)

    # DICT OF INDIVIDUAL DFs (one per evtx log file) - - - - - - - - - - - - - - - - - - - 
    print("- Processing:")
    for evtxf in evtxfs:
        evtxfbase = re.sub('^.*\\\\', '', evtxf)
        print("  + "+evtxfbase)
        xml_filename_file=None
        # XML Filename
        if save_xml :
            xml_filename_base = re.sub(r"\.xml$", "", xml_filename)
            xml_filename_file = xml_filename_base + "-" + evtxfbase + ".xml"

        evtxdf = plaso_extract_single_evtx(plevtxdf, evtxf, nwf, recovered, save_xml=save_xml, xml_filename=xml_filename_file)

        orchestrator = plevtxdf['D4_Orchestrator_'].iloc[0]
        tool         = plevtxdf['D4_Tool_'].iloc[0]
        plugin       = plevtxdf['D4_Plugin_'].iloc[0]
        hostname     = plevtxdf['D4_Hostname_'].iloc[0]
        datatype     = "evtx-raw"
        
        if not 'evtxFileName_' in evtxdf.columns:
            evtxdf.insert(0, 'evtxFileName_', evtxfbase)

        # Insert Tool / Data related HAM Columns
        if not 'D4_Hostname_' in evtxdf.columns:
            evtxdf.insert(0, 'D4_Hostname_', hostname)
        if not 'D4_Plugin_' in evtxdf.columns:
            evtxdf.insert(0, 'D4_Plugin_', plugin)
        if not 'D4_Tool_' in evtxdf.columns:
            evtxdf.insert(0, 'D4_Tool_', tool)
        if not 'D4_Orchestrator_' in evtxdf.columns:
            evtxdf.insert(0, 'D4_Orchestrator_', orchestrator)
        if not 'D4_DataType_' in evtxdf.columns:
            evtxdf.insert(0, 'D4_DataType_', datatype)

        evtxdfs[evtxfbase] = evtxdf

    print("- Done")

    return evtxdfs

def plaso_get_evtxdfs(pldfs, hostname, datapath="", notebook_file="", evtxdfs_usepickle=True, save_xml=False, xml_filename=""):

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    print("Extracting evtx DFs dict from plaso DFs dict.")

    if notebook_file != "":
        
        print('- Looking for notebook cell containing pickle file to read ('+hostname+'_evtxdfspklf)')
        hits = d4utl.nbgrep(notebook_file, "# "+hostname+'_evtxdfspklf_f2read = "/.*$')

        if hits:
            f2read = hits[0].split(" = ")[1].strip('"')
            print("  + Found: "+f2read)
        else:
            print("  + Not found")
            f2read = ""
    else:
        f2read = ""
        
    if f2read != "":
        pklinnb = True
        evtxdfspklf = f2read
    else:
        pklinnb = False
        # If we don't find the cell with the saved path for evtxdfspklf,
        # we will try to find the file in the datapath directory
        if datapath != "":
            evtxdfspklf = datapath+'/'+hostname+'.evtxdfs.pickle'
            print("- Looking for evtxdfs pickle file in datapath directory:")
          
            if os.path.exists(evtxdfspklf):
                print("  + Found: "+evtxdfspklf)
            else:
                print("  + Not found.")

        else:
            evtxdfspklf = ""
            extract_records = True

    if evtxdfs_usepickle  and evtxdfspklf != "":
        if os.path.exists(evtxdfspklf):
            # Read from pickle
            print("- Reading evtx DFs dictionary (evtxdfs) from pickle file...")
            evtxdfs = pickle.load(open(evtxdfspklf,"rb"))
            print("- Done.")
            extract_records = False
        else:
            if pklinnb :
                print("- WARNING: pickle file not found but notebook cell references it.")
                print("-          Remove from the notebook the cell that contains the "+hostname+"_evtxdfspklf variable definition")
            extract_records = True
    else:
        extract_records = True

    if extract_records :
        # plevtxdf = pldfs['windows_evtx_record']
        print("- Extracting records...")
        evtxdfs = plaso_extract_evtx(pldfs, save_xml=save_xml, xml_filename=xml_filename)

        if len(evtxdfs) == 0:
            print("ERROR: No records read.")
            return
        
        # Save to pickle
        if evtxdfs_usepickle :
            if datapath == "":
                print("WARNING: data_path argument not provided. Cannot save pickle file.")
            else:
                print("- Saving evtx DFs dictionary (dfs) as pickle file:")
                print("   "+evtxdfspklf)

                if os.path.exists(datapath):
                    # Save to pickle - - - - - - - - - - - - - - - - - - - - - - - - - -
                    pickle.dump(evtxdfs, open(evtxdfspklf, "wb" ))
                    print("- Done.")

                savepathtonb = True
                if savepathtonb :
                    cell_a = "# Automatically created - DO NOT EDIT OR REMOVE unless you want to change the file to read (in that case, remove this cell)\n"
                    cell_b = "# "+str(datetime.datetime.now())+"\n"
                    cell_c = "# "+hostname+'_evtxdfspklf_f2read = "'+evtxdfspklf+'"'
                    cell   = cell_a + cell_b + cell_c
                    get_ipython().set_next_input(cell)

                    print("- pickle File path saved to next notebook cell")
                    print("")

                else:
                    print("- ERROR: Directory does not exist:")
                    print("         "+datapath)

    return evtxdfs

# windows_registry_key_value ==================================================
def plaso_extract_win_reg_kv(plobj):
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Input can be either a full plaso DFs dict or a DF of plaso evtx

    d4datatype = "winreg_kv"
    pldatatype = "windows:registry:key_value"

    print("- Extracting plaso "+pldatatype+" events (d4: "+d4datatype+")")
    dfs={}

    if isinstance(plobj, dict):
        pldf = plobj['windows_registry_key_value']
    elif isinstance(plobj, pd.DataFrame):
        pldf = plobj
    else:
        print("ERROR: Unsupported Input Data")
        return

    df = pldf.copy().query('data_type == @pldatatype', engine="python")

    df['D4_DataType_'] = d4datatype

    # Explode values entries into multiple rows (one per value) ---------------

    # Harmonization -----------------------------------------------------------
    print("- Harmonizing")
    df = harmonize_plaso_artifact_common(df)
    df = df.drop(columns=['timestamp', 'date_time'])

    df = df.rename(columns={'Timestamp_': 'KeyLastWriteTimestamp_'})

    df['KeyLastWriteDate_'] = df['KeyLastWriteTimestamp_'].dt.date
    df['KeyLastWriteTime_'] = df['KeyLastWriteTimestamp_'].dt.ceil('s').dt.time

    df['KeyPath_']      = df['key_path']
    df['KeyPath-Hash_'] = df['key_path'].str.lower().apply(hash)
    df['Hive_']         = df['key_path'].str.replace('^(HKEY_LOCAL_MACHINE[^\\\\]*\\\\[^\\\\]*)\\\\.*','\\1')
    df['KeyRelPath_']   = df['key_path'].str.replace('^[^\\\\]*\\\\[^\\\\]*\\\\','')
    df['Values_']       = df['values'].str.replace('^[^\\\\]*\\\\[^\\\\]*\\\\','')
    
    df = df.drop(columns=['key_path', 'values'])

    # Return ------------------------------------------------------------------
    dfs[d4datatype] = df

    print("- Done")

    return dfs

# services - windows_registry_service =========================================
def plaso_extract_win_reg_service(plobj):
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Input can be either a full plaso DFs dict or a DF of plaso evtx

    d4datatype = "winservices"
    pldatatype = "windows:registry:service"

    print("- Extracting plaso "+pldatatype+" events (d4: "+d4datatype+")")
    dfs={}

    if isinstance(plobj, dict):
        pldf = plobj[pldatatype]
    elif isinstance(plobj, pd.DataFrame):
        pldf = plobj
    else:
        print("ERROR: Unsupported Input Data")
        return

    df = pldf.copy().query('data_type == @pldatatype', engine="python")

    df['D4_DataType_'] = d4datatype


    # Harmonization -----------------------------------------------------------
    print("- Harmonizing")
    df = harmonize_plaso_artifact_common(df)
    df = df.drop(columns=['timestamp', 'date_time'])

    #df = df.rename(columns={'Timestamp_': 'KeyLastWriteTimestamp_'})

    df['Date_'] = df['Timestamp_'].dt.date
    df['Time_'] = df['Timestamp_'].dt.ceil('s').dt.time

    df['Name_']           = df['name']
    df['ServiceType_']    = df['service_type']
    df['StartType_']      = df['start_type']
    df['ServiceDLL_']     = df['service_dll']
    df['ObjectName_']     = df['object_name']
    df['ImagePath_']      = df['image_path']
    df['ErrorControl_']   = df['error_control']

    # Common Fields
    df['Group']              = 'd4_niy'
    df['Tag']                = 'd4_niy'
    df['DependOnGroup']      = 'd4_niy'
    df['DependOnService']    = 'd4_niy'
    df['Description']        = 'd4_niy'
    df['DisplayName']        = 'd4_niy'
    df['DriverDisplayName']  = 'd4_niy'
    df['FailureActions']     = 'd4_niy'
    # Specific Fields
    
    df = df.drop(columns=['name', 'service_type', 'start_type', 'service_dll', 'object_name', 'image_path', 'error_control', 'key_path', 'values'])

    # Return ------------------------------------------------------------------
    dfs[d4datatype] = df

    print("- Done")

    return dfs

# amcache =====================================================================

# amcache - windows_registry_amcache =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plaso_extract_win_reg_amcache(plobj):
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Input can be either a full plaso DFs dict or a DF of plaso evtx

    d4datatype = "amcache"
    pldatatype = "windows:registry:amcache"

    print("- Extracting plaso "+pldatatype+" events (d4: "+d4datatype+")")
    dfs={}

    if isinstance(plobj, dict):
        pldf = plobj[pldatatype]
    elif isinstance(plobj, pd.DataFrame):
        pldf = plobj
    else:
        print("ERROR: Unsupported Input Data")
        return

    df = pldf.copy().query('data_type == @pldatatype', engine="python")

    df['D4_DataType_'] = d4datatype


    # Harmonization -----------------------------------------------------------
    print("- Harmonizing")
    df = harmonize_plaso_artifact_common(df)
    df = df.drop(columns=['timestamp', 'date_time'])

    #df = df.rename(columns={'Timestamp_': 'KeyLastWriteTimestamp_'})

    df['Date_'] = df['Timestamp_'].dt.date
    df['Time_'] = df['Timestamp_'].dt.ceil('s').dt.time

    df['FileReference_']           = df['file_reference']
    df['FileSize_']                = df['file_size']
    df['FilePath_']                = df['full_path']
    df['LanguageCode_']            = df['language_code']
    df['ProgramIdentifier_']       = df['program_identifier']
    df['CompanyName_']             = df['company_name']
    df['FileDescription_']         = df['file_description']
    df['FileVersion_']             = df['file_version']
    df['ProductName_']             = df['product_name']
    df['SHA1_']                    = df['sha1']

    df = df.drop(columns=['file_reference', 'file_size', 'full_path', 'language_code', 'program_identifier', 'company_name', 'file_description', 'file_version', 'product_name', 'sha1'])

    # Return ------------------------------------------------------------------
    dfs[d4datatype] = df

    print("- Done")

    return dfs

# amcache - windows_registry_amcache_programs =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ANALYSIS ####################################################################

# simple ======================================================================
# def simple_func(df,key_path=False,message=False,pathspec=True,ret=False):
def simple_func(df, **kwargs):
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


    if d4.debug >= 4:
        print("DEBUG: [pl] [simple_func()]")

    # Variables ----------------------------------------------------------------
    hiddencolscommon =  ['__container_type__', '__type__', 'data_type', 'inode', 'parser', 'pevtnum', 'message', 'sha256_hash', 'pathspec']
    hiddencolsdf = pd.DataFrame([])

    # Maximum number of lines in DF for beautification
    maxdfbprintlines = 1000

    # Arg. Parsing -------------------------------------------------------------
    collapse_constant_cols = kwargs.get('collapse_constant_cols', True)
    hide_cols              = kwargs.get('hide_cols',              True)
    wrap_cols              = kwargs.get('wrap_cols',              True)
    beautify_cols          = kwargs.get('beautify_cols',          False)

    # Show whatever selected on the screen or don't show anything   
    out                    = kwargs.get('out',                    True)
    # Show the resulting DF on the screen
    out_df                 = kwargs.get('out_df',                 True)
    # Return the resulting DataFrame
    ret                    = kwargs.get('ret',                    False)
    # Return the resulting DataFrame as shown in the screen
    # (without hidden cols, collapsed cols, etc.; not in full)
    ret_out                = kwargs.get('ret_out',                False)

    # Check if out is set (will be needed later)
    outargset = 'out' in kwargs.keys()

    # ret_out implies ret
    if 'ret_out' in kwargs.keys() and ret_out:
        ret = True

    # ret implies no out, unless out is set specifically
    if ( ('ret' in kwargs.keys() or 'ret_out' in kwargs.keys()) or 'ret_out' in kwargs.keys()) and not outargset and ret:
        # If ret is set to True by the user we will not provide stdout output,
        # unless the user has specifically set out = True
        out = False

    # Health Check -------------------------------------------------------------
    dfnrows = len(df)

    if dfnrows == 0:
        print("ERROR: Empty DataFrame.")
        return

    # Artifact / Tool specific processing --------------------------------------
    # df.index = df.index.ceil(freq='s')  

    hiddencols = hiddencolscommon
    # pathspec simplification
    if 'pathspec' in df.columns:
        # PREFETCH - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if   df['parser'].iloc[0] in "prefetch":
            pathspec_simple = df['pathspec'].apply(lambda x: str(x['location']))
            df['pathspec_simple_'] = pathspec_simple
            hiddencols.append('mapped_files')
        # OTHER  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        elif df['parser'].iloc[0] not in "filestat":
            # Keep a simplified pathspec for certain artifacts
            if 'parent' in df['pathspec'].iloc[0].keys():
                pathspec_simple = df['pathspec'].apply(lambda x: "["+str(x['parent']['location'])+"]"+str(x['location']) if 'location' in x['parent'].keys() else "-").str.replace("/","",1)
                df['pathspec_simple_'] = pathspec_simple

    # Var. Init. ---------------------------------------------------------------
    dfb = df

    # Hidden Columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if dfb['D4_Plugin_'].head(1).values != "windows_registry_key_value":
         hiddencols.append('key_path')

    if hide_cols :
        for col in  hiddencols:
           if col in dfb.columns:
               hiddencolsdf = pd.concat([hiddencolsdf, pd.DataFrame([col])], ignore_index=True)
               dfb = dfb.drop(columns=[col])

    # Constant Columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if collapse_constant_cols :
        concolsdf, dfb = d4utl.collapse_constant_columns(dfb)

    # Wrap wide columns  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if wrap_cols :
        for col in dfb.columns:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.wrap(100)
                beautify_cols = True

    # DISPLAY ==========================================================
    nhiddencols = len(hiddencolsdf)

    if out :
        print("")
        display(Markdown("**Statistics:**\n<br>No. Entries: "+str(dfnrows)))

        # Hidden / Constant Columns  - - - - - - - - - - - - - - - - - - - - - - - -
        if collapse_constant_cols :
            show_constant_cols = True
        else:
            show_constant_cols = False

        if hide_cols  and nhiddencols != 0:
            show_hidden_cols = True
        else:
            show_hidden_cols = False

        if show_constant_cols  and show_hidden_cols :
            d4utl.display_side_by_side([hiddencolsdf, concolsdf], ['HIDDEN COLUMNS', 'CONSTANT COLUMNS'])
        elif show_constant_cols  and show_hidden_cols == False:
            display(Markdown("**Constant Columns**"))
            max_rows = pd.get_option("display.max_rows")
            pd.set_option("display.max_rows", None)
            display(concolsdf)
            pd.set_option("display.max_rows", max_rows)
        elif show_constant_cols == False and show_hidden_cols :
            display(Markdown("**Hidden Columns**"))
            display(hiddencolsdf)

        # Beautify columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if len(dfb) >= maxdfbprintlines:
            print('WARNING: Too many lines (>'+str(maxdfbprintlines)+') in DataFrame for beautification. Returning non-beautified output.')
            beautify_cols = False

        if out_df :
            if beautify_cols :
                if dfnrows == 1:
                    display(dfb.T.reset_index().style.set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap' }))
                else:
                    display(dfb.reset_index().style.set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap' }))
            else:
                if dfnrows == 1:
                    display(dfb.T)
                else:
                    display(dfb)

    # RETURN  ==========================================================
    if ret :
        if ret_out :
            return dfb
        else:
            return df
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plaso original 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #key_path = kwargs.get('key_path',False)
    #message  = kwargs.get('message',False)
    #pathspec = kwargs.get('pathspec',True)


def get_source_options():
    return ['wrap_cols', 'beautify_cols']

# analysis ====================================================================
def analysis(*args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    if d4.debug >= 3:
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
        d4list("str-list")
        return

    def d4list(objtype):
        print("Available plaso analysis types:")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-plaso", objtype):
            print("- plaso_categories:  No.events & first/last event per plaso category (Input: pldfs)")
        if objtype == None or objtype == "str-help" or objtype == "str-list" or  re.search("^dict-pandas_dataframe-plaso", objtype):
            print("- plaso_overview: Overview (Input: pldfs)")

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

    # pldfs ------------------------------------------------------------------
    if re.search("^dict-pandas_dataframe-plaso", objtype):
        if anltype == "plaso_categories":
            return analysis_plaso_categories(*args, **kwargs)
        if anltype == "plaso_overview":
            return analysis_plaso_overview(*args, **kwargs)

    print("INFO: [d4pl] No analysis functions available for this data type ("+objtype+")")

def analysis_plaso_categories(*args, **kwargs):
    """ Analysis that gives plaso categories

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    dfs = args[0]

    objtype = d4com.data_identify(dfs)

    if objtype != "dict-pandas_dataframe-plaso-raw":
        print("ERROR: Invalid object for function: "+objtype)
        print("       Input object should be:      dict-pandas_dataframe-plaso")
        return

    outdf = pd.DataFrame([],columns=['Category','NEvts','TSMin','TSMax'])
    row = pd.Series()

    keys = list(dfs.keys())
    keys.sort()

    for key in keys:
        if len(dfs[key]) != 0:
            row['TSMin'] = dfs[key]['Timestamp_'].min()
            row['TSMax'] = dfs[key]['Timestamp_'].max()
        else:
            row['TSMin'] = ""
            row['TSMax'] = ""

        row['Category'] = key
        row['NEvts'] = len(dfs[key])

        outdf = outdf.append(row,ignore_index=True)

    return outdf

def analysis_plaso_overview(*args, **kwargs):
    """ Analysis that gives general device information

        Args: 
        obj:          Input data (typically DF or dict of DFs)
        Returns: 
        pandas.Dataframe with the results of the analysis

    """    
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    dfs = args[0]
    
    objtype = d4com.data_identify(dfs)
    
    if objtype != "dict-pandas_dataframe-plaso-raw":
        print("ERROR: Invalid object for function: "+objtype)
        print("       Input object should be:      dict-pandas_dataframe-plaso")
        return
    
    df = pd.DataFrame()
    overview_df = pd.DataFrame()
    users_df = pd.DataFrame()
    
    for key in dfs:
        if key == 'windows_registry_installation':
            df = dfs[key]
            overview_df['Product Name'] = df['product_name']
            overview_df['Installation Date'] = df['Timestamp_'].iloc[1]
            
        if key == 'windows_registry_timezone':
            df = dfs[key]
            overview_df['Timezone'] = df['configuration']
        
        if key == 'windows_registry_sam_users':
            df = dfs[key]
            users_df['User Name'] = df['username']
            users_df['First Connection'] = df['Timestamp_']
            
    outdf = overview_df.iloc[0].to_frame()
    
    outdf['System Information'] = outdf[0]
    
    outdf = outdf.drop(columns=[0,])
    
    display(outdf)
    
    return users_df
    
    

# DATAFRAME ACCESSORS =========================================================

@pd.api.extensions.register_dataframe_accessor("d4pl")
class Ds4n6PlAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df=self._obj

        return simple_func(df, **kwargs)

    def regfindraw(self,str,full=False):
        df=self._obj
        if full :
            return df.fillna("null_value").query('values.str.contains(@str)',engine="python")
        else:
            return df.fillna("null_value").query('values.str.contains(@str)',engine="python").d4plaso.simple().drop(columns=['values','timestamp_desc'])

    def regfind(self,searchstr):
        df=pd.DataFrame(self.regfindraw(searchstr,full=True)['message'].apply(lambda x: re.sub(r"([A-Za-z][A-Za-z]*:) (\[[A-Z_][A-Z_]*\])","-=- \\1 | \\2 | ",x)))
        html0=df.to_html()
        html1=re.sub("</td>","</td></tr></table></td>",re.sub("<td>","<td><table><tr><td>",html0))
        html2=re.sub("] -=- ","]</td><td></td><td></td></tr><tr><td>",html1)
        html3=re.sub(" -=- ","</td></tr><tr><td>",html2)
        html4=re.sub(r"\|","</td><td>",html3)
        html=html4
        display(HTML(html))

@pd.api.extensions.register_dataframe_accessor("d4_plaso")
class Ds4n6PlasoAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def simple(self, *args, **kwargs):
        """ Redirects execution to simple_func()
        """
        df=self._obj

        return simple_func(df, **kwargs)

    def regfindraw(self, str, full=False):

        df = self._obj

        if full :
            return df.fillna("null_value").query('values.str.contains(@str)',engine="python")
        else:
            return df.fillna("null_value").query('values.str.contains(@str)',engine="python").d4plaso.simple().drop(columns=['values','timestamp_desc'])

    def regfind(self, searchstr):
        df=pd.DataFrame(self.regfindraw(searchstr,full=True)['message'].apply(lambda x: re.sub(r"([A-Za-z][A-Za-z]*:) (\[[A-Z_][A-Z_]*\])","-=- \\1 | \\2 | ",x)))
        html0 = df.to_html()
        html1 = re.sub("</td>","</td></tr></table></td>",re.sub("<td>","<td><table><tr><td>",html0))
        html2 = re.sub("] -=- ","]</td><td></td><td></td></tr><tr><td>",html1)
        html3 = re.sub(" -=- ","</td></tr><tr><td>",html2)
        html4 = re.sub(r"\|","</td><td>",html3)

        html = html4

        display(HTML(html))

