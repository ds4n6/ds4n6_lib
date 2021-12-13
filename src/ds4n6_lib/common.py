###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import glob
import re
import inspect

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
from IPython.display import display, Markdown, HTML
import pickle
#---
# DS4N6 IMPORTS ---------------------------------------------------------------

# Common  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import ds4n6_lib.d4          as d4
import ds4n6_lib.gui         as d4_gui
import ds4n6_lib.unx         as d4_unx
import ds4n6_lib.utils       as d4_utils
# Tools - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import ds4n6_lib.autoruns    as d4_autoruns
import ds4n6_lib.evtx_parser as d4_evtx_parser
import ds4n6_lib.kansa       as d4_kansa
import ds4n6_lib.kape        as d4_kape
import ds4n6_lib.macrobber   as d4_macrobber
import ds4n6_lib.mactime     as d4_mactime
import ds4n6_lib.plaso       as d4_plaso
import ds4n6_lib.volatility  as d4_volatility
import ds4n6_lib.tshark      as d4_tshark
# Artifacts - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import ds4n6_lib.evtx        as d4_evtx
import ds4n6_lib.flist       as d4_flist
import ds4n6_lib.fstl        as d4_fstl
import ds4n6_lib.pslist      as d4_pslist
import ds4n6_lib.svclist     as d4_svclist
import ds4n6_lib.winreg      as d4_winreg
import ds4n6_lib.amcache     as d4_amcache
# ML  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import ds4n6_lib.ml     as d4_ml
###############################################################################
# VARIABLES
###############################################################################

# All ds4n6_lib libs
d4libs=['d4_autoruns', 'd4_evtx', 'd4_evtx_parser', 'd4_fstl', 'd4_gui', 'd4_kansa', 'd4_kape', 'd4_plaso', 'd4_unx', 'd4_utils', 'd4_winreg', 'd4_volatility', 'd4_pslist', 'd4_svclist', 'd4_tshark']
# Analysis-oriented ds4n6_libs (with analysis(), etc.)
d4anllibs=['d4_autoruns', 'd4_evtx', 'd4_fstl', 'd4_kansa', 'd4_kape', 'd4_plaso', 'd4_volatility', 'd4_pslist', 'd4_svclist']

###############################################################################
# FUNCTIONS
###############################################################################

# FILE READING FUNCTIONS ######################################################
def read_data_common(evdl, **kwargs):
    """ Read data from files or a folder

        Args: 
            evdl (str): path to file/folder source
            kwargs: read options
        Returns: 
            pandas.Dataframe or dictionary of pandas.DataFrame
    """
    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # Parse Arguments =========================================================

    # Tool Args ---------------------------------------------------------------
    tool                     = kwargs.get('tool',                    '')
    # Path & File Args --------------------------------------------------------
    path_filter              = kwargs.get('path_filter',             '')
    prefix                   = kwargs.get('prefix',                  '')
    suffix                   = kwargs.get('suffix',                  '')
    ext                      = kwargs.get('ext',                     '.csv')
    header_names             = kwargs.get('header_names',            None)
    folder_parsing_mode      = kwargs.get('folder_parsing_mode',     'generic')
    path_prefix              = kwargs.get('path_prefix',             '')
    maxdepth                 = kwargs.get('maxdepth',                4)
    # pluginisdfname -> Each file corresponds to the output of a plugin
    # The D4_plugin field will be populated w/ the file name (minus prefix/ext)
    pluginisdfname            = kwargs.get('pluginisdfname',           False)     
    # Hostname Args -----------------------------------------------------------
    hostname                 = kwargs.get('hostname',                None)
    get_hostname_from_folder = kwargs.get('get_hostname_from_folder', False)
    get_hostname_from_file   = kwargs.get('get_hostname_from_file',   False)
    # Data Args ---------------------------------------------------------------
    separator                = kwargs.get('separator',               ',')
    encoding                 = kwargs.get('encoding',                'utf-8')
    filetype                 = kwargs.get('filetype',                'csv')
    # Harmonization Args ------------------------------------------------------
    do_harmonize             = kwargs.get('harmonize',               True)
    # Output Args -------------------------------------------------------------
    merge_files              = kwargs.get('merge_files'             , False)
    build_all_df             = kwargs.get('build_all_df'            , False)
    # Save Args ---------------------------------------------------------------
    use_pickle               = kwargs.get('use_pickle'              , True)

    if not os.path.exists(evdl):
        print("")
        print("ERROR: File/Folder does not exist.")
        print("         "+evdl)
        return

    # Generic =================================================================
    # Pickle file
    pklrawf = evdl+'.raw.pkl' # Raw
    pklhamf = evdl+'.ham.pkl' # Harmonized

    if do_harmonize:
        pklf = pklhamf
    else:
        pklf = pklrawf
    
    if   tool == "kansa" and folder_parsing_mode != "multiple_hosts_in_single_filetype_folder":
        print("")
        print('WARNING: kansa data reading typically needs the following argument:')
        print('         folder_parsing_mode="multiple_hosts_in_single_filetype_folder"')
        print("")
    elif tool == "kape" and folder_parsing_mode != "single_host_with_categories":
        print("")
        print('WARNING: kape data reading typically needs the following argument:')
        print('         folder_parsing_mode="single_host_with_categories"')
        print("")
       

    #==========================================================================
    # There are 4 parsing modes:
    # - generic
    # - multiple_hosts_one_per_folder
    # - multiple_hosts_in_single_filetype_folder  -> kansa
    # - single_host_with_categories               -> kape
    #
    #==========================================================================
    # generic parsing
    if folder_parsing_mode == 'generic':

        if d4.debug >= 2:
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] folder_parsing_mode -> generic")

        # If pickle exists, read from pickle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if os.path.exists(pklf) and use_pickle:
    
            # Read from pickle 
            print("- Saved pickle file found:")
            print("      "+pklf)
            print("- Reading data from pickle file.")
            obj = pickle.load(open(pklf, "rb"))
            print("- Done.")

            return obj
    
        # If pickle does not exist, read original data ~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # If evdl is a folder -> Read all csv files in that folder
            # If evdl is a file -> Read file
            # Read Directory --------------------------------------------------
            if os.path.isdir(evdl):
                if d4.debug >= 3:
                    print("DEBUG: [com-read_data_common()] [folder_parsing_mode: generic] -> Directory")

                if merge_files:
                    dfall = pd.DataFrame([])
                else:
                    dfs = {}

                    if build_all_df:
                        dfs['all'] = pd.DataFrame([])
                files = d4_utils.find_files_in_folder(evdl, path_filter, ext, maxdepth=maxdepth)

                nfiles2read = len(files)

                if nfiles2read == 0:
                    print('  + WARNING: No files found with extension "' + ext + '"')
                else:
                    print('  + INFO: {} files found to read.\n'.format(nfiles2read))

                    cnt = 1
                    for f in files:
                        fbase = os.path.basename(f)
                        frel  = re.sub("^"+evdl+"/", "", f)

                        if get_hostname_from_folder:
                            dfname = re.sub("/.*", "", frel)
                        else:
                            dfname = fbase
                            if ext != '':
                                dfname = dfname.replace(ext, '')

                        if prefix != '':
                            dfname = re.sub('^'+prefix, '', dfname)

                        if suffix != '':
                            dfname = re.sub(suffix+'$', '', dfname)

                        if get_hostname_from_folder:
                            hostname = dfname
                            kwargs['hostname']=hostname

                        if get_hostname_from_file:
                            hostname = f
                            hostname = re.sub(".*/", "", hostname)
                            hostname = re.sub(r"\.[^\.]*$", "", hostname)

                        print('  + ['+str(cnt)+'/'+str(nfiles2read)+'] %-60s -> %-40s' % (frel, dfname), end='')

                        # Read File - - - - - - - - - - - - - - - - - - - - - - - - - -
                        if os.stat(f).st_size == 0:
                            print("[Empty File]")
                        else:
                            if re.search('.csv$', f, re.IGNORECASE) or re.search('.psv$', f, re.IGNORECASE) or filetype == "csv"  or filetype == "psv" :
                                if encoding == '':
                                    try:
                                        df = pd.read_csv(f, sep=separator, names=header_names)
                                        print('['+str(len(df))+']')
                                    except:
                                        df = pd.DataFrame([])
                                        print("[ERROR]")
                                else:
                                    try:
                                        df = pd.read_csv(f, encoding=encoding, names=header_names, sep=separator)
                                        print('['+str(len(df))+']')
                                    except:
                                        print("[ERROR]")
                                        df = pd.DataFrame([])
                            elif re.search('.evtx', f):
                                df = d4_evtx_parser.read_evtx_file(f)

                            # Fix column names
                            for col in df.columns:
                                # Remove leading / trailing spaces
                                colnew = col.strip()
                                if colnew != col:
                                    df = df.rename(columns={col: colnew})
                            # Harmonize - - - - - - - - - - - - - - - - - - - - - - - - - -
                            dftype = df.d4.df_source_identify()

                            if do_harmonize and len(df) > 0:
                                if pluginisdfname == True:
                                    df = harmonize(df, **kwargs, plugin=dfname)
                                else:
                                    df = harmonize(df, **kwargs)

                                if merge_files:
                                    dfall = pd.concat([dfall, df], ignore_index=True)
                                else:
                                    dfs[dfname] = df

                                    if build_all_df:
                                        dfs['all'] = pd.concat([dfs['all'], dfs[fbase]], ignore_index=True)

                        cnt += 1

                    # Convert D4_Hostname_ col to categorical
                    if merge_files:
                        if len(dfall) != 0:
                            if 'D4_Hostname_' in df.columns:
                                if not dfall['D4_Hostname_'].isna().any():
                                    dfall['D4_Hostname_'] = pd.Categorical(dfall['D4_Hostname_'], categories=dfall['D4_Hostname_'].drop_duplicates())
                    else:
                        if build_all_df:
                            if len(dfall) != 0:
                                if not dfs['all']['D4_Hostname_'].isna().any():
                                    dfs['all']['D4_Hostname_'] = pd.Categorical(dfs['all']['D4_Hostname_'], categories=dfs['all']['D4_Hostname_'].drop_duplicates())


                print("")

                # Save to pickle & return - - - - - - - - - - - - - - - - - - - - - - - 
                if merge_files:
                    save_pickle(evdl, dfall, use_pickle, do_harmonize)
                    return dfall
                else:
                    save_pickle(evdl, dfs, use_pickle, do_harmonize)
                    return dfs

            # Read File -------------------------------------------------------
            elif os.path.isfile(os.path.realpath(evdl)):
                if d4.debug >= 3:
                    print("DEBUG: [com-read_data_common()] [folder_parsing_mode: generic] -> File")

                print("- Reading tool file:")
                print("    "+evdl)

                df = pd.read_csv(evdl, encoding=encoding, names=header_names)
                df = df.fillna('')

                print("- No. lines: "+str(len(df)))

                # Harmonize - - - - - - - - - - - - - - - - - - - - - - - - - -
                dftype = df.d4.df_source_identify()

                
                if do_harmonize and len(df) > 0:
                    if dftype == "pandas_dataframe-"+tool+"-raw":
                        print("- Harmonizing to "+tool+" HAM...")
                        df = harmonize(df, **kwargs)

                # Save to pickle  - - - - - - - - - - - - - - - - - - - - - - - 
                save_pickle(evdl, df, use_pickle, do_harmonize)

                return df

    #==========================================================================
    # Folders contain category for multiple hosts (e.g. kansa) 
    elif folder_parsing_mode == 'multiple_hosts_one_per_folder':

        if d4.debug >= 3:
            print("DEBUG: [com-read_data_common()] folder_parsing_mode -> multiple_hosts_in_single_filetype_folder")


        # If pickle exists, read from pickle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if os.path.exists(pklf) and use_pickle:

            # Read from pickle
            print("- Saved pickle file found:")
            print("      "+pklf)
            print("- Reading data from pickle file.")
            obj = pickle.load(open(pklf, "rb"))
            print("- Done.")

            return obj

        # If pickle does not exist, read original data ~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            dfs = {}
            # Let's parse all host files into category-specific dataframes
            dirs = [x[0] for x in os.walk(evdl)]
            for dir in dirs:
                if dir != evdl:
                    catd = dir
                    cat = os.path.basename(catd)
                    dfs[cat] = pd.DataFrame()
                    print('  + Reading csv files for category %-20s into dataframe ->  %-20s' % (cat,cat))
                    hostfs = os.listdir(evdl + "/" + cat)
                    for hostf in hostfs:
                        hostffull = evdl + "/" + cat + '/' + hostf
                        hostname = str.replace(os.path.basename(hostffull), '-' + cat + '.csv', '')
                        try:
                            df = pd.read_csv(hostffull, encoding='utf16', names=header_names)
                        except:                
                            df = pd.DataFrame()    

                        dfs[cat] = pd.concat([dfs[cat], df], ignore_index=True)

                    # Harmonize - - - - - - - - - - - - - - - - - - - - - - - -
                    dftype = dfs[cat].d4.df_source_identify()

                    if do_harmonize  and len(dfs[cat]) > 0:
                        dfs[cat] = harmonize(dfs[cat], **kwargs)

            print("\n\nNOTE: Now you can use the syntax <yourvar>['Category'] to access your dataframe\n")

            # Save to pickle - - - - - - - - - - - - - - - - - - - - - - - - - -
            save_pickle(evdl, dfs, use_pickle, do_harmonize)

            # return  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            return dfs

    #==========================================================================
    # Folders contain category for multiple hosts (e.g. kansa) 
    elif folder_parsing_mode == 'multiple_hosts_in_single_filetype_folder':

        if d4.debug >= 3:
            print("DEBUG: [com-read_data_common()] folder_parsing_mode -> multiple_hosts_in_single_filetype_folder")


        # If pickle exists, read from pickle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if os.path.exists(pklf) and use_pickle:

            # Read from pickle
            print("- Saved pickle file found:")
            print("      "+pklf)
            print("- Reading data from pickle file.")
            obj = pickle.load(open(pklf, "rb"))
            print("- Done.")

            return obj

        # If pickle does not exist, read original data ~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            dfs = {}
            # Let's parse all host files into category-specific dataframes
            dirs = [x[0] for x in os.walk(evdl)]
            for dir in dirs:
                if dir != evdl:
                    catd = dir
                    cat = os.path.basename(catd)
                    dfs[cat] = pd.DataFrame()
                    print('  + Reading csv files for category %-20s into dataframe ->  %-20s' % (cat,cat))
                    hostfs = os.listdir(evdl + "/" + cat)
                    for hostf in hostfs:
                        hostffull = evdl + "/" + cat + '/' + hostf
                        hostname = str.replace(os.path.basename(hostffull), '-' + cat + '.csv', '')
                        try:
                            df = pd.read_csv(hostffull, encoding='utf16', names=header_names)
                        except:                
                            df = pd.DataFrame()    

                        dfs[cat] = pd.concat([dfs[cat], df], ignore_index=True)

                    # Harmonize - - - - - - - - - - - - - - - - - - - - - - - -
                    dftype = dfs[cat].d4.df_source_identify()

                    if do_harmonize  and len(dfs[cat]) > 0:
                        if pluginisdfname == True:
                            dfs[cat] = harmonize(dfs[cat], **kwargs, plugin=cat)
                        else:
                            dfs[cat] = harmonize(dfs[cat], **kwargs)

            print("\n\nNOTE: Now you can use the syntax <yourvar>['Category'] to access your dataframe\n")

            # Save to pickle - - - - - - - - - - - - - - - - - - - - - - - - - -
            save_pickle(evdl, dfs, use_pickle, do_harmonize)

            # return  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            return dfs

    #==========================================================================
    # Folders contain categories for a single host (e.g. kape) 
    elif folder_parsing_mode == 'single_host_with_categories':
    # WARNING: At this point this is very kape-specific.

        if d4.debug >= 3:
            print("DEBUG: [com-read_data_common()] folder_parsing_mode -> single_host_with_categories")

        # If pickle exists, read from pickle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if os.path.exists(pklf) and use_pickle:

            # Read from pickle
            print("- Saved pickle file found:")
            print("      "+pklf)
            print("- Reading data from pickle file.")
            obj = pickle.load(open(pklf, "rb"))
            print("- Done.")

            return obj

        # If pickle does not exist, read original data ~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            dfs = {}
            # Let's parse all files into category-specific dataframes
            for rootd, dirs, files in os.walk(evdl):
                filesuniq = set(files)
                for fname in filesuniq:
                    if rootd != evdl:
                        category = re.sub(evdl+'/', '', rootd)
                    else:
                        category = "-"

                    fnamebase, _fnameext = os.path.splitext(fname)
                    if ".csv" in fname:
                        artf = os.path.join(rootd, fname)

                        # Clean-up file name to obtain meaningful artifact description
                        artbase = fnamebase
                        artbase = re.sub('^[0-9][0-9]*_', '', artbase)
                        if path_prefix != "":
                            artbase = re.sub(path_prefix+'_', '', artbase)
                            path_prefix2 = re.sub('__', '_', path_prefix, 1)
                            artbase = re.sub('_'+path_prefix2+'_', '-', artbase)
                            artbase = re.sub('_Output$','', artbase)
                            if category != "-":
                                art = category + '-' + artbase
                            else:
                                art = artbase

                        if not art in dfs:                            
                            dfs[art] = pd.DataFrame()                        
                        else:
                            print("- File repeat: %s" % art)

                        print('  + Reading csv file %-110s into dataframe ->  %-100s \n' % (fname, art),end='')
                        try:
                            df = pd.read_csv(artf, names=header_names)
                        except:
                            df = pd.DataFrame()

                        # Harmonize ---------------------------------------------------------------
                        dftype = df.d4.df_source_identify()
                        if do_harmonize  and len(df) > 0:
                            if pluginisdfname == True:
                                df = harmonize(df, **kwargs, plugin=art)
                            else:
                                df = harmonize(df, **kwargs)
                        
                        dfs[art] = pd.concat([dfs[art], df],ignore_index=True)

            print("\n\nNOTE: Now you can use the syntax <yourvar>['Category'] to access your dataframe\n")

            # Save to pickle - - - - - - - - - - - - - - - - - - - - - - - - - -
            save_pickle(evdl, dfs, use_pickle, do_harmonize)

            # return  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            return dfs

    # Unknown parsing type ====================================================
    else:
        print("ERROR: Unknown folder parsing type.")


def save_pickle(evdl, dfs, use_pickle=True, do_harmonize=True):
    """Save a pickle
    Input:
        evdl: evidence path
        dfs:  dataframe to save
        use_pickle : True by default. Bypass value from last function.
        do_harmonize : True by default save as Harmonized, False save as RAW.
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    pklrawf = evdl+'.raw.pkl' # Raw
    pklhamf = evdl+'.ham.pkl' # Harmonized

    if not len(dfs) == 0:
        if use_pickle:
            if do_harmonize:
                if not os.path.exists(pklhamf):
                        # Save to pickle - HAM
                        print("- Saving Harmonized DataFrames dict (dfs) as pickle file:")
                        print("    "+pklhamf) 
                        pickle.dump(dfs, open(pklhamf, "wb"))
                        print("- Done.")
            else:
                if not os.path.exists(pklrawf):
                        # Save to pickle - RAW
                        print("- Saving Raw DataFrames dict (dfs) as pickle file:")
                        print("    "+pklrawf) 
                        pickle.dump(dfs, open(pklrawf, "wb"))
                        print("- Done.")
    
    
# DATA IDENTIFICATION #########################################################

def whatis(obj):
    """Same as data_identify.
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return data_identify(obj)

def data_identify(obj):
    """Return type of obj.
    Input:
        obj: object to identify
    Return:
        type of object (str, dict, df, or unknown.)
    """    
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if   isinstance(obj, str):
        return str_source_identify_func(obj)
    elif isinstance(obj, dict):
        return dict_source_identify_func(obj)
    elif isinstance(obj, pd.DataFrame):
        return df_source_identify_func(obj)
    else:
        return "unknown"

def str_source_identify_func(mystr):
    """ Indentify string option in analysis functions

        Args: 
            mystr (str)
        Returns: 
            string type
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if mystr == "":
        return "str-empty"
    elif mystr == "help":
        return "str-help"
    elif mystr == "list":
        return "str-list"
    else:
        return "str-unknown"

def dict_source_identify_func(dict):
    """ indentify type of source in a dictionary of DF

        Args: 
            dict (dict): dict of DF 
        Returns: 
            type of source

    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # We will use 2 methods in order to identify the dict:
    # - 1: Identify dict by the keys of the dictionary
    # - 2: Identify dict by the cols of the first DF in the dict
    # If method 1 fails then method 2 will be attempted

    # Identify dict by the cols of the first DF in the dict =============

    # NEW METHOD
    eltypes = []
    for key in dict.keys():
        if len(dict[key]) != 0:
            obj = dict[key]
            eltype = data_identify(obj)
            eltypes.append(eltype)

    # Identify the common substring
    eltype1   = eltypes[0]
    eltype1cs = eltype1.split("-")

    if d4.debug >= 3:
        display(eltype1)
        display(eltype1cs)
        display(eltypes)

    etcom = ""
    for et1c in eltype1cs:
        if etcom == "":
            etcom = et1c
        else:
            etcom = etcom + "-" + et1c

        allok = True
        for et in eltypes:
            if not re.match(etcom, et) and allok == True:
                allok = False
                
        if allok == True:
            etcomfinal = etcom

        if d4.debug >= 3:
            print(et1c)
            print(etcom)

    if allok == False:
        etcomfinal = etcomfinal + "-raw"

    if d4.debug >= 3:
        print("FINAL: "+etcomfinal)

    if d4.debug >= 4:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] eltype: "+eltype)

    if etcomfinal != "unknown":
        return 'dict-'+etcomfinal
    else:
        return "unknown"
    

def df_source_identify_func(df):
    """ Identify the type of a DF

        Args: 
            df (pandas.DataFrame): DF to indentify
        Returns: 
            str: type of DF
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    # autoruns --------------------------------------------------------------------
    autoruns_raw_df_cols = pd.Series(['Time', 'Entry Location', 'Entry', 'Enabled', 'Category', 'Profile', 'Description', 'Signer', 'Company', 'Image Path', 'Version', 'Launch String', 'MD5', 'SHA-1', 'PESHA-1', 'PESHA-256', 'SHA-256', 'IMP'])
    autoruns_ham_df_cols = pd.Series(['EntryLocation', 'Entry', 'Enabled', 'Category', 'Profile', 'Description', 'Signer', 'Company', 'ImagePath', 'Version', 'LaunchString', 'MD5', 'SHA-1', 'PESHA-1', 'PESHA-256', 'SHA-256', 'IMP'])
    # fstl ------------------------------------------------------------------------
    mactime_raw_df_cols = pd.Series(['Date', 'Size', 'Type', 'Mode', 'UID', 'GID', 'Meta', 'File Name'])
    mactime_ham_df_cols = pd.Series(['Size', 'MACB', 'UID', 'GID', 'Meta', 'Deleted_', 'Reallocated_', 'DriveLetter_', 'VSS_', 'EVOName_', 'EvidenceName_', 'Partition_', 'Tag_', 'FilePath_'])
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    dfsrc = "unknown"

    # D4_DataType_-col-based identification ------------------------------------------
    # We will assume that, if the D4_DataType_ col is set, the format is HAM
    if   'D4_DataType_' in df.columns and len(df) > 0 and not pd.isna(df['D4_DataType_'].iloc[0]):
        dfsrc = df['D4_DataType_'].iloc[0] + "-ham"
    # D4_Tool_-col-based identification ----------------------------------------------
    # Tool output can be RAW or HTM
    elif 'D4_Tool_' in df.columns and len(df) > 0 and not pd.isna(df['D4_Tool_'].iloc[0]):
        if df['D4_Plugin_'].iloc[0] == "" or pd.isna(df['D4_Plugin_'].iloc[0]):
            dfsrc = df['D4_Tool_'].iloc[0] + "-raw"
        else:
            dfsrc = df['D4_Tool_'].iloc[0] + "-" + df['D4_Plugin_'].iloc[0] + "-raw"
    # autoruns --------------------------------------------------------------------
    elif   autoruns_raw_df_cols.isin(df.columns).all():
        dfsrc="autoruns-raw"
    elif autoruns_ham_df_cols.isin(df.columns).all():
        dfsrc="autoruns-ham"
    # fstl ------------------------------------------------------------------------
    elif mactime_raw_df_cols.isin(df.columns).all():
        dfsrc="mactime-raw"
    elif mactime_ham_df_cols.isin(df.columns).all():
        dfsrc="mactime-ham"
    # plaso -----------------------------------------------------------------------
    elif 'timestamp_desc' in df.columns and '__container_type__' in df.columns and '__type__' in df.columns and 'data_type' in df.columns:
        dfsrc="plaso"
    # evtx  -----------------------------------------------------------------------
    elif '@xmlns' in df.columns and 'System > Provider > @Name' in df.columns and 'System > TimeCreated > @SystemTime' in df.columns:
        dfsrc="evtx-raw"
    # tshark  ---------------------------------------------------------------------
    elif 'frame.time' in df.columns or 'ip.src' in df.columns:
        dfsrc="tshark"
    # unknown ---------------------------------------------------------------------
    else:
        dfsrc = "unknown"
    objtype = 'pandas_dataframe-' + dfsrc

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] objtype: "+objtype)

    return objtype

def get_source_options(source_type):
    source_options = {
        'kape':       d4_kape.get_source_options(),
        'plaso':      d4_plaso.get_source_options(),
        'evtx':       d4_evtx.get_source_options()
    }

    return source_options.get(source_type,[])
    

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

    # Identify data source
    dfsrc = df.d4.df_source_identify()

    if d4.debug >= 3:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] df type -> "+dfsrc)

    # TOOLS ================================================
    # autoruns ---------------------------------------------
    if   re.search("^pandas_dataframe-autoruns", dfsrc):
        dfout = df.d4_autoruns.simple(*args, **kwargs)
    # kansa ------------------------------------------------
    elif re.search("pandas_dataframe-kansa", dfsrc):
        dfout = df.d4_kansa.simple(*args, **kwargs)
    # kape  ------------------------------------------------
    elif re.search("pandas_dataframe-kape", dfsrc):
        dfout = df.d4_kape.simple(*args, **kwargs)
    # macrobber ----------------------------------------------
    elif re.search("pandas_dataframe-macrobber-raw", dfsrc):
        dfout = df.d4_macrobber.simple(*args, **kwargs)
    elif re.search("pandas_dataframe-macrobber-ham", dfsrc):
        dfout = df.d4_macrobber.simple(*args, **kwargs)
    # mactime ----------------------------------------------
    elif re.search("pandas_dataframe-mactime-raw", dfsrc):
        dfout = df.d4_mactime.simple(*args, **kwargs)
    elif re.search("pandas_dataframe-mactime-ham", dfsrc):
        dfout = df.d4_mactime.simple(*args, **kwargs)
    # plaso ------------------------------------------------
    elif re.search("^pandas_dataframe-plaso", dfsrc):
        dfout = df.d4_plaso.simple(*args, **kwargs)
    # volatility -------------------------------------------
    elif re.search("^pandas_dataframe-volatility", dfsrc):
        dfout = df.d4_volatility.simple(*args, **kwargs)
    # tshark -----------------------------------------------
    elif re.search("^pandas_dataframe-tshark", dfsrc):
        dfout = df.d4_tshark.simple(*args, **kwargs)
    # DATA TYPES ===========================================

    # amcache ----------------------------------------------
    elif re.search("pandas_dataframe-amcache-ham$", dfsrc):
        dfout = df.d4_amcache.simple(*args, **kwargs)
    # flist  -----------------------------------------------
    elif re.search("pandas_dataframe-flist-ham$", dfsrc):
        dfout = df.d4_flist.simple(*args, **kwargs)
    # pslist  ----------------------------------------------
    elif re.search("pandas_dataframe-pslist-ham$", dfsrc):
        dfout = df.d4_pslist.simple(*args, **kwargs)
    # svclist  ---------------------------------------------
    elif re.search("pandas_dataframe-svclist-ham$", dfsrc):
        dfout = df.d4_svclist.simple(*args, **kwargs)
    # evtx  ------------------------------------------------
    elif re.search("^pandas_dataframe-evtx-raw$", dfsrc):
        dfout = df.d4_evtx.simple(*args, **kwargs)
    elif re.search("^pandas_dataframe-evtx-ham$", dfsrc):
        dfout = df.d4_evtx.simple(*args, **kwargs)
    # reg --------------------------------------------------
    elif re.search("^pandas_dataframe-winreg_kv-ham$", dfsrc):
        dfout = df.d4_winreg.simple(*args, **kwargs)
    # winservices-------------------------------------------
    elif re.search("^pandas_dataframe-winservices-ham$", dfsrc):
        dfout = df.d4_winservices.simple(*args, **kwargs)
    # unknown ----------------------------------------------
    elif dfsrc == "pandas_dataframe-unknown":
        print(Markdown("<font color='orange'>WARNING: DataFrame source not identified. simple() cannot be aplied.</font>"))
        dfout = df
    else:
        print("ERROR: [com] [simple_func] Unexpected error. Unhandled df source ("+dfsrc+").")
        dfout = df

    return dfout

def simple_common_series(se, *args, **kwargs):
    """ Reformat the input Series so the data is presented to the analyst in the
        friendliest possible way

    Parameters:
    se  (pd.Series):  Input data 
    
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    display(se.value_counts().to_frame())
    if se.dtype == 'int64' or se.dtype == 'float64':
        display("Max Value: {}".format(se.max()))
        display("Min Value: {}".format(se.min()))

def simple_common(df, *args, **kwargs):
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

    # Arg. Parsing -------------------------------------------------------------
    hiddencols             = kwargs.get('hiddencols',             [])
    nonhiddencols          = kwargs.get('nonhiddencols',          [])
    maxdfbprintlines       = kwargs.get('maxdfbprintlines',       20)
    max_rows               = kwargs.get('max_rows',               maxdfbprintlines)
    collapse_constant_cols = kwargs.get('collapse_constant_cols', True)
    hide_cols              = kwargs.get('hide_cols',              True)
    wrap_cols              = kwargs.get('wrap_cols',              True)
    round_ts_to_secs       = kwargs.get('round_ts_to_secs',       True)
    beautify_cols          = kwargs.get('beautify_cols',          True)


    # Show whatever selected on the screen or don't show anything   
    out_opt                = kwargs.get('out',                    True)
    # Show the resulting DF on the screen
    out_df_opt             = kwargs.get('out_df',                 True)
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
    out      = out_opt
    out_df   = out_df_opt
    out_meta = out_meta_opt
    ret      = ret_opt
    ret_out  = ret_out_opt

    hiddencolsdf = pd.DataFrame([])

    # ret implies no out, unless out is set specifically
    if ('ret' in kwargs.keys() or 'ret_out' in kwargs.keys()) and not outargset and (ret or ret_out):
        # If ret is set to True by the user we will not provide stdout output,
        # unless the user has specifically set out = True
        out      = False # Do not output to the screen at all
        out_df   = False # Do not output to the screen the DF, only the headers
        out_meta = False # Do not output to the screen the Metadata

    if 'out' in kwargs.keys():
        if out_opt == False:
            out_df   = False
            out_meta = False

    if 'out_df' in kwargs.keys():
        if out_df_opt:
            out    = True
            out_df = True

    if 'out_meta' in kwargs.keys():
        if out_meta_opt:
            out      = True
            out_meta = True

    if 'ret_out' in kwargs.keys():
        if ret_out:
            ret = True

    # Health Check -------------------------------------------------------------
    dfnrows = len(df)

    if dfnrows == 0:
        print("ERROR: Empty DataFrame.")
        return

    # Artifact / Tool specific processing --------------------------------------
    #
    # <INSERT HERE THE ARTIFACT / TOOL SPECIFIC PROCESSING ACTIONS>
    #

    # Var. Init. ---------------------------------------------------------------
    dfb = df

    # Hidden Columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if hide_cols:
        if d4.debug >= 3:
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] hide_cols")

        for col in hiddencols:
            if col not in nonhiddencols:
                if col in dfb.columns:
                    hiddencolsdf = pd.concat([hiddencolsdf, pd.DataFrame([col])], ignore_index=True)
                    dfb = dfb.drop(columns=[col])

    # Round TStamp columns to seconds  - - - - - - - - - - - - - - - - - - - - -
    if round_ts_to_secs:
        for col in dfb.select_dtypes(include=[np.datetime64]).columns:
            dfb[col] = dfb[col].dt.ceil("S")

    # Constant Columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if collapse_constant_cols:
        if d4.debug >= 3:
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] collapse_constant_cols")

        concolsdf, dfb = d4_utils.collapse_constant_columns(dfb)

    # Wrap wide columns  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if wrap_cols:
        if d4.debug >= 3:
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] wrap_cols")

        for col in dfb.columns:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.wrap(100)
                beautify_cols = True

    # Beautify columns - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if beautify_cols and out_df and len(dfb) >= max_rows:
        if d4.debug >= 3:
            print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] beautify_cols")

        print('WARNING: Too many rows (>'+str(max_rows)+') in DataFrame for beautification. Returning non-beautified output.')
        print('         Specify the "max_rows=" option to increase this value')
        beautify_cols = False
        pd.set_option("max_rows", max_rows)

    # DISPLAY ==========================================================
    nhiddencols = len(hiddencolsdf)

    if out:
        if out_meta:
            print("")
            display(Markdown("**Statistics:**\n<br>No. Entries: "+str(dfnrows)))

            if collapse_constant_cols:
                show_constant_cols = True
            else:
                show_constant_cols = False

            if hide_cols and nhiddencols != 0:
                show_hidden_cols = True
            else:
                show_hidden_cols = False

            if show_constant_cols and show_hidden_cols:
                d4_utils.display_side_by_side([hiddencolsdf, concolsdf], ['HIDDEN COLUMNS', 'CONSTANT COLUMNS'])
            elif show_constant_cols and show_hidden_cols == False:
                display(Markdown("**Constant Columns**"))
                max_rows = pd.get_option("display.max_rows")
                pd.set_option("display.max_rows", None)
                display(concolsdf)
                pd.set_option("display.max_rows", max_rows)
            elif show_constant_cols == False and show_hidden_cols:
                display(Markdown("**Hidden Columns**"))
                display(hiddencolsdf)

        if out_df:
            if beautify_cols:
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
    if ret:
        if ret_out:
            return dfb
        else:
            return df

# harmonize ===================================================================
def harmonize(*args, **kwargs):
    """ Alias for harmonize_func
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return harmonize_func(*args, **kwargs) 

def ham(*args, **kwargs):
    """ Alias for harmonize_func
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return harmonize_func(*args, **kwargs) 

def harmonize_func(*args, **kwargs):
    """ Convert DF in HAM format

        Args: 
            df (pandas.DataFrame): DF to harmonize
            kwargs(dict): harmonize options
        Returns: 
            pandas.DataFrame in HAM Format
    """
    def syntax():
        print('Syntax: harmonize(obj)\n')

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if len(args) == 0:
        syntax()
        return

    obj = args[0]

    tool         = kwargs.get('tool', None)

    # Identify data source
    objtype = data_identify(obj)

    # If the orchestrator or tool arguments are set, we directly invoke the 
    # corresponding module...
    if not tool == None:
        return eval('d4_'+tool).harmonize(*args, **kwargs)

    # ... otherwise, we will try to identify what type of data it is
    # and then invoke the corresponding module

    # help - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if   objtype == "str-help":
        syntax()
    # atrs - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-autoruns", objtype) or re.search("^pandas_dataframe-autoruns", objtype):
        return d4_autoruns.harmonize(*args, **kwargs)
    # fstl - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-fstl", objtype) or re.search("^pandas_dataframe-fstl", objtype):
        return obj
    # kansa  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-kansa", objtype) or re.search("^pandas_dataframe-kansa", objtype):
        return d4_kansa.harmonize(*args, **kwargs)
    # kape - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-kape", objtype) or re.search("^pandas_dataframe-kape", objtype):
        return d4_kape.harmonize(*args, **kwargs)
    # evtx - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-evtx", objtype) or re.search("^pandas_dataframe-evtx", objtype):
        return d4_evtx.harmonize(*args, **kwargs)
    # fstl - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-mactime", objtype) or re.search("^pandas_dataframe-mactime", objtype):
        return obj
    # plaso  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-plaso", objtype) or re.search("^pandas_dataframe-plaso", objtype):
        return d4_plaso.harmonize(*args, **kwargs)
    # tshark  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-tshark", objtype) or re.search("^pandas_dataframe-tshark", objtype):
        return d4_tshark.harmonize(*args, **kwargs)
    # volatility - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-volatility", objtype) or re.search("^pandas_dataframe-volatility", objtype):
        return d4_volatility.harmonize(*args, **kwargs)
    elif objtype == "unknown":
        #display(Markdown("<font color='orange'>WARNING: [d4com] Object source not identified. harmonize() cannot be aplied.</font>"))
        return obj
    else:
        #display(Markdown("<font color='orange'>WARNING: [d4com] Object source ("+objtype+") not supported for harmonization.</font>"))
        return obj

def harmonize_common(df, **kwargs):
    """ Convert DF in HAM format, common operations for all kind of DF

        Args: 
            df (pandas.DataFrame): DF to harmonize
            kwargs(dict): harmonize options
        Returns: 
            pandas.DataFrame in HAM Format
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    orchestrator      = kwargs.get('orchestrator',      None)
    tool              = kwargs.get('tool',              None)
    plugin            = kwargs.get('plugin',            None)
    datatype          = kwargs.get('datatype',          None)
    hostname          = kwargs.get('hostname',          None)
    drop_non_ham_cols = kwargs.get('drop_non_ham_cols', False)
    convert_string    = kwargs.get('convert_string',    True)

    # If we only want HAM cols, let's drop the others asap for performance
    if drop_non_ham_cols:
        df.drop(columns=df.columns[~df.columns.str.contains('_$')],inplace=True)

    # Change <NA> to NaN
    # This needs to be done before the "covert_dtypes" because otherwise
    # convert_dtypes will not properly identify timestamp columns
    from pandas.api.types import is_string_dtype

    # Try to adjust timestamp cols to datetime
    for col in df.columns:
        if df[col].dtypes == object:
            if type(df[col].iloc[0]) == str:
                df[col] = pd.to_datetime(df[col], errors='ignore')
    #df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
    #              if col.dtypes == object
    #              else col, 
    #              axis=0)

    # Change dtypes via auto-detect 
    # (Note: convert_dtypes does not convert timestamp cols to datetime,
    #        that's why we did it before above)
    if not df.empty:
        df = df.convert_dtypes(convert_string=convert_string)

    # Adjust column names
#    for col in df.columns:
        # Remove spaces from column names
#        colnew = re.sub(" ", "_", col)
#        df = df.rename(columns={col: colnew})

    # These settings may be already set if the data has been generated by 
    # a different collection tools (e.g. kansa -> autoruns) and read/parsed
    # by a different d4 library
    # Or some of these columns may have been created by the pre-processing 
    # section of the calling d4 harmonize function
    if not 'D4_Hostname_' in df.columns:
        df.insert(0, 'D4_Hostname_', hostname)
    if not 'D4_Plugin_' in df.columns:
        df.insert(0, 'D4_Plugin_', plugin)
    if not 'D4_Tool_' in df.columns:
        df.insert(0, 'D4_Tool_', tool)
    if not 'D4_Orchestrator_' in df.columns:
        df.insert(0, 'D4_Orchestrator_', orchestrator)
    if not 'D4_DataType_' in df.columns:
        df.insert(0, 'D4_DataType_', datatype)

    # Resort columns
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('D4_Hostname_')))
    cols.insert(0, cols.pop(cols.index('D4_Plugin_')))
    cols.insert(0, cols.pop(cols.index('D4_Tool_')))
    cols.insert(0, cols.pop(cols.index('D4_Orchestrator_')))
    cols.insert(0, cols.pop(cols.index('D4_DataType_')))
    df = df.reindex(columns=cols)

    # Set D4_* columns to categorical
    d4_cols = ['D4_Hostname_', 'D4_Plugin_', 'D4_Tool_', 'D4_Orchestrator_', 'D4_DataType_']

    for col in d4_cols:
        if len(df[col].drop_duplicates()) == 1:
            if df[col].isna().any():
                df[col] = pd.Categorical(df[col], categories=None)
            else:
                df[col] = pd.Categorical(df[col], categories=df[col].drop_duplicates())

    return df

# analysis ====================================================================
def analysis(*args, **kwargs):
    """ Redirects execution to analysis_func()
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return analysis_func(*args, **kwargs) 

def anl(*args, **kwargs):
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
        d4list("list")

    def d4list(objtype):
        for d4lib in d4anllibs:
            mystr=d4lib+'.analysis("list")'
            exec(mystr)
            print("")

    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    if len(args) == 0:
        syntax()
        return

    obj = args[0]

    # Identify data source
    objtype = data_identify(obj)

    # help - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if   objtype == "str-help":
        syntax()
    # Tools ------------------------------------------------------ 
    # autoruns - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-autoruns", objtype) or re.search("^pandas_dataframe-autoruns", objtype):
        return d4_autoruns.analysis(*args, **kwargs)
    # fstl - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-flist", objtype) or re.search("^pandas_dataframe-flist", objtype):
        return d4_flist.analysis(*args, **kwargs)
    # kansa  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-kansa", objtype) or re.search("^pandas_dataframe-kansa", objtype):
        return d4_kansa.analysis(*args, **kwargs)
    # kape - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-kape", objtype) or re.search("^pandas_dataframe-kape", objtype):
        return d4_kape.analysis(*args, **kwargs)
    # mactime  - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-mactime", objtype) or re.search("^pandas_dataframe-mactime", objtype):
        return d4_mactime.analysis(*args, **kwargs)
    # plaso  - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-plaso", objtype) or re.search("^pandas_dataframe-plaso", objtype):
        return d4_plaso.analysis(*args, **kwargs)
    # volatility - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-volatility", objtype) or re.search("^pandas_dataframe-volatility", objtype):
        return d4_volatility.analysis(*args, **kwargs)
    # tshark - - - - - - - - - - - - - - - - - - - - - - - - - 
    elif re.search("^dict-pandas_dataframe-tshark", objtype) or re.search("^pandas_dataframe-tshark", objtype):
        return d4_tshark.analysis(*args, **kwargs)
    # Artifacts -------------------------------------------------- 
    elif re.search("^dict-pandas_dataframe-evtx", objtype) or re.search("^pandas_dataframe-evtx", objtype):
        return d4_evtx.analysis(*args, **kwargs)
    elif re.search("^pandas_dataframe-pslist-ham", objtype):
        return d4_pslist.analysis(*args, **kwargs)
    elif objtype == "unknown":
        display(Markdown("<font color='orange'>WARNING: [d4com] Object source not identified. analysis() cannot be aplied.</font>"))
        # return obj
        return 
    else:
        display(Markdown("<font color='orange'>WARNING: [d4com] Object source ("+objtype+") not supported for analysis.</font>"))
        # return obj
        return

def find_anomalies(df, **kwargs):
    return d4_ml.find_anomalies(df, **kwargs)


# ACCESSOR ####################################################################
@pd.api.extensions.register_dataframe_accessor("d4")
class Ds4n6commonAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def df_source_identify(self):
        df=self._obj
        return df_source_identify_func(df)

    def simple(self, *args, **kwargs):
        df=self._obj
        return simple_func(df, *args, **kwargs)

    def anl(self):
        df=self._obj
        return analysis_func(df)

@pd.api.extensions.register_series_accessor("simple")
class Ds4n6SeriesCommonSimpleAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        se=self._obj
        return simple_common_series(se, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("simple")
class Ds4n6CommonSimpleAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        df=self._obj
        return simple_func(df, *args, **kwargs)

# spl() -> Short alias for simple()
@pd.api.extensions.register_dataframe_accessor("spl")
class Ds4n6CommonSplAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        df=self._obj
        return simple_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("analysis")
class Ds4n6CommonAnalysisAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        df=self._obj
        return analysis_func(df, *args, **kwargs)

# anl() -> Short alias for analysis()
@pd.api.extensions.register_dataframe_accessor("anl")
class Ds4n6CommonAnlAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        df=self._obj
        return analysis_func(df, *args, **kwargs)

@pd.api.extensions.register_dataframe_accessor("xgrep")
class Ds4n6CommonXgrepAccessor:
    def __init__(self, pandas_obj):
            self._obj = pandas_obj

    def __call__(self, *args, **kwargs):
        return d4_unx.xgrep_func(self._obj, *args, **kwargs)
