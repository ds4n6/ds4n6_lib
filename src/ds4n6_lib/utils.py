# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4utl

###############################################################################
# IMPORTS
###############################################################################

# python IMPORTS --------------------------------------------------------------
import os
import re
import inspect
import xmltodict
import glob

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd
import ipywidgets as ui
from IPython.display import display, Markdown, HTML

import ds4n6_lib.d4          as d4
import ds4n6_lib.common  as d4com


###############################################################################
# FUNCTIONS
###############################################################################

# EXTENDER pandas COLUMN/ROW FUNCTIONS ########################################
def insert_column_after_another_by_name(*args, **kwargs):
    return insert_or_move_column_after_another_by_name(*args, **kwargs)

def move_column_after_another_by_name(*args, **kwargs):
    return insert_or_move_column_after_another_by_name(*args, **kwargs)

def insert_or_move_column_after_another_by_name(df, col, refcol, defvalue=None, where2move="after"):
    """
    This function supports both inserting a new column or moving a column to
    the designated location
    """
    # Locate index of refcol
    loc = df.columns.get_loc(refcol)

    # New location
    if where2move == "after":
       newloc = loc + 1
    elif where2move == "before":
       newloc = loc - 1

    df.insert(loc=newloc, column=col+'_%%NEW%%_', value=defvalue) 

    # Insert col at location
    if col in df.columns:
        df[col+'_%%NEW%%_'] = df[col]
        df.drop(columns=col, inplace=True)

    df = df.rename(columns={col+'_%%NEW%%_': col})

    return df

# FILE READING/SAVING FUNCTIONS ###############################################
def find_files_in_folder(folder, filter="", ext='', maxdepth=4):

    def filter_files(objs):
        # Check which objects are files
        files = []
        for obj in objs:
            if re.search(filter, obj):
                if os.path.isfile(obj):
                    files.append(obj)

        return files

    if d4.debug >= 4:
        print("DEBUG: - Max Depth:   "+str(maxdepth))
        print("DEBUG: - Top Folder:  "+folder)

    lev       = 1
    files     = []
    nnewfiles = 0
    nfiles    = 0
    wildcard  ="/*"

    while lev <= maxdepth:
        path_name = "{}{}{}".format(folder, wildcard*lev, ext)
        newfiles = filter_files(glob.glob(path_name))
        if newfiles == []:
            nnewfiles = 0
        else:
            files.extend(newfiles)
            nnewfiles = len(newfiles)
            nfiles    = len(files)

        if d4.debug >= 4:
            print("DEBUG: - Finding objs at level {}".format(lev))
            print("DEBUG:     "+path_name)
            print("DEBUG:   + "+str(nnewfiles)+" new   files found.")
            print("DEBUG:   + "+str(nfiles)+" total files found.")

        lev += 1

    if d4.debug >= 4:
        print("DEBUG: - Total Files Found:   "+str(nfiles))
    
    return files

# Read/Save from/to HDF =======================================================

def save_df_to_hdf_chunked(df, filename, chunksize=1000000, complib='blosc', complevel=9):
    print("Saving DF to HDF Store")
    print("")
    print("- Details:")
    print("  + File Name:  "+filename)
    print("  + DF Length:  "+str(len(df)))
    print("  + Chunk Size: "+str(chunksize))
    print("  + No. Chunks: "+str(int(len(df)/chunksize)))
    print("  + Comp. Lib.: "+complib)
    print("  + Comp. Lev.: "+str(complevel))
    print("")

    print("- Saving", end='')
    start = 0
    end = chunksize-1
    cnt = 1
    while end < df.shape[0]:         
        chunk = df.iloc[start:end]
        #chunk.to_hdf(filename, 'df'+str(int(cnt)), mode='a', format='table')
        chunk.to_hdf(filename, 'df'+str(cnt), mode='a', format='table', complib=complib, complevel=complevel)
        start += chunksize
        end += chunksize
        cnt += 1
        
        if cnt % 10 == 0:
            print("["+str(cnt)+"]", end='')
        else:
            print(".", end='')

    print("")
    print("- Done")

def read_df_from_hdf_chunked(filename, chunksize=1000000):

    # Figure out the no. of chunks
    with pd.HDFStore(filename) as hdf:
        import re
        pat = re.compile(r'^/df[0-9][0-9]*$')
        dfkeys = pd.Series([i for i in hdf.keys() if pat.match(i)])
        dfkeys = dfkeys.str.replace('^/df','').astype(int)
        maxdfidx = max(dfkeys)

    print("Details:")
    print("  + File Name:  "+filename)
    print("  + Chunk Size: "+str(chunksize))
    print("  + No. Chunks: "+str(maxdfidx))
    print("")

    print("- Reading", end='')
    with pd.HDFStore(filename, mode='r') as hdf:
        cnt = 1
        df = pd.DataFrame()
        while cnt <= maxdfidx:
            chunkdf = hdf.get('df'+str(cnt))
            df = pd.concat([df, chunkdf], ignore_index=True)
            cnt += 1

            if cnt % 10 == 0:
                print('['+str(cnt)+']', end='')
            else:
                print('.', end='')
    print("")
    print("- Resulting DF:")
    print("  + DF Length:  "+str(len(df))) 
    print("")    
    
    return df

# DISPLAY FUNCTIONS ###########################################################
def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


# dict FUNCTIONS ##############################################################
def dfsdict_stats(dfsdict):
    """ Show stats of len keys of dictionary
    Input:
        dfsdict: dictionary        
    """
    for key in dfsdict.keys():
        print('%8i - %-45s' % (len(dfsdict[key]),key))


def dict_to_flatdict(dct,sep="_"):
    """Flaten the dictionary 
    Input:
        dct: dictionary
        sep: separator
    """
    import collections

    flatdict = collections.OrderedDict()

    def recurse(t,parent_key=""):

        if isinstance(t,list):
            for i in range(len(t)):
                recurse(t[i],parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t,dict):
            for k,v in t.items():
                recurse(v,parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t
    
    for _l1k,l1v in dct['Events'].items():
        for x in range(len(l1v)):
            obj = collections.OrderedDict()
            recurse(l1v[x])
            flatdict[x]=obj            

    return flatdict


def dict_to_df(dct,sep="_"):
    """Convert dictionary to dataframe
    Input:
        dct: dictionary
        sep: separator
    """
    flatdict = dict_to_flatdict(dct, sep=sep)

    # Let's clear dct to free memory
    dct = {}

    df = pd.DataFrame.from_dict(flatdict, orient='index')

    return df


# XML  FUNCTIONS ##############################################################
def xml_to_df(xmlstr, sep="_"):
    """Convert xml to dataframe
    Input:
        xmlstr: string (xml content)
        sep: separator
    """
    import xmltodict
    # unicode escape invalid XML strings
    xmlstr = escapeInvalidXML(xmlstr)

    try:
        xmldct = xmltodict.parse(xmlstr)
    # except Exception as err:
    except xmltodict.expat.ExpatError as err:
        print("    => ERROR: line: {} column: {} error_string: {}".format(err.lineno,err.offset, xmlstr.splitlines()[err.lineno-1]))
        raise

    # Let's clear xmlstr to free memory. We no longer need it.
    xmlstr=""

    return dict_to_df(xmldct, sep=sep)


def escapeInvalidXML(string):
    """ Escape invalid XML from ranges invalid unicode chars """

    r = re.compile('[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\xFF' \
      + '\u0100-\uD7FF\uE000-\uFDCF\uFDE0-\uFFFD]')

    def replacer(m):
        return "<u>\\u"+('%04X' % ord(m.group(0)))+"</u>"
    return re.sub(r,replacer,string)

# df HELPERS ##################################################################
def collapse_constant_columns(df):
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    dfnrows = len(df)

    if dfnrows != 0:
        concolsdf = pd.DataFrame()

        ncols  = df.shape[1]
        colcnt = 1
        for col in df.columns:
            if d4.debug >= 4:
                print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()] collapse_constant_cols -> ["+str(colcnt)+"/"+str(ncols)+"] "+col)
 
            if col != 'nan':
                if df[col].dtype == object:
                    duplicates = df[col].astype('string').drop_duplicates()
                else:
                    duplicates = df[col].drop_duplicates()
                if len(duplicates) == 1:
                    concolsdf = concolsdf.append([[col,str(duplicates.iloc[0])]], ignore_index=True)
                    df = df.drop(columns=col)

            colcnt +=1

        concolsdf = concolsdf.rename(columns={ 0: 'Column', 1: 'Value' })

        return concolsdf, df

# NOTEBOOK FUNCTIONS ##########################################################

def nbgrep(nb,regex,celltype="code"):
    """
    Input:
        nb: notebook file name.
        regex: regex 
        celltype: celltype to apply regex (code by default)
    """
    import nbformat
    import re

    if nb == "":
        print("- ERROR: notebook file name empty.")
        return

    if os.path.exists(nb):
        allhits = []
        nb = nbformat.read(nb,as_version=4)
        for c in dict(nb)['cells']:
            cdict = dict(c)
            if cdict['cell_type'] == celltype:
                cellsrc = cdict['source']
                hits = re.findall(regex,cellsrc)
                allhits = allhits+hits

        return allhits
    else:
        print("- ERROR: notebook file not found.")
        print("         "+nb)
        return


def extract_dfs_from_type(df_type, dfs):
    new_dfs = {}
    for dfkey in dfs:
        if d4com.data_identify(dfs[dfkey]).startswith(df_type):
            new_dfs[dfkey] = dfs[dfkey]
    return new_dfs

# ACCESSOR ####################################################################

@pd.api.extensions.register_dataframe_accessor("d4utl")
class Ds4n6UtilsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def dummy(self,arg1):
        dummy=0

class PathSelector():

    def __init__(self,start_dir,select_file=True):
        self.file        = None
        self.select_file = select_file
        self.cwd         = start_dir
        self.select      = ui.SelectMultiple(options=['init'],value=(),rows=10,description='')
        self.accord      = ui.Accordion(children=[self.select])

        self.accord.selected_index = None # Start closed (showing path only)
        self.refresh(self.cwd)
        self.select.observe(self.on_update,'value')

    def on_update(self,change):
        if len(change['new']) > 0:
            self.refresh(change['new'][0])

    def refresh(self,item):
        global path
        path = os.path.abspath(os.path.join(self.cwd,item))

        if os.path.isfile(path):
            if self.select_file:
                self.accord.set_title(0,path)
                self.file = path
                self.accord.selected_index = None
            else:
                self.select.value = ()

        else: # os.path.isdir(path)
            self.file = None
            self.cwd  = path

            # Build list of files and dirs
            keys = ['[..]']
            for item in os.listdir(path):
                if item[0] == '.':
                    continue
                elif os.path.isdir(os.path.join(path,item)):
                    keys.append('['+item+']')
                else:
                    keys.append(item)

            # Sort and create list of output values
            keys.sort(key=str.lower)
            vals = []
            for k in keys:
                if k[0] == '[':
                    vals.append(k[1:-1]) # strip off brackets
                else:
                    vals.append(k)

            # Update widget
            self.accord.set_title(0,path)
            self.select.options = list(zip(keys,vals))
            with self.select.hold_trait_notifications():
                self.select.value = ()

def df_outlier_analysis(indf,sensitivity):

    # Method: analyze all fields and find the ones different from others
    # Select cols that have a small no of different values
    # - df.nunique() < 10          --> Bool Series w/ cols that have < 10 different values
    # - df.T[df.nunique() < 10].T  --> Show those values
    # - drop_duplicates()          --> Drop duplicate rows
    # - .index                     --> Show row # for those rows
    # - .iloc                      --> Select df rows based on row index

    # Readable version
    #      intcolsmsk=df.nunique() < 10
    #      inting=df.iloc[df.T[intcolsmsk].T.drop_duplicates().index]
    #      return inting
    # One liner version

    df = indf.copy()
    nrows = df.shape[0]
    maxrows = int(sensitivity*(nrows*.02))
    df.drop(columns=df.columns[df.nunique()>maxrows], inplace=True) 
    intgdf = df.iloc[df.T[df.nunique() < maxrows/5].T.drop_duplicates().index]

    return intgdf


# KNOWLEDGE ###################################################################

# Regex Patterns ==============================================================
ipregex=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"

