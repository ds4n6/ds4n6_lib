# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# IDEAS
###############################################################################
# dfsed
# multicol -> For a series or DF col, show in multiple cols to optimize screen
#             Equiv. to Linux: pr -l1 -t -3 /t

###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4unx

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import re
import inspect

# DS IMPORTS ------------------------------------------------------------------
import numpy  as np
import pandas as pd

###############################################################################
# FUNCTIONS
###############################################################################

def xgrep_func(*args, **kwargs):

    def syntax():
        print('Syntax: xgrep(<object>,[<column(s)>],"<regex>"[,"<options>"])')
        print("        [column(s)] -> If object is a DataFrame")
        print("        Options: i: case insensitive")
        print("                 v: reverse")
        print("                 p: explode cols with series elements")
        print("                 t: do not apply style (highlight hits)")

    import collections

    nargs=len(args)

    if nargs == 0:
        syntax()
        return

    obj = args[0]

    if isinstance(obj, dict):
        return dictgrep(*args, **kwargs)
    elif isinstance(obj, pd.DataFrame):
        return dfgrep(*args, **kwargs)
    elif isinstance(obj, collections.abc.KeysView):
        return keysgrep(*args, **kwargs)

def xgrep(*args, **kwargs):
    return xgrep_func(*args, **kwargs)

def dfgrep(*args):
    """
    Syntax: dfgrep("<column>","<regex>"[,"<options>"])
            Options: i: case insensitive
                     v: reverse
                     p: explode cols with series elements
                     t: do not apply style (highlight hits)
            If your DF has only 1 column, you can skip the column name,
            just specify ""

    """

    nargs=len(args)

    df    = args[0]

    # If the user supplies just one arg we will assume that it is the regex 
    # and that he wants to search the full DF for that regex
    if nargs == 2:
        cols  = "*"
        regex = args[1]
    else:
        cols  = args[1]
        regex = args[2]

    if nargs == 4:
        opt = args[3]
    else:
        opt = ""

    ndfcols=len(df.columns)

    if ndfcols == 1 and cols == "":
        cols = df.columns

    if cols == "*":
        cols = df.columns

    if regex == "":
        print("ERROR: regex cannot be empty")
        return

    # Parse Options
    if "v" in opt:
        reverse=True
    else:
        reverse=False

    if "i" in opt:
        case=False
    else:
        case=True

    if "t" in opt:
        applystyle = False
    else:
        applystyle = True

    dfout = pd.DataFrame([])

    if isinstance(cols, str):
        cols=list([cols])

    for col in cols:
        # Check if col is an existing column
        if col not in df.columns:
            print ('ERROR: column '+col+' not found in DF')
            return

        if "p" in opt:
            df=df.explode(col)
  
        # Identify if there are null values and fill them
        df=df.copy()
        # df[col]=df[col].fillna("d4_null")

        if reverse :
            resdf = df[~df[col].astype(str).str.contains(regex,case=case)]
        else:
            resdf = df[df[col].astype(str).str.contains(regex,case=case)]

        dfout = dfout.append(resdf)

    # for col in cols:
    #     dfout[col]=dfout[col].fillna("d4_null")
 
    dfout = dfout.drop_duplicates()

    if applystyle :
        maxdfoutprintlines = 1000
        if len(dfout) >= maxdfoutprintlines:
            print('WARNING: Too many lines (>'+str(maxdfoutprintlines)+') in DataFrame for formatting. Returning unformatted output.')
            return dfout
        else:
            dfout = dfout.reset_index()
            return dfout.style.apply(lambda x: ["background: yellow" if re.search(regex, str(v)) else '' for v in x], axis = 1)
    else:
        return dfout

def keysgrep(keys, regex, opt=""):
    df = pd.DataFrame(list(keys), columns=['Key'])
    return df.d4unx.dfgrep('Key',regex, opt)

def dictgrep(mydict, regex, opt=""):
    # DFs dict -----------------------------------------------
    if isinstance(mydict[list(mydict.keys())[0]], pd.DataFrame):

        outdf = pd.DataFrame([])

        # Do not apply style on dfgrep
        dfgrepopt = opt+"t"

        for key in mydict.keys():
            thisdf = dfgrep(mydict[key], "*", regex, dfgrepopt)
            thisdf.insert(0, 'dict-Key_', key) 
            outdf = pd.concat([outdf, thisdf], ignore_index=True)

        # Return resulting DF
        if "t" in opt:
            return outdf.dropna(axis=1, how='all')
        else:
            return outdf.dropna(axis=1, how='all').style.apply(lambda x: ["background: yellow" if re.search(regex, v) else '' for v in x], axis = 1)
    else:
        print("ERROR: dict variant not supported.")

def dfsed_func(df,col,regex,repl,opt=""):

    df[col]=df[col].str.replace(regex,repl)

    return df

def vc_func(df,col,countfilter="",ascending=False):

    dfout = df[col].value_counts(ascending=ascending).reset_index().rename(columns={"index": col, col: "Count"})
      
 
    if countfilter != "":
    
        n=int(countfilter)        
        dfout=dfout.query(f'Count == {n}')

    return dfout

def ddups_func(df):
    dfout=df.drop_duplicates()

    return dfout

# ACCESSOR ####################################################################
@pd.api.extensions.register_dataframe_accessor("d4unx")
class Ds4n6UnxAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def dfgrep(self, *args, **kwargs):
        obj = self._obj
        return xgrep_func(obj, *args, **kwargs)

    def dfsed(self,col,regex,repl,opt=""):
        df=self._obj.copy()
        return dfsed_func(df,col,regex,repl,opt)

    def vc(self,col):
        df=self._obj
        return vc_func(df,col)

    def ddups(self):
        df=self._obj
        return df.ddups_func()

