# DS4N6
#
# Description: library of functions to appy Data Science in several forensics
#              artifacts
#

###############################################################################
# IMPORTS
###############################################################################
# python IMPORTS
import os
import glob
import re
import time
import xmltodict
import json
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as et

# DS IMPORTS
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense

###############################################################################
# FUNCTIONS
###############################################################################


def python_env_set():
    # ipython Visualization Settings
    pd.set_option('display.width', 2000)
    pd.set_option("max_columns", 500)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.colheader_justify', 'left')
    pd.set_option('display.max_rows', 500)

###############################################################################
# NON-DS FUNCTIONS
###############################################################################


def save_obj_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

###############################################################################
# DS4N6 FUNCTIONS
###############################################################################


# FILE SYSTEM TIMELINE (fstl) #################################################
def read_fstl(fstlf, windows=False):
    fstl = pd.read_csv(fstlf)
    fstl['Date'] = fstl['Date'].astype('datetime64')
    fstl = fstl.rename(columns={"File Name": "FileName"})
    if windows:
        fstl.drop(columns=['Mode','UID','GID'],inplace=True)
    return fstl


def fstl_size_top_n(fstl, n):
    return fstl[~fstl['FileName'].str.contains("\(\$FILE_NAME\)")][['Size','FileName']].sort_values(by='Size', ascending=False).drop_duplicates().head(n)


def read_fstls_filetypes(fstld, hosts, file_types, verbose=False):
    fstl_names = ['1', 'path', 'inode', 'perms', 'user', 'group', 'fsize', 'mtime', 'atime', 'ctime', 'btime']
    fstl_hostname_names = ['host-vol', '1', 'path', 'inode', 'perms', 'user', 'group', 'fsize', 'mtime', 'atime', 'ctime', 'btime']
    fstl_hostname_names_short = ['host-vol', 'path', 'inode', 'fsize', 'mtime', 'atime', 'ctime', 'btime']

    # Initialize dictionary of dfs
    dfs = {}
    for file_type in file_types:
        dfs[file_type] = pd.DataFrame(columns = fstl_hostname_names_short)

    nhosts = len(hosts)

    if verbose:
        print("No. Hosts: "+str(nhosts))
        print("- Reading files:")

    start_time = time.time()

    cnt = 1
    for host in hosts:
        fstlf = fstld + "/" + host + "/fstlmaster.body.raw"
        os.system("ls -l " + fstlf + " | sed 's:" + fstld + "/::' | awk '{ print \"      \" $0 }'")

        filename=fstlf
        if verbose:
            print("  + [" + str(cnt) + "/" + str(nhosts) + "] Reading file: " + filename)
        dirname = os.path.dirname(filename)
        dirnamebase = os.path.basename(dirname)
        parse_dates = ['mtime', 'atime','ctime']
        fstlraw = pd.read_csv(filename, sep='|', names=fstl_names, parse_dates=parse_dates, date_parser=lambda col: pd.to_datetime(col, unit="s"))
        fstlraw.insert(0,'host-vol',dirnamebase)

        # Remove meaningless cols -------------------------------
        # Delete first col
        del fstlraw['1']
        # Delete Meaningless Windows cols
        del fstlraw['perms']
        del fstlraw['user']
        del fstlraw['group']
        # Add path-hash col
        fstlraw.insert(2,'path-hash',0)
        fstlraw['path-hash'] = fstlraw['path'].str.lower().apply(hash)

        thisdfs={}
        for file_type in file_types:
            thisdfs[file_type] = fstlraw[fstlraw['path'].str.contains("."+file_type+"$")]
            dfs[file_type] = pd.concat([dfs[file_type], thisdfs[file_type]])

        if verbose:
            print("    - No.lines fstls:   " + str(fstlraw.path.size))
            for file_type in file_types:
                print("    - No.lines " + file_type + ":     " + str(thisdfs[file_type].path.size))
                print("    - No.lines " + file_type + " acc: " + str(dfs[file_type].path.size))
        else:
            if verbose:
                print(".", end='')
            if ( cnt % 10 == 0 ):
                print("[" + str(cnt) + "]", end='')
        cnt = cnt + 1

    if verbose:
        print("- "+str(nhosts)+" files read")
        print("- Creating Low-Res TStamp versions of DFs")

    for file_type in file_types:
        dfs[file_type]=dfs[file_type].astype(
            {
                'path-hash': 'int64', 
                'mtime': 'datetime64[s]', 
                'atime': 'datetime64[s]', 
                'ctime': 'datetime64[s]', 
                'btime': 'datetime64[s]'})

    elapsed_time = time.time() - start_time
    if verbose:
        print("- Elapsed time: "+str(elapsed_time))

    return dfs


# plaso #######################################################################
def read_plaso_l2tcsv(plasof):
    dfs = {}
    # When creating the .csv file with psort, make sure to filter out events 
    # with absurd timestamps (e.g. < 1970 or > the current year). 
    # plaso sometimes includes "broken" timestamps and the will break pandas
    # psort.py -z UTC -o l2tcsv -w plaso_file.csv plaso_file.dump \ 
    #     "date > '1960-01-01 00:00:00' AND date < '2020-12-31 23:59:59'"
    plines=pd.read_csv(plasof)

    # Now, we need to clean the data. There may be wrong values.
    # TODO: Check more thoroughly for consistency in timestamps
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
    return dfs


def read_plaso_json(plasof):
    #
    # Generate your plaso json file with the following command:
    #     psort.py -z UTC -o json -w plaso_file.json plaso_file.dump "date > '1960-01-01 00:00:00' AND date < '2021-01-01 00:00:00'"
    dfs = {}
    print("File: " + plasof)
    
    print('Reading json file lines ',end='')
    cnt = 0
    with open(plasof) as infile:
        evtsdict = {}
        for line in infile:
            if line.strip() != "}":
                lineclean = re.sub('^[,{]\ *','', line)
                datatype = lineclean.split('"data_type": ', maxsplit=1)[-1].split(maxsplit=1)[0].replace('"', '').replace(",", "").replace(":","_")
    
                if datatype not in evtsdict.keys():
                    evtsdict[datatype]={}
                linecleanjson = json.loads("{"+lineclean+"}")
                evtsdict[datatype].update(linecleanjson)
            if cnt % 50000 == 0:
                print("[" + str(cnt) + "] ", end='')
            cnt=cnt+1
    
    print("\n")
    print("Generating pandas dataframes: ")
    keys = evtsdict.keys()
    for key in keys:
        # Convert json string to pandas DF
        nevtskey = len(evtsdict[key])
        print('- %-45s ... ' % (key), end='')
        dfs[key] = pd.DataFrame(evtsdict[key]).T
        print(' [%s]' % (str(nevtskey)))
    print("\nNOTE: Now you can use the syntax <yourvar>['<datatype>'] to access your dataframe")
    return dfs


# KANSA #######################################################################
def read_kansa(kansad):
    dfs = {}
    # Let's parse all KANSA files into category-specific dataframes
    dirs = [x[0] for x in os.walk(kansad)]
    for dir in dirs:
        if dir != kansad:
            catd = dir
            cat = os.path.basename(catd)
            dfs[cat] = pd.DataFrame()
            print('Reading csv files for category %-15s into dataframe ->  %-15s' % (cat,cat))
            hostfs = os.listdir(kansad + "/" + cat)
            for hostf in hostfs:
                hostffull = kansad + "/" + cat + '/' + hostf
                host = str.replace(os.path.basename(hostffull),'-' + cat + '.csv', '')
                try:
                    hostlines = pd.read_csv(hostffull, encoding='utf16')
                except:                
                    hostlines = pd.DataFrame()    
                hostlines.insert(0,'Hostname',host)
                dfs[cat]=pd.concat([dfs[cat],hostlines],ignore_index=True)
    print("\n\nNOTE: Now you can use the syntax <yourvar>['Category'] to access your dataframe")
    return dfs


# VOLATILITY ##################################################################
def read_volatility(evd, prefix, ext):
    """ Read volatility files from a directory and put in a pandas Dataframe for analysis

    Parameters:
    evd (str): Path of volatilty files
    prefix (str): Get files with this prefix
    ext (str): Get files with this extension
    
    Returns:
    pd.DataFrame: Contains volatility files info.
                  You can use the syntax <yourvar>['Category'] to access your dataframe

    """
    dfs = {}
    volfsf = [f for f in glob.glob(evd + "/*/*" + ext)]
    # TODO: Use regex to include "^" & "$" instead of a vanilla replace
    cats = [os.path.basename(volff).replace(prefix, '').replace(ext, '') for volff in volfsf]
    cats = np.unique(cats)
    for cat in cats:
            dfs[cat] = pd.DataFrame()
            print('Reading csv files for category %-20s into dataframe ->  %-20s' % (cat, cat))
            hostcatfs = [f for f in glob.glob(evd+"/*/" + prefix + cat + ext)]
            for hostcatf in hostcatfs:
                hostdfull = os.path.dirname(hostcatf)
                host = os.path.basename(hostdfull)
                try:
                    hostcatlines = pd.read_csv(hostcatf,sep="|")
                except:                
                    hostcatlines = pd.DataFrame()    
                hostcatlines.insert(0,'Hostname',host)
                if cat == "pslist":
                    hostcatlines['Start'] = pd.to_datetime(hostcatlines['Start'])
                    hostcatlines['PID'] = hostcatlines['PID'].astype('int64')
                    hostcatlines['PPID'] = hostcatlines['PPID'].astype('int64')
                    hostcatlines['Thds'] = hostcatlines['Thds'].astype('int64')
                    hostcatlines['Hnds'] = hostcatlines['Hnds'].astype('int64')
                    hostcatlines['Sess'] = hostcatlines['Sess'].astype('int64')
                    hostcatlines['Wow64'] = hostcatlines['Wow64'].astype('int64')
                    hostcatlines['Exit'] = pd.to_datetime(hostcatlines['Exit'])
                dfs[cat] = pd.concat([dfs[cat], hostcatlines], ignore_index=True)
    print("\n\nNOTE: Now you can use the syntax <yourvar>['Category'] to access your dataframe")
    return dfs


def volatility_pslist_unfrequent_process_analysis(pslistdf, n):
    sr = (pslistdf['Name'].value_counts() <= n)
    print("No. Processes with less than " + str(n) +" occurrences: " + str(len(sr.index[sr])))
    return pd.Series.reset_index(sr.index[sr].to_series()).drop(columns=['index']).rename(columns={0: "Name"})


def volatility_pslist_boot_time_anomaly_analysis(pslistdf, secs=30):
    """ Find anomalies in boot time

    Parameters:
    pslistdf (pd.DataFrame): Dataframe with pslist volatility info
    secs (int): Diference of boot time
    
    Returns:
    pd.DataFrame: Analysis results, processes that have an anomalous boottime

    """
    bootps = pslistdf[pslistdf['Name'].isin(boot_start_processes)  & (pslistdf['Sess'] <= 1) & pslistdf['Exit'].isnull() ]
    return bootps[bootps['Start'] >= bootps['Start'].min() + pd.Timedelta(seconds=secs)]

def volatility_processes_parent_analysis(pslistdf, critical_only=False):
    """ Find anomalies in parent processes

    Parameters:
    pslistdf (pd.DataFrame): Dataframe with pslist volatility info
    critical_only (bool): Only critical process
    
    Returns:
    None
    """
    pslistdf_alive = pslistdf[pslistdf['Exit'].isna()]
    hnpid = pslistdf_alive[['Hostname', 'Name', 'PID']]
    hnppid = pslistdf_alive[['Hostname', 'Name', 'PPID']]
    family = pd.merge(
                    hnppid, hnpid, left_on=['Hostname', 'PPID'], right_on=['Hostname', 'PID'], how='left'
              ).dropna(
              ).drop(
                    columns=['Hostname', 'PPID', 'PID']
              ).rename(
                    columns={'Name_x': 'Child', 'Name_y': 'Parent'}
              ).reset_index(
              ).drop(
                    columns=['index'])
    if critical_only == True:
        thisfamily = family.query('Child == @critical_processes')
    else:
        thisfamily = family
    family_unknown = pd.merge(
                            thisfamily,process_parents, indicator=True, how='outer'
                      ).query(
                            '_merge=="left_only"'
                      ).drop(
                            '_merge', axis=1)
    print(family_unknown.groupby(["Child", "Parent"]).size().sort_values(ascending=False))


# EVENT LOG (evtx) ############################################################
def handle_evtx(_, evtx):
    print(evtx['Event'])
    return True

def evtx_xml(evtxf):
    import Evtx.Evtx as evtx
    import Evtx.Views as e_views

    print("EVTX -> XML")
    
    thistr = ''
    with evtx.Evtx(evtxf) as log:
        thistr = e_views.XML_HEADER
        thistr = thistr + '<Events>'
        for record in tqdm(log.records()):
            thistr = thistr + record.xml()
        thistr = thistr + '</Events>'

    return thistr


def evtx_new_xml_parse(evtxxmlf, file=False):
    ns = {"xml": "http://schemas.microsoft.com/win/2004/08/events/event"}

    et.register_namespace("xml", "http://schemas.microsoft.com/win/2004/08/events/event")

    tree = et.ElementTree()
    msg = ""
    if file:
        tree.parse(evtxxmlf)
        msg = "Reading from XML File"
    else:
        tree = et.ElementTree(et.fromstring(evtxxmlf))
        msg = "Parsing XML from Memory"
    
    print(msg)
        
    root = tree.getroot()
    rows = []

    evtfull = pd.DataFrame()

    for node in tqdm(root.findall("./xml:Event", ns)):
        default_data = {}
        for nodes in node.findall("./xml:System", ns):
            for nodesc in nodes.getchildren():
                if nodesc.text:
                    default_data[
                        nodesc.tag.replace('{http://schemas.microsoft.com/win/2004/08/events/event}', '')] = nodesc.text
                if nodesc.attrib.items():
                    for nodesca in nodesc.attrib.items():
                        default_data[
                            nodesc.tag.replace('{http://schemas.microsoft.com/win/2004/08/events/event}', '') + "_" +
                            nodesca[0]] = nodesca[1]
        for noded in node.findall("./xml:EventData", ns):
            for nodedd in noded.findall("./xml:Data", ns):
                default_data[
                    nodedd.attrib["Name"].replace('{http://manifests.microsoft.com/win/2004/08/windows/eventlog}',
                                                  '')] = nodedd.text
        for nodeu in node.findall("./xml:UserData", ns):
            for nodeuu in nodeu.getchildren():
                default_data[nodeuu.tag.replace('{http://manifests.microsoft.com/win/2004/08/windows/eventlog}',
                                                '')] = nodeuu.text

        rows.append(default_data)

    evtfull = evtfull.append(rows)
    return evtfull


def evtx2df(evtxf, evtsave=""):
    import tempfile
    """
    Convert evtx file to dataframe.
    """
    print("Executing evtx to dataframe...")
    
    if evtsave:
        evtdf = evtx_new_xml_parse(evtxf,True)
    else:
        evtdf = evtx_new_xml_parse(evtx_xml(evtxf))            


# NOT DEPRECATED YET - (DEPRECATED BY NEW FUNCTION evtx_new_xml_parse())
def evtx_xml_parse(evtxxmlf,compact):
    print("- Reading XML file")
    with open(evtxxmlf) as fd:
        doc = xmltodict.parse(fd.read())

    print("No. Event Records: " + str(len(doc['Events']['Event'])))

    cnt = 1
    usrdatacnt = 1
    evtdatacnt = 1

    for event in doc['Events']['Event']:
        if ( cnt % 100 == 0 ):
            print(" " + str(cnt) + " ",end='')

        # System --------------------------------------------------------------
        if ( cnt % 100 == 0 ):
            print("A",end='')

        systemindex = [
            'System_TimeCreated_SystemTime', 'System_Provider_Name', 'System_Provider_Guid',
            'System_EventID_Qualifiers', 'System_EventID_VALUE', 'System_Version', 'System_Level',
            'System_Task', 'System_Opcode', 'System_Keywords', 'System_EventRecordID',
            'System_Correlation_ActivityID', 'System_Correlation_RelatedActivityID',
            'System_Execution_ProcessID', 'System_Execution_ThreadID', 'System_Channel',
            'System_Computer', 'System_Security_UserID']

        System_Provider_Name=event['System']['Provider']['@Name']
        System_Provider_Guid=event['System']['Provider']['@Guid']
        System_EventID_Qualifiers=event['System']['EventID']['@Qualifiers']
        System_EventID_VALUE=event['System']['EventID']['#text']
        System_Version=event['System']['Version']
        System_Level=event['System']['Level']
        System_Task=event['System']['Task']
        System_Opcode=event['System']['Opcode']
        System_Keywords=event['System']['Keywords']
        System_TimeCreated_SystemTime=event['System']['TimeCreated']['@SystemTime']
        System_EventRecordID=event['System']['EventRecordID']
        System_Correlation_ActivityID=event['System']['Correlation']['@ActivityID']
        System_Correlation_RelatedActivityID=event['System']['Correlation']['@RelatedActivityID']
        System_Execution_ProcessID=event['System']['Execution']['@ProcessID']
        System_Execution_ThreadID=event['System']['Execution']['@ThreadID']
        System_Channel=event['System']['Channel']
        System_Computer=event['System']['Computer']
        System_Security_UserID=event['System']['Security']['@UserID']

        evtcmn = pd.Series([
                System_TimeCreated_SystemTime, System_Provider_Name, System_Provider_Guid, System_EventID_Qualifiers,
                System_EventID_VALUE, System_Version, System_Level, System_Task,System_Opcode, System_Keywords,             
                System_EventRecordID, System_Correlation_ActivityID, System_Correlation_RelatedActivityID,
                System_Execution_ProcessID, System_Execution_ThreadID, System_Channel, System_Computer, System_Security_UserID],
                index=systemindex)

        evtrecid_df = pd.DataFrame([{'System_EventRecordID': System_EventRecordID}])

        # Build DataFrame 
        if ( cnt == 1 ):
            evtsysdf = pd.DataFrame(evtcmn).T
        else:
            row_df = pd.DataFrame([evtcmn])
            evtsysdf = pd.concat([evtsysdf, row_df], axis=0, ignore_index=True)
    
        # UserData ------------------------------------------------------------
        if "UserData" in event:
            if ( cnt % 100 == 0 ):
                print("B",end='')

            usrdata_row_df_raw = pd.DataFrame(event['UserData']).T.reset_index()
            usrdata_row_df = pd.concat([evtrecid_df,usrdata_row_df_raw], axis=1, ignore_index=False)

            # Build DataFrame 
            if ( usrdatacnt == 1 ):
                usrdatadf = usrdata_row_df
                #print(usrdatadf)
            else:
                usrdatadf = pd.concat([usrdatadf, usrdata_row_df], axis=0, ignore_index=True)
    
            usrdatacnt = usrdatacnt+1

        # EventData -----------------------------------------------------------
        if "EventData" in event:
            if ( cnt % 100 == 0 ):
                print("C",end='')
            if event['EventData'] is not None:
                if ( isinstance(event['EventData']['Data'], list) ):
                    evtdata_row_df_raw=pd.DataFrame(event['EventData']['Data']).set_index('@Name').T.reset_index()
                    evtdata_row_df=pd.concat([evtrecid_df, evtdata_row_df_raw], axis=1, ignore_index=False)
    
                    # Build DataFrame 
                    if ( evtdatacnt == 1 ):
                        evtdatadf=evtdata_row_df
                    else:
                        evtdatadf=pd.concat([evtdatadf, evtdata_row_df], axis=0, ignore_index=True)
                else:
                    # TODO: FIX: COMPLETE THIS CASE!!! !!! !!!
                    # If not a list then it's a dict
                    dummy=0
        
                evtdatacnt=evtdatacnt+1

        cnt += 1

    # System DF -------------------------------------------------------------------
    if ( cnt % 100 == 0 ):
        print("D", end='')

    evtsysdf = evtsysdf.set_index('System_EventRecordID')

    evtsysdf_short = evtsysdf.drop(columns=[
        'System_Provider_Name', 'System_Provider_Guid', 'System_EventID_Qualifiers', 'System_Version',
        'System_Level', 'System_Task', 'System_Opcode', 'System_Keywords', 'System_Correlation_ActivityID',
        'System_Correlation_RelatedActivityID', 'System_Execution_ProcessID', 'System_Execution_ThreadID',
        'System_Channel'])

    # UserData DF -----------------------------------------------------------------
    if ( cnt % 100 == 0 ):
        print("E",end='')

    usrdatadf=usrdatadf.drop(columns=['index','@xmlns']).set_index('System_EventRecordID')
    #print(usrdatadf)

    # EventData DF ----------------------------------------------------------------
    if ( cnt % 100 == 0 ):
        print("F",end='')

    evtdatadf=evtdatadf.drop(columns=['index']).set_index('System_EventRecordID')

    # EventFull DF ----------------------------------------------------------------
    if ( cnt % 100 == 0 ):
        print("G",end='')

    if compact:
        evtfull = pd.concat([evtsysdf_short,usrdatadf,evtdatadf],axis=1)
    else:
        evtfull = pd.concat([evtsysdf,usrdatadf,evtdatadf],axis=1)
    
    # De-duplicate column names ----------------------------------------------
    if ( cnt % 100 == 0 ):
        print("H",end='')

    evtcols = pd.Series(evtfull.columns)

    for dup in evtcols[evtcols.duplicated()].unique(): 
        evtcols[evtcols[evtcols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(evtcols == dup))]

    # rename the columns with the cols list.
    evtfull.columns = evtcols

    # Generate evt-specific DFs
    dfs = {}

    return evtfull


def read_evtx(evtxf,verbose=True):
    import os
    
    filename, file_extension = os.path.splitext(evtxf)
    if file_extension == ".evtx":
        evtalldf = evtx2df(evtxf)
    else:
        # True - .xml file
        evtalldf = evtx2df(evtxf,True)

    dfs = {}
    dfs["all"] = evtalldf
    # Ok, the "System_TimeCreated_SystemTime" column is "object" and should be of type "datetime", so let's change it
    evtalldf['TimeCreated_SystemTime'] = pd.to_datetime(evtalldf['TimeCreated_SystemTime'])
    # The same happens with "System_EventID_VALUE" which should be an integer
    evtalldf['EventID'] = evtalldf['EventID'].astype(int)

    if verbose:
        print("\n")
        print("Generating pandas dataframes: ")
    evtids=pd.Series(evtalldf['EventID']).drop_duplicates().sort_values(ascending=True)
    for evtid in evtids:
        if verbose:
            print('- %-10s ... ' % (evtid),end='')
        dfs[evtid] = evtalldf.query('EventID == @evtid').dropna(axis=1, how='all')
        if verbose:
            print(' [%s]' % (str(len(dfs[evtid]))))
        # Event-specific tuning
        if evtid == 4624:
            dfs[evtid]['LogonType'] = dfs[evtid]['LogonType'].astype(int)

    if verbose:
        print("\nNOTE: Now you can use the syntax <yourvar>['<datatype>'] to access your dataframe")

    return dfs


# Enrich an evt ID, providing its long description
def evtid_enrich(evtid):
    return evtids['evtid']


# Give an enriched listing of evtid statistics
def evtid_stats(evt):
    counts = evt['EventID'].value_counts()
    evtidssrv = evtidssr()
    evtidstats = pd.concat([counts, evtidssrv], axis=1, keys=['Count','Description']).dropna().astype({'Count': int})
    return evtidstats


def evtidsdf():
    evtidssr = pd.Series(evtids)
    evtidsdf = evtidssr.to_frame('Description')
    return evtidsdf


def evtidssr():
    evtidssr = pd.Series(evtids)
    evtidssr.index.astype('int64')

    return evtidssr


def evt_nonsysusers_stats(evts4624):
    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    print("\nWorkstationName ----------------------------------------------------")
    print(evts4624_nonsysusers['WorkstationName'].value_counts())
    print("\nIPAddress ----------------------------------------------------------")
    print(evts4624_nonsysusers['IpAddress'].value_counts())
    print("\nTargetUserName -----------------------------------------------------")
    print(evts4624_nonsysusers['TargetUserName'].value_counts())
    print("\nTargetUserSid ------------------------------------------------------")
    print(evts4624_nonsysusers.groupby(["TargetUserSid", "TargetUserName"]).size())


def evt_nonsysusers_access_stats(evts4624,firstdate,lastdate,freq):
    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    useraccess=evts4624_nonsysusers[["TimeCreated_SystemTime","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('TimeCreated_SystemTime')

    x=useraccess.loc[firstdate:lastdate].groupby([pd.Grouper(freq=freq), "WorkstationName", "IpAddress",'TargetUserName','LogonType']).size()

    # Convert multi-Index to DF
    y = pd.DataFrame(x)
    y.reset_index(inplace=True)
    y['WorkstationName'] = y['WorkstationName'].str.lower()
    y.columns = ['TimeCreated_SystemTime','WorkstationName','IpAddress','TargetUserName','LogonType','Count']
    return y


def evt_nonsysusers_access_graph(evts4624,firstdate,lastdate):
    evts4624_nonsysusers=evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    useraccess=evts4624_nonsysusers[["TimeCreated_SystemTime","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('TimeCreated_SystemTime')
    user_access_uwil=useraccess[["WorkstationName", "IpAddress",'TargetUserName','LogonType']].loc[firstdate:lastdate].copy()

    user_access_uwil['WorkstationName'] = user_access_uwil['WorkstationName'].str.lower()
    user_access_uwil['TargetUserName'] = user_access_uwil['TargetUserName'].str.lower()
    user_access_uwil['LogonType'] = user_access_uwil['LogonType'].astype(str)
    user_access_uwil['IP-WN-TU-LT'] = "["+user_access_uwil['TargetUserName']+"]["+user_access_uwil['WorkstationName']+"]["+user_access_uwil['IpAddress']+"]["+user_access_uwil['LogonType']+"]"
    user_access_uwil.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)

    ## Let's do some graphing
    fig, ax0 = plt.subplots()

    label = 'IP-WN-TU-LT'
    ihtl = user_access_uwil['IP-WN-TU-LT']
    
    ihtl = ihtl.sort_values()
    
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
    
    fig.tight_layout()
    plt.show()


def evt_nonsysusers_autoencoder_analysis(evts4624,firstdate,lastdate):
    np.random.seed(8)

    # DATA PREPARATION ------------------------------------
    evts4624_nonsysusers = evts4624[evts4624['TargetUserSid'].str.contains('S-1-5-21-')]
    useraccess = evts4624_nonsysusers[["TimeCreated_SystemTime","WorkstationName", "IpAddress",'TargetUserName','LogonType']].set_index('TimeCreated_SystemTime')
    this_useraccess = useraccess.loc[firstdate:lastdate]
    
    user_access_uwil = this_useraccess[['TargetUserName',"WorkstationName","IpAddress",'LogonType']].copy()
    
    # Lower-case WorkstationName col
    user_access_uwil['WorkstationName'] = user_access_uwil['WorkstationName'].str.lower()
    user_access_uwil['TargetUserName'] = user_access_uwil['TargetUserName'].str.lower()
    user_access_uwil['LogonType'] = user_access_uwil['LogonType'].astype(str)
    
    user_access_uwil_str = user_access_uwil.copy()
    
    user_access_uwil_str['TU-WN-IP-LT'] = "[" + user_access_uwil['TargetUserName'] + "]" + "[" + user_access_uwil['IpAddress'] + "][" + user_access_uwil['LogonType'] + "]"
    user_access_uwil_str.drop(columns=['WorkstationName','IpAddress','TargetUserName','LogonType'],inplace=True)
    user_access_uwil_str = user_access_uwil_str.sort_values(by='TU-WN-IP-LT')

    df = user_access_uwil
    df.head()

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
    
    print("X -> "+str(X.shape))
    print("X_train -> "+str(X_train.shape))
    print("X_test -> "+str(X_test.shape))

    # CREATING THE NEURAL NETWORK ARCHITECTURE ------------
    # There are 4 different input features, and as we plan to use all the features in the autoencoder,
    # we define the number of input neurons to be 4.
    nfeatures = 4
    input_dim = X_train.shape[1]
    encoding_dim = nfeatures-2
    input_layer = Input(shape=(input_dim,))

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
    
    X_train = np.array(X_train)
    autoencoder.fit(X_train, X_train,epochs=40,batch_size=4)

    # DOING PREDICTIONS -----------------------------------

    # Once the model is fitted, we predict the input values by passing the same X_train dataset to the autoencoder's predict method.
    # Next, we calculate the mse values to know whether the autoencoder was able to reconstruct the dataset correctly and how much the reconstruction error was:

    predictions = autoencoder.predict(X_train)
    mse = np.mean(np.power(X_train - predictions, 2), axis=1)

    return mse

# OTHER FUNCTIONS #############################################################


def df_outlier_analysis(indf,sensitivity):
    # TODO --------------------------------------------------------------------
    # Ideas:
    # - Eliminate cols with a lot of varition based on .nunique() output
    # -------------------------------------------------------------------------

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


def exefile_analysis(exefs, thisexef_path):
    # Description:
    #     Analysis of the multiple instances of a specific file on many hosts
    #     - ...

    # TODO: Think how Ransom Attacks can be detected with this analysis technique

    # Variables
    exef_intg_max_occs = 3

    print("EXEFILE ANALYSIS --------------------------------------------------------\n")

    # Select instances of exef
    exefs[exefs['path'].str.contains(thisexef_path,case=False)].head()
    thisexefs = exefs[exefs['path'].str.contains(thisexef_path,case=False)]

    print("fsize   ANALYSIS - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    exefgrps = thisexefs.groupby('fsize');
    exefgrps_groups = exefgrps.groups
    nexefgrps = len(exefgrps_groups)
    print("No.groups: " + str(nexefgrps) + "\n")
    exef_sizes = exefgrps.groups.keys()
    exef_sizes_occs = exefgrps.size()
    print("Groups (sorted by no. occurrances of 'fsize'):    ")
    print(exef_sizes_occs.sort_values(ascending=False))

    print("Interesting (no. occurrences <=" + str(exef_intg_max_occs) + "):    ")
    exef_intg=exefgrps.filter(lambda x: len(x) <= exef_intg_max_occs)
    print(exef_intg)

    
def unique_files_folder_analysis(exefs, thisexed_path, exef_intg_max_occs, compop='==', recurse=False, prevdays=0, tsfield='m', verbose=False):
    # TODO:
    # - Include "recurse" option so the sub-folders can be included or excluded

    if compop not in ['>', '<', '>=', '==', '<=']:
        print("Invalid Comparison Operator: "+compop)
        return False

    regexrec=thisexed_path+"/"
    regexnorec=thisexed_path+"/[^/]*$"

    if recurse == True:
        thisexefsrec=exefs[exefs['path'].str.contains(regexrec,case=False,regex=True)]
        nexefsrec=len(thisexefsrec)
        thisexefs=thisexefsrec
        if verbose == True:
            print("No. files (recursive):     "+str(nexefsrec)+"\n")
    else:
        thisexefsnorec=exefs[exefs['path'].str.contains(regexnorec,case=False,regex=True)]
        nexefsnorec=len(thisexefsnorec)
        thisexefs=thisexefsnorec
        if verbose == True:
            print("No. files (non-recursive): "+str(nexefsnorec)+"\n")

    exefgrps = thisexefs.groupby('path-hash')
    exefgrps_groups = exefgrps.groups
    nexefgrps = len(exefgrps_groups)
    exef_sizes=exefgrps.groups.keys()
    exef_sizes_occs=exefgrps.size()
    if verbose == True:
        print("phash ANALYSIS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
        print("RECURSION: "+str(recurse))
        print("No.groups: "+str(nexefgrps)+"\n")

    if prevdays == 0 :
        exef_intg = exefgrps.filter(lambda x: eval( str(len(x)) + compop + str(exef_intg_max_occs)) )
    else:
        print("No. Interesting (no. occurrences <=" + str(exef_intg_max_occs) + "): " + str(nexef_intg) + "\n")
        lastmtime = exef_intg.sort_values(by="mtime").tail(1)['mtime']
        print("Last mtime: " + str(lastmtime))
        prevdate = lastmtime + pd.DateOffset(days=-prevdays)
        print("Previous Date: "+  str(prevdate))

    return exef_intg


def exefs_analysis(exefs,thisexef_path):
    # Description:
    #     Macro analysis of all EXEs in the exefs df as a whole
    #     - Files that appear only a few times
    #     - ...

    # path-hash analysis  ===============================================================
    rare_phash_occs = 3
    thisexefilegrps = thisexefs.groupby('path-hash')
    thisexefilegrps_groups = thisexefilegrps.groups
    nthisexefilegrps = len(thisexefilegrps_groups)
    thisexefile_phash = thisexefilegrps.groups.keys()
    thisexefile_phash_occs = thisexefilegrps.size()
    print("Groups (sorted by no. occurrances of 'path-hash'):    ")
    print(thisexefile_phash_occs.sort_values(ascending=False))
    thisexefile_phash_rare = thisexefile_phash_occs[thisexefile_phash_occs == 1].sort_values(ascending=False)

    # Rare Files Analysis ---------------------------------------------------------------
    # Files which appear only n times in the whole host set. 
    # Since we are grouping by phash, this means they appear once in each of the n hosts
    # If n=1 -> file appears only in 1 host
    exef_intg = exefgrps.filter(lambda x: len(x) <= n)    

    # Creation time analysis ------------------------------------------------------------
    # Files created recently
    # TODO: Surely can be improved by specifying a relative date rather than absolute 
    exefs[exefs['atime'] > '2019-12-01']


###############################################################################
# KNOWLEDGE
###############################################################################

# PROCESSES ###################################################################
critical_processes = [
    'System', 'smss.exe', 'wininit.exe', 'RuntimeBroker.exe', 'taskhostw.exe', 'winlogon.exe', 
    'csrss.exe', 'services.exe', 'svchost.exe', 'lsaiso.exe', 'lsass.exe', 'explorer.exe']

boot_start_processes = [
    'System', 'smss.exe', 'wininit.exe', 'winlogon.exe', 'csrss.exe', 'services.exe', 'lsaiso.exe', 
    'lsass.exe' ]

process_parents = pd.DataFrame([
    ['System', ''],
    ['smss.exe', 'System'],
    ['wininit.exe', 'smss.exe'],
    ['RuntimeBroker.exe', 'svchost.exe'],
    ['taskhostw.exe', 'svchost.exe'],
    ['winlogon.exe', 'smss.exe'],
    ['csrss.exe', 'smss.exe'],
    ['services.exe', 'wininit.exe'],
    ['svchost.exe', 'services.exe'],
    ['lsaiso.exe', 'wininit.exe'],
    ['lsass.exe', 'wininit.exe'],
    ['explorer.exe', 'userinit.exe']],
    columns=['Child', 'Parent'])

# EVENT IDs ###################################################################
evtids = {
    1100: 'The event logging service has shut down',
    1101: 'Audit events have been dropped by the transport.',
    1102: 'The audit log was cleared',
    1104: 'The security Log is now full',
    1105: 'Event log automatic backup',
    1108: 'The event logging service encountered an error',
    4608: 'Windows is starting up',
    4609: 'Windows is shutting down',
    4610: 'An authentication package has been loaded by the Local Security Authority',
    4611: 'A trusted logon process has been registered with the Local Security Authority',
    4612: 'Internal resources allocated for the queuing of audit messages have been exhausted, leading to the loss of some audits.',
    4614: 'A notification package has been loaded by the Security Account Manager.',
    4615: 'Invalid use of LPC port',
    4616: 'The system time was changed.',
    4618: 'A monitored security event pattern has occurred',
    4621: 'Administrator recovered system from CrashOnAuditFail',
    4622: 'A security package has been loaded by the Local Security Authority.',
    4624: 'An account was successfully logged on',
    4625: 'An account failed to log on',
    4626: 'User/Device claims information',
    4627: 'Group membership information.',
    4634: 'An account was logged off',
    4646: 'IKE DoS-prevention mode started',
    4647: 'User initiated logoff',
    4648: 'A logon was attempted using explicit credentials',
    4649: 'A replay attack was detected',
    4650: 'An IPsec Main Mode security association was established',
    4651: 'An IPsec Main Mode security association was established',
    4652: 'An IPsec Main Mode negotiation failed',
    4653: 'An IPsec Main Mode negotiation failed',
    4654: 'An IPsec Quick Mode negotiation failed',
    4655: 'An IPsec Main Mode security association ended',
    4656: 'A handle to an object was requested',
    4657: 'A registry value was modified',
    4658: 'The handle to an object was closed',
    4659: 'A handle to an object was requested with intent to delete',
    4660: 'An object was deleted',
    4661: 'A handle to an object was requested',
    4662: 'An operation was performed on an object',
    4663: 'An attempt was made to access an object',
    4664: 'An attempt was made to create a hard link',
    4665: 'An attempt was made to create an application client context.',
    4666: 'An application attempted an operation',
    4667: 'An application client context was deleted',
    4668: 'An application was initialized',
    4670: 'Permissions on an object were changed',
    4671: 'An application attempted to access a blocked ordinal through the TBS',
    4672: 'Special privileges assigned to new logon',
    4673: 'A privileged service was called',
    4674: 'An operation was attempted on a privileged object',
    4675: 'SIDs were filtered',
    4688: 'A new process has been created',
    4689: 'A process has exited',
    4690: 'An attempt was made to duplicate a handle to an object',
    4691: 'Indirect access to an object was requested',
    4692: 'Backup of data protection master key was attempted',
    4693: 'Recovery of data protection master key was attempted',
    4694: 'Protection of auditable protected data was attempted',
    4695: 'Unprotection of auditable protected data was attempted',
    4696: 'A primary token was assigned to process',
    4697: 'A service was installed in the system',
    4698: 'A scheduled task was created',
    4699: 'A scheduled task was deleted',
    4700: 'A scheduled task was enabled',
    4701: 'A scheduled task was disabled',
    4702: 'A scheduled task was updated',
    4703: 'A token right was adjusted',
    4704: 'A user right was assigned',
    4705: 'A user right was removed',
    4706: 'A new trust was created to a domain',
    4707: 'A trust to a domain was removed',
    4709: 'IPsec Services was started',
    4710: 'IPsec Services was disabled',
    4711: 'PAStore Engine (1%)',
    4712: 'IPsec Services encountered a potentially serious failure',
    4713: 'Kerberos policy was changed',
    4714: 'Encrypted data recovery policy was changed',
    4715: 'The audit policy (SACL) on an object was changed',
    4716: 'Trusted domain information was modified',
    4717: 'System security access was granted to an account',
    4718: 'System security access was removed from an account',
    4719: 'System audit policy was changed',
    4720: 'A user account was created',
    4722: 'A user account was enabled',
    4723: 'An attempt was made to change an account\'s password',
    4724: 'An attempt was made to reset an accounts password',
    4725: 'A user account was disabled',
    4726: 'A user account was deleted',
    4727: 'A security-enabled global group was created',
    4728: 'A member was added to a security-enabled global group',
    4729: 'A member was removed from a security-enabled global group',
    4730: 'A security-enabled global group was deleted',
    4731: 'A security-enabled local group was created',
    4732: 'A member was added to a security-enabled local group',
    4733: 'A member was removed from a security-enabled local group',
    4734: 'A security-enabled local group was deleted',
    4735: 'A security-enabled local group was changed',
    4737: 'A security-enabled global group was changed',
    4738: 'A user account was changed',
    4739: 'Domain Policy was changed',
    4740: 'A user account was locked out',
    4741: 'A computer account was created',
    4742: 'A computer account was changed',
    4743: 'A computer account was deleted',
    4744: 'A security-disabled local group was created',
    4745: 'A security-disabled local group was changed',
    4746: 'A member was added to a security-disabled local group',
    4747: 'A member was removed from a security-disabled local group',
    4748: 'A security-disabled local group was deleted',
    4749: 'A security-disabled global group was created',
    4750: 'A security-disabled global group was changed',
    4751: 'A member was added to a security-disabled global group',
    4752: 'A member was removed from a security-disabled global group',
    4753: 'A security-disabled global group was deleted',
    4754: 'A security-enabled universal group was created',
    4755: 'A security-enabled universal group was changed',
    4756: 'A member was added to a security-enabled universal group',
    4757: 'A member was removed from a security-enabled universal group',
    4758: 'A security-enabled universal group was deleted',
    4759: 'A security-disabled universal group was created',
    4760: 'A security-disabled universal group was changed',
    4761: 'A member was added to a security-disabled universal group',
    4762: 'A member was removed from a security-disabled universal group',
    4763: 'A security-disabled universal group was deleted',
    4764: 'A groups type was changed',
    4765: 'SID History was added to an account',
    4766: 'An attempt to add SID History to an account failed',
    4767: 'A user account was unlocked',
    4768: 'A Kerberos authentication ticket (TGT) was requested',
    4769: 'A Kerberos service ticket was requested',
    4770: 'A Kerberos service ticket was renewed',
    4771: 'Kerberos pre-authentication failed',
    4772: 'A Kerberos authentication ticket request failed',
    4773: 'A Kerberos service ticket request failed',
    4774: 'An account was mapped for logon',
    4775: 'An account could not be mapped for logon',
    4776: 'The domain controller attempted to validate the credentials for an account',
    4777: 'The domain controller failed to validate the credentials for an account',
    4778: 'A session was reconnected to a Window Station',
    4779: 'A session was disconnected from a Window Station',
    4780: 'The ACL was set on accounts which are members of administrators groups',
    4781: 'The name of an account was changed',
    4782: 'The password hash an account was accessed',
    4783: 'A basic application group was created',
    4784: 'A basic application group was changed',
    4785: 'A member was added to a basic application group',
    4786: 'A member was removed from a basic application group',
    4787: 'A non-member was added to a basic application group',
    4788: 'A non-member was removed from a basic application group..',
    4789: 'A basic application group was deleted',
    4790: 'An LDAP query group was created',
    4791: 'A basic application group was changed',
    4792: 'An LDAP query group was deleted',
    4793: 'The Password Policy Checking API was called',
    4794: 'An attempt was made to set the Directory Services Restore Mode administrator password',
    4797: 'An attempt was made to query the existence of a blank password for an account',
    4798: 'A user\'s local group membership was enumerated.',
    4799: 'A security-enabled local group membership was enumerated',
    4800: 'The workstation was locked',
    4801: 'The workstation was unlocked',
    4802: 'The screen saver was invoked',
    4803: 'The screen saver was dismissed',
    4816: 'RPC detected an integrity violation while decrypting an incoming message',
    4817: 'Auditing settings on object were changed.',
    4818: 'Proposed Central Access Policy does not grant the same access permissions as the current Central Access Policy',
    4819: 'Central Access Policies on the machine have been changed',
    4820: 'A Kerberos Ticket-granting-ticket (TGT) was denied because the device does not meet the access control restrictions',
    4821: 'A Kerberos service ticket was denied because the user, device, or both does not meet the access control restrictions',
    4822: 'NTLM authentication failed because the account was a member of the Protected User group',
    4823: 'NTLM authentication failed because access control restrictions are required',
    4824: 'Kerberos preauthentication by using DES or RC4 failed because the account was a member of the Protected User group',
    4825: 'A user was denied the access to Remote Desktop. By default, users are allowed to connect only if they are members of the Remote Desktop Users group or Administrators group',
    4826: 'Boot Configuration Data loaded',
    4830: 'SID History was removed from an account',
    4864: 'A namespace collision was detected',
    4865: 'A trusted forest information entry was added',
    4866: 'A trusted forest information entry was removed',
    4867: 'A trusted forest information entry was modified',
    4868: 'The certificate manager denied a pending certificate request',
    4869: 'Certificate Services received a resubmitted certificate request',
    4870: 'Certificate Services revoked a certificate',
    4871: 'Certificate Services received a request to publish the certificate revocation list (CRL)',
    4872: 'Certificate Services published the certificate revocation list (CRL)',
    4873: 'A certificate request extension changed',
    4874: 'One or more certificate request attributes changed.',
    4875: 'Certificate Services received a request to shut down',
    4876: 'Certificate Services backup started',
    4877: 'Certificate Services backup completed',
    4878: 'Certificate Services restore started',
    4879: 'Certificate Services restore completed',
    4880: 'Certificate Services started',
    4881: 'Certificate Services stopped',
    4882: 'The security permissions for Certificate Services changed',
    4883: 'Certificate Services retrieved an archived key',
    4884: 'Certificate Services imported a certificate into its database',
    4885: 'The audit filter for Certificate Services changed',
    4886: 'Certificate Services received a certificate request',
    4887: 'Certificate Services approved a certificate request and issued a certificate',
    4888: 'Certificate Services denied a certificate request',
    4889: 'Certificate Services set the status of a certificate request to pending',
    4890: 'The certificate manager settings for Certificate Services changed.',
    4891: 'A configuration entry changed in Certificate Services',
    4892: 'A property of Certificate Services changed',
    4893: 'Certificate Services archived a key',
    4894: 'Certificate Services imported and archived a key',
    4895: 'Certificate Services published the CA certificate to Active Directory Domain Services',
    4896: 'One or more rows have been deleted from the certificate database',
    4897: 'Role separation enabled',
    4898: 'Certificate Services loaded a template',
    4899: 'A Certificate Services template was updated',
    4900: 'Certificate Services template security was updated',
    4902: 'The Per-user audit policy table was created',
    4904: 'An attempt was made to register a security event source',
    4905: 'An attempt was made to unregister a security event source',
    4906: 'The CrashOnAuditFail value has changed',
    4907: 'Auditing settings on object were changed',
    4908: 'Special Groups Logon table modified',
    4909: 'The local policy settings for the TBS were changed',
    4910: 'The group policy settings for the TBS were changed',
    4911: 'Resource attributes of the object were changed',
    4912: 'Per User Audit Policy was changed',
    4913: 'Central Access Policy on the object was changed',
    4928: 'An Active Directory replica source naming context was established',
    4929: 'An Active Directory replica source naming context was removed',
    4930: 'An Active Directory replica source naming context was modified',
    4931: 'An Active Directory replica destination naming context was modified',
    4932: 'Synchronization of a replica of an Active Directory naming context has begun',
    4933: 'Synchronization of a replica of an Active Directory naming context has ended',
    4934: 'Attributes of an Active Directory object were replicated',
    4935: 'Replication failure begins',
    4936: 'Replication failure ends',
    4937: 'A lingering object was removed from a replica',
    4944: 'The following policy was active when the Windows Firewall started',
    4945: 'A rule was listed when the Windows Firewall started',
    4946: 'A change has been made to Windows Firewall exception list. A rule was added',
    4947: 'A change has been made to Windows Firewall exception list. A rule was modified',
    4948: 'A change has been made to Windows Firewall exception list. A rule was deleted',
    4949: 'Windows Firewall settings were restored to the default values',
    4950: 'A Windows Firewall setting has changed',
    4951: 'A rule has been ignored because its major version number was not recognized by Windows Firewall',
    4952: 'Parts of a rule have been ignored because its minor version number was not recognized by Windows Firewall',
    4953: 'A rule has been ignored by Windows Firewall because it could not parse the rule',
    4954: 'Windows Firewall Group Policy settings has changed. The new settings have been applied',
    4956: 'Windows Firewall has changed the active profile',
    4957: 'Windows Firewall did not apply the following rule',
    4958: 'Windows Firewall did not apply the following rule because the rule referred to items not configured on this computer',
    4960: 'IPsec dropped an inbound packet that failed an integrity check',
    4961: 'IPsec dropped an inbound packet that failed a replay check',
    4962: 'IPsec dropped an inbound packet that failed a replay check',
    4963: 'IPsec dropped an inbound clear text packet that should have been secured',
    4964: 'Special groups have been assigned to a new logon',
    4965: 'IPsec received a packet from a remote computer with an incorrect Security Parameter Index (SPI).',
    4976: 'During Main Mode negotiation, IPsec received an invalid negotiation packet.',
    4977: 'During Quick Mode negotiation, IPsec received an invalid negotiation packet.',
    4978: 'During Extended Mode negotiation, IPsec received an invalid negotiation packet.',
    4979: 'IPsec Main Mode and Extended Mode security associations were established.',
    4980: 'IPsec Main Mode and Extended Mode security associations were established',
    4981: 'IPsec Main Mode and Extended Mode security associations were established',
    4982: 'IPsec Main Mode and Extended Mode security associations were established',
    4983: 'An IPsec Extended Mode negotiation failed',
    4984: 'An IPsec Extended Mode negotiation failed',
    4985: 'The state of a transaction has changed',
    5024: 'The Windows Firewall Service has started successfully',
    5025: 'The Windows Firewall Service has been stopped',
    5027: 'The Windows Firewall Service was unable to retrieve the security policy from the local storage',
    5028: 'The Windows Firewall Service was unable to parse the new security policy.',
    5029: 'The Windows Firewall Service failed to initialize the driver',
    5030: 'The Windows Firewall Service failed to start',
    5031: 'The Windows Firewall Service blocked an application from accepting incoming connections on the network.',
    5032: 'Windows Firewall was unable to notify the user that it blocked an application from accepting incoming connections on the network',
    5033: 'The Windows Firewall Driver has started successfully',
    5034: 'The Windows Firewall Driver has been stopped',
    5035: 'The Windows Firewall Driver failed to start',
    5037: 'The Windows Firewall Driver detected critical runtime error. Terminating',
    5038: 'Code integrity determined that the image hash of a file is not valid',
    5039: 'A registry key was virtualized.',
    5040: 'A change has been made to IPsec settings. An Authentication Set was added.',
    5041: 'A change has been made to IPsec settings. An Authentication Set was modified',
    5042: 'A change has been made to IPsec settings. An Authentication Set was deleted',
    5043: 'A change has been made to IPsec settings. A Connection Security Rule was added',
    5044: 'A change has been made to IPsec settings. A Connection Security Rule was modified',
    5045: 'A change has been made to IPsec settings. A Connection Security Rule was deleted',
    5046: 'A change has been made to IPsec settings. A Crypto Set was added',
    5047: 'A change has been made to IPsec settings. A Crypto Set was modified',
    5048: 'A change has been made to IPsec settings. A Crypto Set was deleted',
    5049: 'An IPsec Security Association was deleted',
    5050: 'An attempt to programmatically disable the Windows Firewall using a call to INetFwProfile.FirewallEnabled(FALSE',
    5051: 'A file was virtualized',
    5056: 'A cryptographic self test was performed',
    5057: 'A cryptographic primitive operation failed',
    5058: 'Key file operation',
    5059: 'Key migration operation',
    5060: 'Verification operation failed',
    5061: 'Cryptographic operation',
    5062: 'A kernel-mode cryptographic self test was performed',
    5063: 'A cryptographic provider operation was attempted',
    5064: 'A cryptographic context operation was attempted',
    5065: 'A cryptographic context modification was attempted',
    5066: 'A cryptographic function operation was attempted',
    5067: 'A cryptographic function modification was attempted',
    5068: 'A cryptographic function provider operation was attempted',
    5069: 'A cryptographic function property operation was attempted',
    5070: 'A cryptographic function property operation was attempted',
    5071: 'Key access denied by Microsoft key distribution service',
    5120: 'OCSP Responder Service Started',
    5121: 'OCSP Responder Service Stopped',
    5122: 'A Configuration entry changed in the OCSP Responder Service',
    5123: 'A configuration entry changed in the OCSP Responder Service',
    5124: 'A security setting was updated on OCSP Responder Service',
    5125: 'A request was submitted to OCSP Responder Service',
    5126: 'Signing Certificate was automatically updated by the OCSP Responder Service',
    5127: 'The OCSP Revocation Provider successfully updated the revocation information',
    5136: 'A directory service object was modified',
    5137: 'A directory service object was created',
    5138: 'A directory service object was undeleted',
    5139: 'A directory service object was moved',
    5140: 'A network share object was accessed',
    5141: 'A directory service object was deleted',
    5142: 'A network share object was added.',
    5143: 'A network share object was modified',
    5144: 'A network share object was deleted.',
    5145: 'A network share object was checked to see whether client can be granted desired access',
    5146: 'The Windows Filtering Platform has blocked a packet',
    5147: 'A more restrictive Windows Filtering Platform filter has blocked a packet',
    5148: 'The Windows Filtering Platform has detected a DoS attack and entered a defensive mode; packets associated with this attack will be discarded.',
    5149: 'The DoS attack has subsided and normal processing is being resumed.',
    5150: 'The Windows Filtering Platform has blocked a packet.',
    5151: 'A more restrictive Windows Filtering Platform filter has blocked a packet.',
    5152: 'The Windows Filtering Platform blocked a packet',
    5153: 'A more restrictive Windows Filtering Platform filter has blocked a packet',
    5154: 'The Windows Filtering Platform has permitted an application or service to listen on a port for incoming connections',
    5155: 'The Windows Filtering Platform has blocked an application or service from listening on a port for incoming connections',
    5156: 'The Windows Filtering Platform has allowed a connection',
    5157: 'The Windows Filtering Platform has blocked a connection',
    5158: 'The Windows Filtering Platform has permitted a bind to a local port',
    5159: 'The Windows Filtering Platform has blocked a bind to a local port',
    5168: 'Spn check for SMB/SMB2 fails.',
    5169: 'A directory service object was modified',
    5170: 'A directory service object was modified during a background cleanup task',
    5376: 'Credential Manager credentials were backed up',
    5377: 'Credential Manager credentials were restored from a backup',
    5378: 'The requested credentials delegation was disallowed by policy',
    5379: 'Credential Manager credentials were read',
    5380: 'Vault Find Credential',
    5381: 'Vault credentials were read',
    5382: 'Vault credentials were read',
    5440: 'The following callout was present when the Windows Filtering Platform Base Filtering Engine started',
    5441: 'The following filter was present when the Windows Filtering Platform Base Filtering Engine started',
    5442: 'The following provider was present when the Windows Filtering Platform Base Filtering Engine started',
    5443: 'The following provider context was present when the Windows Filtering Platform Base Filtering Engine started',
    5444: 'The following sub-layer was present when the Windows Filtering Platform Base Filtering Engine started',
    5446: 'A Windows Filtering Platform callout has been changed',
    5447: 'A Windows Filtering Platform filter has been changed',
    5448: 'A Windows Filtering Platform provider has been changed',
    5449: 'A Windows Filtering Platform provider context has been changed',
    5450: 'A Windows Filtering Platform sub-layer has been changed',
    5451: 'An IPsec Quick Mode security association was established',
    5452: 'An IPsec Quick Mode security association ended',
    5453: 'An IPsec negotiation with a remote computer failed because the IKE and AuthIP IPsec Keying Modules (IKEEXT) service is not started',
    5456: 'PAStore Engine applied Active Directory storage IPsec policy on the computer',
    5457: 'PAStore Engine failed to apply Active Directory storage IPsec policy on the computer',
    5458: 'PAStore Engine applied locally cached copy of Active Directory storage IPsec policy on the computer',
    5459: 'PAStore Engine failed to apply locally cached copy of Active Directory storage IPsec policy on the computer',
    5460: 'PAStore Engine applied local registry storage IPsec policy on the computer',
    5461: 'PAStore Engine failed to apply local registry storage IPsec policy on the computer',
    5462: 'PAStore Engine failed to apply some rules of the active IPsec policy on the computer',
    5463: 'PAStore Engine polled for changes to the active IPsec policy and detected no changes',
    5464: 'PAStore Engine polled for changes to the active IPsec policy, detected changes, and applied them to IPsec Services',
    5465: 'PAStore Engine received a control for forced reloading of IPsec policy and processed the control successfully',
    5466: 'PAStore Engine polled for changes to the Active Directory IPsec policy, determined that Active Directory cannot be reached, and will use the cached copy of the Active Directory IPsec policy instead',
    5467: 'PAStore Engine polled for changes to the Active Directory IPsec policy, determined that Active Directory can be reached, and found no changes to the policy',
    5468: 'PAStore Engine polled for changes to the Active Directory IPsec policy, determined that Active Directory can be reached, found changes to the policy, and applied those changes',
    5471: 'PAStore Engine loaded local storage IPsec policy on the computer',
    5472: 'PAStore Engine failed to load local storage IPsec policy on the computer',
    5473: 'PAStore Engine loaded directory storage IPsec policy on the computer',
    5474: 'PAStore Engine failed to load directory storage IPsec policy on the computer',
    5477: 'PAStore Engine failed to add quick mode filter',
    5478: 'IPsec Services has started successfully',
    5479: 'IPsec Services has been shut down successfully',
    5480: 'IPsec Services failed to get the complete list of network interfaces on the computer',
    5483: 'IPsec Services failed to initialize RPC server. IPsec Services could not be started',
    5484: 'IPsec Services has experienced a critical failure and has been shut down',
    5485: 'IPsec Services failed to process some IPsec filters on a plug-and-play event for network interfaces',
    5632: 'A request was made to authenticate to a wireless network',
    5633: 'A request was made to authenticate to a wired network',
    5712: 'A Remote Procedure Call (RPC) was attempted',
    5888: 'An object in the COM+ Catalog was modified',
    5889: 'An object was deleted from the COM+ Catalog',
    5890: 'An object was added to the COM+ Catalog',
    6144: 'Security policy in the group policy objects has been applied successfully',
    6145: 'One or more errors occured while processing security policy in the group policy objects',
    6272: 'Network Policy Server granted access to a user',
    6273: 'Network Policy Server denied access to a user',
    6274: 'Network Policy Server discarded the request for a user',
    6275: 'Network Policy Server discarded the accounting request for a user',
    6276: 'Network Policy Server quarantined a user',
    6277: 'Network Policy Server granted access to a user but put it on probation because the host did not meet the defined health policy',
    6278: 'Network Policy Server granted full access to a user because the host met the defined health policy',
    6279: 'Network Policy Server locked the user account due to repeated failed authentication attempts',
    6280: 'Network Policy Server unlocked the user account',
    6281: 'Code Integrity determined that the page hashes of an image file are not valid...',
    6400: 'BranchCache: Received an incorrectly formatted response while discovering availability of content.',
    6401: 'BranchCache: Received invalid data from a peer. Data discarded.',
    6402: 'BranchCache: The message to the hosted cache offering it data is incorrectly formatted.',
    6403: 'BranchCache: The hosted cache sent an incorrectly formatted response to the client\'s message to offer it data.',
    6404: 'BranchCache: Hosted cache could not be authenticated using the provisioned SSL certificate.',
    6405: 'BranchCache: %2 instance(s) of event id %1 occurred.',
    6406: '%1 registered to Windows Firewall to control filtering for the following:',
    6407: '%1',
    6408: 'Registered product %1 failed and Windows Firewall is now controlling the filtering for %2.',
    6409: 'BranchCache: A service connection point object could not be parsed',
    6410: 'Code integrity determined that a file does not meet the security requirements to load into a process. This could be due to the use of shared sections or other issues',
    6416: 'A new external device was recognized by the system.',
    6417: 'The FIPS mode crypto selftests succeeded',
    6418: 'The FIPS mode crypto selftests failed',
    6419: 'A request was made to disable a device',
    6420: 'A device was disabled',
    6421: 'A request was made to enable a device',
    6422: 'A device was enabled',
    6423: 'The installation of this device is forbidden by system policy',
    6424: 'The installation of this device was allowed, after having previously been forbidden by policy',
    8191: 'Highest System-Defined Audit Message Value',
}
