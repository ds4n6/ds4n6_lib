#
# Description: library to apply Graph Data Science in several forensics artifacts
#

#############################################################################################
# INFO
#############################################################################################
# Recommended "import as": d4mlg

#############################################################################################
# IMPORTS
#############################################################################################
import numpy as np
import pandas as pd
import networkx as nx
from ast import literal_eval
from collections import Counter

#############################################################################################
# FUNCTIONS
#############################################################################################
def build_lm_dataset(dset, mode='hostname', path='',codify=False):
    """ Function to build a Lateral Movement (LM) dataset
    Syntax: build_lm_dataset(dset="<dset>", mode="<mode>", path="<path>", codify="<codify>")
    Args:
        dset (pandas.core.frame.DataFrame): Event Log Dataset. Min. columns: ['time','event_id','hostname','source_ip','source_hostname','logon_type','remote_user']
        mode (str): build mode. 'hostname' to create the LM dataset only by using known hostnames. 'ip_addr' to use IP address for unknown hostnames
        path (str): path to store the LM datasets
            - ds4n6_lm_dataset.csv: dataset with LMs
            - ds4n6_neo4j_dataset.csv: dataset with LMs to be loaded in Neo4j
        codify (bool): 'True' to codify user and host names. 'False' otherwise. Default 'False'
    Returns:
        lm_dset (core.frame.DataFrame): dataset with LMs
    """
    dset    = _clear_dataset(dset)
    dict_ip = _create_ip_dict(dset, path)
    dset    = _ip_to_hostname(dset, dict_ip, mode=mode)
    if codify:     
        cdset = _codify_dataset(dset)
        dset  = cdset[0]
        with open(path + 'dictionary.txt','w') as data: 
            data.write(str(cdset[1]))
            data.write(str(cdset[2]))

    lm_dset,neo4j_dset = _build_lm_dataset(dset)
    lm_dset.to_csv(path + 'ds4n6_lm_dataset.csv', index=False)
    neo4j_dset.to_csv(path + 'ds4n6_neo4j_dataset.csv', index=False)
    return lm_dset

def find_lm_anomalies(lm_dset, model, from_date, to_date, top_n=50, neo4j=True, path=''):
    """ Function to detect anomalous Lateral Movement (LM) with Machine Learning
    Syntax: find_lm_anomalies(lm_dset="<lm_dset>", model="<model>", from_date="<from_date>", to_date="<to_date>", top_n="<top_n>", neo4j="<neo4j>", path="<path>")
    Args:
        lm_dset (pandas.core.frame.DataFrame): Lateral Movement dataset (output of build_lm_dataset func.)
        model (str): ML model to be used. Supported models: ('s2s_lstm', 'transformer')
        from_date (str): init date of the training dataset
        to_date (str): end date of the training dataset
        top_n (str): number of anomalous LMs to be detected
        neo4j (bool): 'True' to export the output to Neo4j format. 'False' otherwise
        path (str): path to store Neo4j output datasets
            - <user>.csv: dataset with anomalous LMs by user
            - <user>_full.csv: dataset with all user activity in the input dataset (lm_dset)
    """
    if model == "transformer":
        from ds4n6_lib.ml_models.transformer import Seq2seqData, Autoencoder
    elif model == "s2s_lstm":
        from ds4n6_lib.ml_models.seq2seq_lstm import Seq2seqData, Autoencoder
    else:
        raise ValueError("Error: model '" + model + "' not supported. Try 's2s_lstm' or 'transformer'")

    data = Seq2seqData()
    ml_dset = data.load_path_dataset(lm_dset, from_date, to_date, min_count=0)
    train_x, train_y = data.process_train_data(ml_dset)
    data.build_train_dset(train_x, train_y)
    
    model = Autoencoder(embed_dim=16, latent_dim=300, data=data)
    model.build_autoencoder()
    model.fit_autoencoder()
    err_mtrx = model.get_anomalies(train_x)
    out = _print_top_anomalies(top_n, err_mtrx, ml_dset)
    if neo4j:
        _safe_anomalies_neo4j(top_n, err_mtrx, ml_dset, path)
    return out

#############################################################################################
# AUX. FUNCTIONS
#############################################################################################
def _build_adjancency(dataframe, origin, destination):
    adj_dict_ = {}
    graph = nx.from_pandas_edgelist(dataframe, source = origin, target = destination, create_using=nx.DiGraph())
    ajc = graph.adjacency()

    for m,n in ajc:
        adj_dict_[m]=list(n.keys())
    adj_dict = {k: v for k, v in adj_dict_.items() if v}
    return adj_dict

def _node_paths(graph_adj_dict, init_path):
    final_paths = [init_path]
    idx = 0
    iters = len(final_paths)
    while iters>idx:
        path = final_paths[idx]
        if path[-1] in graph_adj_dict:
            for node in  graph_adj_dict[path[-1]]:
                if node not in path:
                    new_path = path + [node]
                    final_paths.append(new_path)
        idx += 1
        iters = len(final_paths)    
    return final_paths

def _get_single_paths(paths):
    s_paths=[]
    aux_paths = paths.copy()
    for idx,i in enumerate(paths):  
        aux_paths.remove(i)
        for j in aux_paths:
            if (''.join(str(x) for x in i)) in ((''.join(str(y) for y in j))):
                break
            else:
                continue
        else:
            s_paths.append(i)
        aux_paths = paths.copy()
    return s_paths

def _get_paths(graph_adj):
    nodes = list(graph_adj.keys())
    all_paths = []
    for node in nodes:
        node_paths = _node_paths(graph_adj, [node])
        all_paths += node_paths
    return _get_single_paths(all_paths)

def _build_path_df(adjacency_df):
    path_df_=[]
    for i in adjacency_df:
        paths=_get_paths(i[2])
        [path_df_.append([i[0], i[1], path]) for path in paths]
    return path_df_

def _del_local(lm_dataset): # Delete LMs with len=1
    lenght = []
    for i in lm_dataset['path']:
        lenght.append(str(len(i)))
    lm_dataset['lenght'] = lenght
    lm_dataset = lm_dataset[~lm_dataset['lenght'].str.contains('1')]
    lm_dataset = lm_dataset.drop(columns=['lenght'])
    return lm_dataset

def _clear_dataset(dataframe):
    tools = ['sabonis','masstin']
    df = dataframe.astype(str).copy()
    if 'D4_DataType_' in df.columns:
        if df['D4_DataType_'][0] in tools:
            df = df.rename(columns={'Timestamp_': 'time',
                                     'EventID_': 'event_id',
                                     'Computer_':'hostname',
                                     'SourceIP_':'source_ip',
                                     'SourceComputer_':'source_hostname',
                                     'LogonType_':'logon_type',
                                     'TargetUserName_':'remote_user'})
            df = df[['time','event_id','hostname','source_ip','source_hostname','logon_type','remote_user']]
        else:
            raise TypeError('D4_DataType_ != sabonis|masstin')
    df1 = df[df['event_id'].isin(['21','24','25','4624','1149'])].copy()
    target_columns = ['hostname', 'source_hostname', 'remote_user']
    for column in target_columns:
        df1[column] = df1[column].str.lower()
    df1['hostname'] = df1['hostname'].str.split('.').apply(lambda x: x[0])
    df1['time'] = pd.to_datetime(df1['time'])
    df1 = df1[~df1['remote_user'].str.contains('$', regex = False)]
    df1 = df1[~df1['remote_user'].isin(['system','network service','anonymous','anonymous logon','nan','d4_null'])]
    new_df = df1

    if True: ### Include Events: 1024 y 1102
        df2 = df[df['event_id'].isin(['1024','1102'])].copy()
        for column in target_columns:
            df2[column] = df2[column].str.lower()
        df2['source_hostname'] = df2['source_hostname'].str.split('.').apply(lambda x: x[0])
        df2['time'] = pd.to_datetime(df2['time'])
        new_df = pd.concat([df1, df2])
    return new_df

def _create_ip_dict(dataframe, path):
    dict_df = dataframe[dataframe['logon_type'].str.contains('3', regex = False)]
    dict_df = dict_df[~dict_df['source_hostname'].isin(['-','nan','none'])]
    dict_df['source_ip'] = np.where(dict_df['source_ip'].str.match("[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*"), dict_df['source_ip'], "d4_null")
    dict_df = dict_df[~dict_df['source_ip'].isin(['d4_null'])]
    host_list=dict_df["source_hostname"].value_counts()
    host_dict = {}
    unreliable_hosts = []

    for hostname in host_list.index:
        host_df = dict_df[dict_df['source_hostname'].isin([hostname])]['source_ip']
        c = Counter(host_df)
        ip = c.most_common(1)[0]
        reliability = (ip[1] / len(host_df)) * 100

        if (reliability >= 70) and (ip[0] not in host_dict):
            host_dict[ip[0]]=hostname
        else:
            unreliable_hosts.append([hostname, reliability])
    with open(path + 'ip_dict.txt','w') as data:
            data.write(str(host_dict).replace(",", ",\n"))
    return host_dict

def _ip_to_hostname(dataframe, host_dict, mode):
    for idx,event in dataframe.iterrows():
        if event['source_ip'] in host_dict:
            dataframe.at[idx,'source_hostname'] = host_dict[event['source_ip']]
        if event['hostname'] in host_dict: # For events: 1024 y 1102
            dataframe.at[idx,'hostname'] = host_dict[event['hostname']]
    dataframe = dataframe[~dataframe['source_hostname'].isin(['local','-','nan','none','d4_null'])]
    dataframe.set_index('time', inplace=True)
    return dataframe

def _codify_dataset(dataset):
    users = dataset["remote_user"].unique()
    users = list(users)
    users_ = ['user' + str(i).zfill(6) for i in range(len(users))]
    users_dict = dict(zip(users, users_))

    hosts = pd.concat([dataset['hostname'], dataset['source_hostname']]).unique()
    hosts = list(hosts)
    hosts_ = ['host' + str(i).zfill(6) for i in range(len(hosts))]
    hosts_dict = dict(zip(hosts, hosts_))

    dataset1 = dataset.replace({"remote_user": users_dict})
    dataset2 = dataset1.replace({"hostname": hosts_dict})
    dataset3 = dataset2.replace({"source_hostname": hosts_dict})
    return dataset3, users_dict, hosts_dict

def _build_lm_dataset(dataframe):
    dataframe['timestamp'] = dataframe.index.strftime('%H:%M:%S')
    path_df,adjacency_df = [],[]

    for idx, df_day in dataframe.groupby(dataframe.index.date):
        user_list = df_day["remote_user"].value_counts()
        for user in user_list.index:
            df_day_user = df_day.query("remote_user==@user")
            adj_dict = _build_adjancency(df_day_user, 'source_hostname', 'hostname')
            adjacency_df.append([idx, user, adj_dict])
    path_df = _build_path_df(adjacency_df)
    columns = ['date', 'user', 'path']
    lm_dset = pd.DataFrame(path_df, columns = columns)
    lm_dset = _del_local(lm_dset)

    lm_dset,neo4j_dset = _build_neo_dataset(dataframe, lm_dset)
    return lm_dset,neo4j_dset

def _get_timestamp(date, user, paths, dset):
    timestamps = []
    dset_ = dset[dset.index.date == date]
    for path in paths:
        result = dset_[(dset_['remote_user'] == user) & (dset_['source_hostname'] == path[0]) & (dset_['hostname'] == path[1])].timestamp
        timestamps.append(result.iloc[0])
    return timestamps

def _build_neo_dataset(dataframe, lm_dset):
    df_neo4j,timestamps = [],[]
    columns = ['date', 'remote_user', 'source_hostname', 'hostname']

    for idx,path in enumerate(lm_dset['path']):
        pairs = _sliding_window(path, 2)
        timestamp = _get_timestamp(lm_dset['date'].iloc[idx], lm_dset['user'].iloc[idx], pairs, dataframe)
        timestamps.append(timestamp)
        for idx2,node in enumerate(pairs):
            df_neo4j.append([str(lm_dset['date'].iloc[idx]) + ' ' + timestamp[idx2], \
                             lm_dset['user'].iloc[idx], \
                             node[0],
                             node[1]])
    lm_dset['timestamp'] = timestamps
    neo4j_dset = pd.DataFrame(df_neo4j, columns = columns)
    neo4j_dset = neo4j_dset.drop_duplicates()
    return lm_dset,neo4j_dset

def _print_top_anomalies(top_n, error_matrix, ml_dset):
    cnt = 1
    err,dat,usr,mov,tim = [],[],[],[],[]
    print(" ")
    print("__________________________________________________________________________")
    print("TOP-"+ str(top_n)+" Anomalies")
    print("__________________________________________________________________________")
    for anomalies in error_matrix:
        if cnt > top_n:
            break
        else:
            f1 = ml_dset['user'].iloc[int(anomalies[0])]
            usr.append(f1)
            print(str(cnt), ") User: " + f1)

            print("--------------+-----------------------------------------------------------")

            f2 = f'{(1-anomalies[1]):.0%}'
            err.append(f2)
            print("Timeline      | Lateral Movement (Error=", f2 + ')')

            print("--------------+-----------------------------------------------------------")

            f3 = ml_dset['date'].iloc[int(anomalies[0])].date()
            f4 = ml_dset['path'].iloc[int(anomalies[0])]
            f5 = ml_dset['timestamp'].iloc[int(anomalies[0])]
            dat.append(f3)
            mov.append(f4)
            tim.append(f5)
            print(str(f3) + "    |", f4[0])
            empty = len(f4[0]) + 2
            for idx,path in enumerate(f4[1:]):
                print(" |-> " + str(f5[idx]) + " |" + " "*empty + path)
                empty += len(path) + 1
            print("__________________________________________________________________________")
            cnt += 1
            
    out = pd.DataFrame()
    out['date'] = dat
    out['user'] = usr
    out['path'] = mov
    out['timestamp'] = tim
    out['error'] = err
    return out

def _sliding_window(elements, window_size):
    windows = []
    if len(elements) <= window_size:
        windows.append(elements)
    else:
        for i in range(len(elements)- window_size + 1):
            windows.append(elements[i:i+window_size])
    return windows     

def _safe_anomalies_neo4j(top_n, error_matrix, ml_dataset, path=''):
    users = []
    df_user = []
    df_user_full = []
    columns = ['date', 'remote_user', 'source_hostname', 'hostname']

    for i_anomalies in error_matrix[0:top_n]:
        i_user = ml_dataset['user'].iloc[int(i_anomalies[0])]
        users.append(i_user)
    users = set(users)

    for user in users: 
        
        for anomalies in error_matrix[0:top_n]:
            if user == ml_dataset['user'].iloc[int(anomalies[0])]:
                paths = _sliding_window(ml_dataset['path'].iloc[int(anomalies[0])], 2)
                for nodes in paths:
                    df_user.append([ml_dataset['date'].iloc[int(anomalies[0])], \
                                    user, \
                                    nodes[0],
                                    nodes[1]])
        df_user = pd.DataFrame(df_user, columns = columns)
        df_user = df_user.drop_duplicates()
        df_user.to_csv(path + str(user) + '.csv', index=False)
        df_user = []


        user_full = ml_dataset.query("user==@user")
        for idx, upath in enumerate(user_full['path']):  # Safe full user activity
            pairs = _sliding_window(upath, 2)
            for node in pairs:
                if len(node) < 2:
                    df_user_full.append([user_full['date'].iloc[idx], \
                                     user_full['user'].iloc[idx], \
                                     node[0],
                                     node[0]])
                else:
                    df_user_full.append([user_full['date'].iloc[idx], \
                                     user_full['user'].iloc[idx], \
                                     node[0],
                                     node[1]])
        df_user_full = pd.DataFrame(df_user_full, columns = columns)
        df_user_full = df_user_full.drop_duplicates()
        df_user_full.to_csv(path + str(user) + '_full.csv', index=False)
        df_user_full = []