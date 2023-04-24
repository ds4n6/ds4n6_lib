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
import pandas as pd
import networkx as nx
from collections import Counter

#############################################################################################
# FUNCTIONS
#############################################################################################
def build_lm_dataset(dset, mode='hostname', path=''): # mode: [hostname, ip_addr]
    """ Function to build a lateral movement (LM) dataset
    Syntax: build_lm_dataset(dset="<dset>", mode="<mode>", path="<path>")
    Args:
        dset (pandas.core.frame.DataFrame): path of the csv file to read. Min. columns: ['time','event_id','source_name','host_ip','source_hostname','logon_type','remote_user']
        mode (str): build mode. 'hostname' to create LM dataset only by using known hostnames. 'ip_addr' to use IP addres for unknown hostnames
        path (str): path to store lateral movement datasets
            - ds4n6_lm_dataset.csv: dataset with lateral movements
            - ds4n6_neo4j_dataset.csv: dataset with lateral movements to be loaded in Neo4j
    Returns:
        lm_dset (core.frame.DataFrame): dataset with lateral movements
    """
    dset    = _clear_dataset(dset)
    dict_ip = _create_ip_dict(dset)
    dset    = _ip_to_hostname(dset, dict_ip, mode=mode)
    cdset   = _codify_dataset(dset)  
    lm_dset = _build_lm_dataset(cdset[0])
    lm_neo  = _build_neo_dataset(lm_dset)
    
    lm_dset.to_csv(path + 'ds4n6_lm_dataset.csv', index=False)
    lm_neo.to_csv(path + 'ds4n6_neo4j_dataset.csv', index=False)
    
    with open('dictionary.txt','w') as data: 
        data.write(str(cdset[1]))
        data.write(str(cdset[2]))
    return lm_dset

def find_lm_anomalies(lm_dset, model, from_date, to_date, top_n=50, neo4j=True, path=''):
    """ Function to detect anomalous lateral movement with machine learning
    Syntax: find_lm_anomalies(lm_dset="<lm_dset>", model="<model>", from_date="<from_date>", to_date="<to_date>", top_n="<top_n>", neo4j="<neo4j>", path="<path>")
    Args:
        lm_dset (pandas.core.frame.DataFrame): path of the csv file to read
        model (str): ML algorithm to be used. Supported models: ('s2s_lstm', 'transformer')
        from_date (str): init date of training dataset
        to_date (str): end date of training dataset
        top_n (str): number of anomalous lateral movement to be detected
        neo4j (bool): 'True' to export the output to Neo4j format. 'False' otherwise
        path (str): path to store Neo4j output datasets
            - <user>.csv: dataset with anomalous lateral movements by user
            - <user>_full.csv: dataset with all user activity in the input dataset
    """
    if model == "transformer":
        from ds4n6_lib.ml_models.transformer import Seq2seqData, Autoencoder
    elif model == "s2s_lstm":
        from ds4n6_lib.ml_models.seq2seq_lstm import Seq2seqData, Autoencoder
    else:
        raise ValueError("Error: model '" + model + "' not supported.")
        
    data = Seq2seqData()
    ml_dset = data.load_path_dataset(lm_dset, from_date, to_date, min_count=16)
    train_x, train_y = data.process_train_data(ml_dset)
    data.build_train_dset(train_x, train_y)
    
    model = Autoencoder(embed_dim=16, latent_dim=300, data=data)
    model.build_autoencoder()
    model.fit_autoencoder()
    err_mtrx = model.get_anomalies(train_x)
    _print_top_anomalies(top_n, err_mtrx, ml_dset)
    if neo4j:
        _safe_anomalies_neo4j(top_n, err_mtrx, ml_dset, path)

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

def _clear_dataset(dataframe):
    df=dataframe.astype(str)
    df['time'] = df['time'].str[0:19]
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', utc=True)
    
    df.drop(df[df['event_id'] != '4624'].index, inplace = True)
    df = df[~df['remote_user'].str.contains('$', regex = False)]
    df = df[~df['remote_user'].str.contains('anonymous', regex = False)]
    df['hostname'] = df['hostname'].str.split('.', 1, expand = True)
    
    target_columns = ['hostname', 'source_hostname', 'remote_user']
    for column in target_columns:
        df[column] = df[column].str.lower()
    df.set_index('time', inplace=True)
    return df

def _create_ip_dict(dataframe):
    dict_df = dataframe[dataframe['logon_type'].str.contains('3.0', regex = False)]
    dict_df = dict_df[~dict_df['source_hostname'].isin(['-','nan','none'])]
    host_list=dict_df["source_hostname"].value_counts()

    host_dict = {}
    unreliable_hosts = []

    for hostname in host_list.index:
        host_df = dict_df[dict_df['source_hostname'].isin([hostname])]['source_ip']
        c = Counter(host_df)
        ip = c.most_common(1)[0]
        reliability = (ip[1] / len(host_df)) * 100

        if reliability >= 70:
            host_dict[ip[0]]=hostname
        else:
            unreliable_hosts.append([hostname, reliability]) 
    return host_dict

def _ip_to_hostname(dataframe, host_dict, mode):
    for idx,event in dataframe.iterrows():
        if event['source_ip'] in host_dict:
            event['source_hostname'] = host_dict[event['source_ip']]
        elif mode == 'hostname':
            event['source_hostname'] = 'None'
        elif mode == 'ip_addr':
            event['source_hostname'] = event['source_ip']
        else:
            raise ValueError("Error: not supported mode. Try 'hostname' or 'ip_addr' modes.")
    dataframe = dataframe[~dataframe['source_hostname'].str.contains('None', regex=False)]
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
    path_df      = []
    adjacency_df = []
    for idx, df_day in dataframe.groupby(dataframe.index.date):
        user_list = df_day["remote_user"].value_counts()
        for user in user_list.index:
            df_day_user = df_day.query("remote_user==@user")
            adj_dict = _build_adjancency(df_day_user, 'source_hostname', 'hostname')
            adjacency_df.append([idx, user, adj_dict])

    path_df = _build_path_df(adjacency_df)
    columns = ['time', 'user', 'path']
    path_df = pd.DataFrame(path_df, columns = columns)
    return path_df

def _build_neo_dataset(dataframe):
    df_neo4j = []
    columns = ['time', 'remote_user', 'source_hostname', 'hostname']

    for idx,path in enumerate(dataframe['path']):
        pairs = _sliding_window(path, 2)
        for node in pairs:
            if len(node) < 2:
                df_neo4j.append([dataframe['time'].iloc[idx], \
                                 dataframe['user'].iloc[idx], \
                                 node[0],
                                 node[0]])
            else:
                df_neo4j.append([dataframe['time'].iloc[idx], \
                                 dataframe['user'].iloc[idx], \
                                 node[0],
                                 node[1]])
    df_neo4j = pd.DataFrame(df_neo4j, columns = columns)
    df_neo4j = df_neo4j.drop_duplicates()
    return df_neo4j

def _print_top_anomalies(top_n, error_matrix, ml_dset):
    cnt = 1  
    print(" ")
    print("__________________________________________________________________________")
    print("TOP-"+ str(top_n)+" Anomalies")
    print("==========================================================================")
    for anomalies in error_matrix:
        if cnt > top_n:
            break
        else:
            print(str(cnt), ") Error="+str(1-anomalies[1]))
            print("Date:", ml_dset['time'].iloc[int(anomalies[0])])
            print("User:", ml_dset['user'].iloc[int(anomalies[0])])
            print("Lateral Movement:", ml_dset['path'].iloc[int(anomalies[0])])
            print("==========================================================================")
            cnt += 1

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
    columns = ['time', 'remote_user', 'source_hostname', 'hostname']

    for i_anomalies in error_matrix[0:top_n]:
        i_user = ml_dataset['user'].iloc[int(i_anomalies[0])]
        users.append(i_user)
    users = set(users)

    for user in users: 
        
        for anomalies in error_matrix[0:top_n]:
            if user == ml_dataset['user'].iloc[int(anomalies[0])]:
                paths = _sliding_window(ml_dataset['path'].iloc[int(anomalies[0])], 2)
                for nodes in paths:
                    df_user.append([ml_dataset['time'].iloc[int(anomalies[0])], \
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
                    df_user_full.append([user_full['time'].iloc[idx], \
                                     user_full['user'].iloc[idx], \
                                     node[0],
                                     node[0]])
                else:
                    df_user_full.append([user_full['time'].iloc[idx], \
                                     user_full['user'].iloc[idx], \
                                     node[0],
                                     node[1]])
        df_user_full = pd.DataFrame(df_user_full, columns = columns)
        df_user_full = df_user_full.drop_duplicates()
        df_user_full.to_csv(path + str(user) + '_full.csv', index=False)
        df_user_full = []