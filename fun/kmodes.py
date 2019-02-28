import numpy as np
import pandas as pd

def pickup_centroids(df,k):
    centroids_idx = np.random.choice(a = df.index.values, replace = False, size = k)
    centroid_a = df.loc[centroids_idx[0]].values
    centroid_b = df.loc[centroids_idx[1]].values
    return(centroid_a,centroid_b)

def distance_mismatch(a,b):
    return (a != b).sum()


def compute_distances_to_centroids(centroid_a,centroid_b,df):
    
    dic_distances = {}
    for idx,row in df.iterrows():
        candidat = row.values
        distance_to_a = distance_mismatch(candidat,centroid_a)
        distance_to_b = distance_mismatch(candidat,centroid_b)
        affectation = np.argmin([distance_to_a,distance_to_b])
        dic_distances[idx] = {"distance_to_a" : distance_to_a,
                              "distance_to_b" : distance_to_b,
                              "affectation" : affectation}
        
    return dic_distances


def extract_assigned_data(dic_distances,df):
    alist = []
    blist = []
    for k,v in dic_distances.items():
        if v["affectation"] == 0:
            alist.append(df.loc[k].values)
        else:
            blist.append(df.loc[k].values)
    
    if len(alist)>0:
        a = np.vstack(alist)
    else:
        a = np.array([])
        
    if len(blist)>0:
        b = np.vstack(blist)
    else:
        b = np.array([])
        
    return a,b


def compute_mode(array):
    return ((np.sum(array,axis = 0) - array.shape[0]/2) > 0).astype(int)


def compute_performance(dic_distances):
    
    distances_list = []
    
    for k,v in dic_distances.items():
        if v["affectation"] == 0:
            distances_list.append(v["distance_to_a"])
        else:
            distances_list.append(v["distance_to_b"])
    return (np.array(distances_list)).sum()



def get_max_id(diccy,thekey):

    best_p = np.inf
    
    for k,v in diccy.items():
        p = v[thekey]
        if p < best_p:
            best_p = p
            best_idx = k
    
    return best_idx


def kmodes(df,k=2,threshold=1,iterations=10, verbose = True, n_clusterings = 5):
    
    dic_results = {}
    
    for id_clustering in range(n_clusterings):
        print("Clustering n°",id_clustering + 1,"/",n_clusterings)
    
        # Init centroids
        centroid_a,centroid_b = pickup_centroids(df,k)

        for i in range(iterations):
            if verbose:
                print("iteration : ",i)
                print(df.columns.values[centroid_a.astype(bool)])
                print(df.columns.values[centroid_b.astype(bool)])
            dic_distances = compute_distances_to_centroids(centroid_a,centroid_b,df)
            array_a,array_b = extract_assigned_data(dic_distances,df)

            mycdt = len(array_a)==0 or len(array_b)==0
            if verbose:
                print(mycdt)
            if mycdt:
                continue

            futur_centroid_a = compute_mode(array_a)
            futur_centroid_b = compute_mode(array_b)
            d = distance_mismatch(futur_centroid_a,centroid_a) + distance_mismatch(futur_centroid_b,centroid_b)
            if verbose:
                print("distance parcourue : ", d)
                print("")

            if d<threshold:
                break
            centroid_a = futur_centroid_a
            centroid_b = futur_centroid_b
            
        performance = compute_performance(dic_distances)
        dic_results[id_clustering] = {"performance" : performance,
                                      "dic_distances" : dic_distances,
                                      "array_a" : array_a,
                                      "array_b" : array_b}
    
    idx_max = get_max_id(dic_results,"performance")
    dic_distances = dic_results[idx_max]["dic_distances"]
    array_a = dic_results[idx_max]["array_a"]
    array_b = dic_results[idx_max]["array_b"]
    metric = dic_results[idx_max]["performance"]
    
    clustering_df = pd.DataFrame.from_dict(dic_distances,orient="index")
    clustering_df.drop(['distance_to_a',"distance_to_b"], axis=1, inplace = True) 
    stats_value_cluster_a = np.sum(array_a,axis = 0) / array_a.shape[0]
    cluster_a_caracterisation = stats_value_cluster_a / (df.sum(axis = 0) / len(df))
    stats_value_cluster_b = np.sum(array_b,axis = 0) / array_b.shape[0]
    cluster_b_caracterisation = stats_value_cluster_b / (df.sum(axis = 0) / len(df))
    clusters_caracterisation = pd.concat([cluster_a_caracterisation,cluster_b_caracterisation],axis = 1)
    
    return clustering_df,clusters_caracterisation,metric
    
