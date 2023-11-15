import numpy as np
import pandas as pd
import scipy
import networkx as nx
import os

def get_node_features(file_name, path_name):
    """Gets centrality measures for all nodes in graph

    :param model_id: PATRIC ID # for metabolic model
    :param path_name: Path to stoichiometric matrices
    :return: None; saves csv of node features
    """
    #load stoichiometric matrix and convert to metabolite graph
    data = scipy.sparse.load_npz(path_name + '/' + str(file_name))
    s = data
    s_t = scipy.sparse.csr_matrix.transpose(s)
    metabolite_graph = s.dot(s_t)
    metabolite_graph = nx.from_scipy_sparse_matrix(metabolite_graph)
    number_of_nodes = metabolite_graph.number_of_nodes()
    
    #Calculate centralities for all nodes and append them to master list
    features = []
    between_centrality = nx.betweenness_centrality(metabolite_graph, k=1024).values()
    features.append(list(between_centrality))
    close_centrality = nx.closeness_centrality(metabolite_graph).values()
    features.append(list(close_centrality))
    degree_centrality = nx.degree_centrality(metabolite_graph).values()
    features.append(list(degree_centrality))
    eig_centrality = nx.eigenvector_centrality(metabolite_graph).values()
    features.append(list(eig_centrality))
    
    #Save feature array as csv file
    df_features = pd.DataFrame(features).to_numpy()
    np.savetxt(os.getcwd() + "/feature_files/" + str(file_name) + '.csv', df_features, delimiter = ',')

def get_all_features(i, all_files, path_name):
    """Allows for easy iteration though a list of models

    :param i: Index of model within list of files
    :param all_files: List of all model ID #s
    :return: None
    """
    #iterate through list of files; parallelizable
    get_node_features(all_files[i], path_name)