import numpy as np
import torch 
import pandas as pd
import scipy
import os
import os.path as osp
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join
from torch_geometric.data import Dataset, download_url, Data

class EColiDataset(Dataset):
    def __init__(self, root, data_path, feature_path, metadata, pre_transform, transform=None, pre_filter=None):
        #change type of pre_transform based on desired graph representation. Current options: RAG, metabolite,... 
        #...metabolite_reduced, common_metabolite_reduced
        self.pre_transform = pre_transform
        self.metadata = metadata
        self.path_name = root
        self.data_path = data_path
        self.feature_path = feature_path
        antibiotic_list = list(self.metadata.columns[8:20])
        print("Antibiotic options: ", antibiotic_list)
        antibiotic_index = input("Enter index of desired antibiotic: ")
        self.antibiotic_name = antibiotic_list[int(antibiotic_index)]
        print("Confirmed antibiotic selection: ", self.antibiotic_name)
        super().__init__(root, transform, pre_filter)

    @property
    def raw_file_names(self):
        ids = self.metadata['Genome ID'][1::]
        all_files = []
        for i_d in ids:
            file_name = i_d + '_s.npz'
            all_files.append(file_name)
        return all_files

    @property
    def processed_file_names(self):
        ids = self.metadata['Genome ID'][1::]
        all_files = []
        all_files_processed = []
        for i_d in ids:
            file_name = i_d + '_s.npz'
            all_files.append(file_name)
        for file in all_files:
            new_name = "processed" + file
            all_files_processed.append(new_name)
        return all_files_processed
    
    def process(self):
        path_name = self.data_path
        idx = 0
        ids = self.metadata["Genome ID"][1::]
        labels = self.metadata[self.antibiotic_name][1::]
        all_files = []
        for i_d in ids:
            file_name = i_d + '_s.npz'
            all_files.append(file_name)
        #convert antibiotic resistance profiles to graph feature tensor
        gl = []
        for strain in labels:
            if strain == 'Resistant':
                gl.append(1)
            elif strain == 'Susceptible':
                gl.append(0)
            else:
                gl.append(2) 
            
        graph_labels = torch.tensor(gl)

        #save graphs as Data files if antibiotic resistance profile is known for selected antibiotic
        for raw_file in all_files:
            #print(idx, "AR: ", str(gl[all_files.index(raw_file)]))
            if gl[all_files.index(raw_file)] == 2:
                pass
            else:
                # Read data from `raw_path`.
                data = scipy.sparse.load_npz(str(path_name) + '/' + str(raw_file))
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    adj = self.pre_transform(data)
            
                #convert connectivity to coo for edge_index
                data = scipy.sparse.csr_matrix.tocoo(data)
                coo = data
                row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
                col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
                edge_index = torch.stack([row, col], dim=0)
            
                #set node features
                num_nodes = data.shape[0]
                feature_file = raw_file + ".csv"
                features = pd.read_csv(self.feature_path + '/' + feature_file, header=None).to_numpy()
                x=torch.from_numpy((features.T))
                
                data = Data(x=x, edge_index=edge_index, y=torch.tensor(graph_labels[idx]))
                data.num_nodes = num_nodes
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx = idx +1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        all_files = os.listdir(str(self.path_name))
        data = torch.load(osp.join(self.path_name, all_files[idx]))
        return data
    
    #Transforms the data into a unipartite, unweighted graph
    #Nodes represent metabolites
    #Edges represent the reactions that link metabolites
    def metabolite(self,data): 
        s = data
        s_t = scipy.sparse.csr_matrix.transpose(s)
        data = s.dot(s_t)
        return data
    
    #Transforms the data into a unipartite, unweighted graph
    #Nodes represent metabolites
    #Edges represent the reactions that link metabolites
    #Network has been topologically reduced through removal...
    #...of the 20 nodes of highest degree
    def metabolite_reduced(self, data):
        s = data
        s_t = scipy.sparse.csr_matrix.transpose(s)
        data = s.dot(s_t)
        data = sklearn.preprocessing.binarize(data)
        adj = data.todense()
        g = nx.from_numpy_array(adj)
        g_sort = sorted(g.degree, key=lambda x: x[1], reverse=True)
        for n in range(0,20): 
            g.remove_node(g_sort[n][0])
        data = nx.to_scipy_sparse_array(g)
        return data
    
    #Transforms the data into a unipartite, unweighted graph
    #Nodes represent metabolites
    #Edges represent the reactions that link metabolites
    #Only the 627 metabolites which are common to all strains...
    #...in the dataset are retained
    def common_metabolite(self, data, compound_path):
        self.compound_path = compound_path
        strain_number = raw_file.split('_')[3]
        strain_number = strain_number.strip('.npz')
        compounds = pd.read_csv(compound_path + "Compounds/" + strain_number + '_compounds.csv', header=None).to_numpy()
        common_compounds = pd.read_csv(compound_path + "common_compounds.csv", header=None).to_numpy()
        s = data
        s_t = scipy.sparse.csr_matrix.transpose(s)
        adj = s.dot(s_t)
        adj = adj.todense()
        adj = np.asarray(adj)
        drop_compound_i = []
        for i in range(len(compounds)):
            if compounds[i] not in common_compounds:
                drop_compound_i.append(i)
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i,j] != 0:
                    adj[i,j] = 1
        node_labels = dict(enumerate(compounds.flatten(), 1))
        g_prelabel = nx.from_numpy_matrix(adj)
        g_prelabel.remove_nodes_from(drop_compound_i)
        g_presort = nx.relabel_nodes(g_prelabel, node_labels)
        g = g_presort
        data = nx.to_scipy_sparse_array(g, nodelist=mapping)
        return data
    
    #Same as above, but with topological reduction
    def common_metabolite_reduced(self, data):
        s = data
        s_t = scipy.sparse.csr_matrix.transpose(s)
        data = s.dot(s_t)
        adj = data
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i,j] != 0:
                    adj[i,j] = 1
        g = nx.from_numpy_array(adj)
        data = nx.to_scipy_sparse_array(g, nodelist=common_reduced_compounds)
        return data
    
    #Transforms the data into a unipartite, unweighted graph
    #Nodes represent reaction
    #Edges represent the metabolites involved in linked reactions
    def RAG(self, data):
        s = data
        s_t = scipy.sparse.csr_matrix.transpose(s)
        data = s_t.dot(s)
        #print(data.shape)
        return data
    