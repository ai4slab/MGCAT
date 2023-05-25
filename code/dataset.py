import pickle

import numpy as np
import pandas as pd
import torch
import torch_geometric.utils as pyg_utils


class Dataset(object):
    def __init__(self):
        with open('../data/lncRNA_feature.pkl', 'rb') as f:
            self.lncRNA_feature = pickle.load(f)
        with open('../data/miRNA_feature.pkl', 'rb') as f:
            self.miRNA_feature = pickle.load(f)
        lncRNA_names = pd.read_csv(f'../data/lncRNA_idx.csv', header=None, sep='\t')[1].values.tolist()
        miRNA_names = pd.read_csv(f'../data/miRNA_idx.csv', header=None, sep='\t')[1].values.tolist()
        self.num_lncRNAs = len(lncRNA_names)
        self.num_miRNAs = len(miRNA_names)
        self.num_nodes = self.num_lncRNAs + self.num_miRNAs

        with open('../data/splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.train_pos = splits['train_pos_1']
        self.train_neg = splits['train_neg_1']
        self.test_pos = splits['test_pos_1']
        self.test_neg = splits['test_neg_1']
        edge_index = torch.tensor(self.train_pos.T, dtype=torch.long)
        self.edge_index = pyg_utils.to_undirected(edge_index)

    def fetch_train_data(self):
        interactions = np.concatenate([self.train_pos, self.train_neg])
        labels = np.concatenate([np.ones(len(self.train_pos)), np.zeros(len(self.train_neg))])
        return interactions, labels

    def fetch_test_data(self):
        interactions = np.concatenate([self.test_pos, self.test_neg])
        labels = np.concatenate([np.ones(len(self.test_pos)), np.zeros(len(self.test_neg))])
        return interactions, labels
