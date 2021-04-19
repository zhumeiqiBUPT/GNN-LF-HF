# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:48:06 2020

@author: LENOVO
"""

import numpy as np
import sys
from inout import *
import os
import scipy.sparse as sp
import sys
import pickle as pkl
import numpy as np
import json
import itertools
import networkx as nx
import os.path
from sparsegraph import SparseGraph


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def train_test_split(graph_labels_dict, labelrate):

    idx_train = []
    idx_test = []
    idx_val = []
    val_count = 0

    n = len(graph_labels_dict)
    class_num = max(graph_labels_dict.values()) + 1
    train_num = class_num * labelrate

    idx = list(range(n))

    count = [0] * class_num
    for i in range(len(idx)):
        l = graph_labels_dict[idx[i]]
        if count[l] < labelrate:
            idx_train.append(idx[i])
            count[l] = count[l] + 1
        elif len(idx_train) == train_num and val_count < 500:
            idx_val.append(idx[i])
            val_count = val_count + 1
    for i in range(len(idx)-1000, len(idx)):
        idx_test.append(idx[i])
    idx_np = {}
    idx_np['train'] = idx_train
    idx_np['stopping'] = idx_val
    idx_np['valtest'] = idx_test

    return idx_np


def train_test_split_acm(graph_labels_dict, labelrate):

    idx_train = []
    idx_test = []
    idx_val = []
    val_count = 0

    n = len(graph_labels_dict)
    class_num = max(graph_labels_dict.values()) + 1
    train_num = class_num * labelrate

    idx = list(range(n))

    #random
    np.random.seed(20)
    np.random.shuffle(idx)
    count = [0] * class_num
    for i in range(len(idx)):
        l = graph_labels_dict[idx[i]]
        if count[l] < labelrate:
            idx_train.append(idx[i])
            count[l] = count[l] + 1
        elif len(idx_train) == train_num and val_count < 500:
            idx_val.append(idx[i])
            val_count = val_count + 1
    for i in range(len(idx)-1000, len(idx)):
        idx_test.append(idx[i])
    idx_np = {}
    idx_np['train'] = idx_train
    idx_np['stopping'] = idx_val
    idx_np['valtest'] = idx_test

    return idx_np


def load_new_data_wiki(labelrate):
    data = json.load(open('./data/wiki/data.json'))

    features = np.array(data['features'])
    labels = np.array(data['labels'])

    n_feats = features.shape[1]

    graph_node_features_dict = {}
    graph_labels_dict = {}
    for index in range(len(features)):
        graph_node_features_dict[index] = features[index]
        graph_labels_dict[index] = int(labels[index])

    g = nx.DiGraph()

    for index in range(len(features)):
        g.add_node(index, features=graph_node_features_dict[index],
                   label=graph_labels_dict[index])
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(data['links'])]))

    for edge in edge_list:
        g.add_edge(int(edge[0]), int(edge[1]))

    sG = networkx_to_sparsegraph_floatfeature(g, n_feats)

    idx_np = train_test_split(graph_labels_dict, labelrate)

    return sG, idx_np


def load_new_data_acm(labelrate):
    graph_adjacency_list_file_path = os.path.join('./data/acm/acm_PAP.edge')
    graph_node_features_file_path = os.path.join('./data/acm/acm.feature')
    graph_labels_file_path = os.path.join('./data/acm/acm.label')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    index = 0
    with open(graph_node_features_file_path) as graph_node_features_file:
        for line in graph_node_features_file:
            assert (index not in graph_node_features_dict)
            graph_node_features_dict[index] = np.array(line.strip('\n').split(' '), dtype=np.uint8)
            index = index + 1
    index = 0
    with open(graph_labels_file_path) as graph_labels_file:
        for line in graph_labels_file:
            assert (index not in graph_labels_dict)
            graph_labels_dict[index] = int(line.strip('\n'))
            G.add_node(index , features=graph_node_features_dict[index], label=graph_labels_dict[index])
            index = index + 1

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        for line in graph_adjacency_list_file:
            line = line.rstrip().split(' ')
            assert (len(line) == 2)
            G.add_edge(int(line[0]), int(line[1]))

    sG = networkx_to_sparsegraph_acm(G, 1870)

    
    idx_np = train_test_split_acm(graph_labels_dict, labelrate)

    return sG, idx_np


def load_data_tkipf(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/tkipf_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/tkipf_data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_test, idx_train, idx_val


def load_new_data_tkipf(dataset_name, feature_dim, labelrate):
    adj, features, labels, idx_test, idx_train, idx_val = load_data_tkipf(dataset_name)
    labels = np.argmax(labels, axis=-1)
    features = features.todense()
    G = nx.DiGraph(adj)

    for index in range(len(labels)):
        G.add_node(index , features=features[index], label=labels[index])
    if dataset_name == 'pubmed':
        sG = networkx_to_sparsegraph_floatfeature(G, feature_dim)
    else:
        sG = networkx_to_sparsegraph_intfeature(G, feature_dim)

    graph_labels_dict = {}
    for index in range(len(labels)):
        graph_labels_dict[index] = int(labels[index])

    idx_np = {}
    if labelrate == 20:
        idx_np['train'] = idx_train
        idx_np['stopping'] = idx_val
        idx_np['valtest'] = idx_test
    else:
        idx_np = train_test_split(graph_labels_dict, labelrate)

    return sG, idx_np


def load_new_data_ms(labelrate):
    with np.load('./data/ms/ms_academic.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        dataset = SparseGraph.from_flat_dict(loader)
        graph_labels_dict = {}
        for index in range(len(dataset.labels)):
            graph_labels_dict[index] = int(dataset.labels[index])
        idx_np = train_test_split(graph_labels_dict, labelrate)

        return dataset, idx_np
