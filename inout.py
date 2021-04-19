from numbers import Number
from typing import Union
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sparsegraph import SparseGraph


def networkx_to_sparsegraph_floatfeature(
        nx_graph: Union['nx.Graph'],
        feature_dim = int,
        sparse_node_attrs: bool = True,
        ) -> 'SparseGraph':

    #float feature:pubmed, wiki

    # Extract node names
    node_names = None
    attr_names = None

    # Extract adjacency matrix
    adj = nx.adjacency_matrix(nx_graph)
    adj_matrix = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), feature_dim), dtype=np.float32)
    labels = [0 for _ in range(nx_graph.number_of_nodes())]

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == 'label': 
                labels[inode] = val
            else:
                attr_matrix[inode] = val
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    labels = np.array(labels, dtype=np.int32)
    class_names = None

    return SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, class_names=class_names,
            metadata=None)

def networkx_to_sparsegraph_intfeature(
        nx_graph: Union['nx.Graph'],
        feature_dim = int,
        sparse_node_attrs: bool = True,
        ) -> 'SparseGraph':

    # int feature : cora, citeseer

    # Extract node names
    node_names = None
    attr_names = None

    # Extract adjacency matrix
    adj = nx.adjacency_matrix(nx_graph)
    adj_matrix = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), feature_dim), dtype=np.float32)
    labels = [0 for _ in range(nx_graph.number_of_nodes())]

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == 'label':
                labels[inode] = val
            else:
                # int feature
                index = np.nonzero(val)[1]
                for i in index:
                    attr_matrix[inode, i] = 1

    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    labels = np.array(labels, dtype=np.int32)
    class_names = None

    return SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, class_names=class_names,
            metadata=None)


def networkx_to_sparsegraph_acm(
        nx_graph: Union['nx.Graph'],
        feature_dim = int,
        sparse_node_attrs: bool = True,
        ) -> 'SparseGraph':

    import networkx as nx

    # Extract node names
    node_names = None
    attr_names = None
    # Extract adjacency matrix
    adj = nx.adjacency_matrix(nx_graph)
    adj_matrix = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), feature_dim), dtype=np.float32)
    labels = [0 for _ in range(nx_graph.number_of_nodes())]

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == 'label':
                labels[inode] = val
            else:
                index = np.nonzero(val)
                for i in index:
                    attr_matrix[inode, i] = 1
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    labels = np.array(labels, dtype=np.int32)
    class_names = None

    return SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, class_names=class_names,
            metadata=None)

