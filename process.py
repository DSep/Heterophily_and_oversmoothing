import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from sklearn.model_selection import ShuffleSplit
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor, row_normalized_adjacency, dense_tensor_to_sparse_mx
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
import torch_geometric

from virtual_nodes.augment import augment_graph, _binarize_tensor

# adapted from geom-gcn


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

################ from GCNII/Geom-GCN (with bugs fixed) ###############################################################################################################


def full_load_citation(dataset_str):
    """
    (J+S)
    We don't care about this – this is for citation datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    if len(test_idx_range_full) != len(test_idx_range):
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position, mark them
        # Follow H2GCN code
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
    else:
        non_valid_samples = set()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    ############### Follow H2GCN and fix the bug ##############
    non_valid_samples = non_valid_samples.union(
        set(list(np.where(labels.sum(1) == 0)[0])))
    return adj, features, labels, non_valid_samples


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) # sum over columns, (N x 1)
    rowsum = (rowsum == 0)*1+rowsum # if rowsum is 0, then set it to 1
    r_inv = np.power(rowsum, -1).flatten() # inverse of rowsum, (N x 1)
    r_inv[np.isinf(r_inv)] = 0. 
    r_mat_inv = sp.diags(r_inv) # diagonal matrix with r_inv as diagonal elements
    features = r_mat_inv.dot(features) 
    return features # sum_j F_ij

####### codes from the original GeomGCN git repo #################


def process_geom(G, dataset_name, embedding_method):
    embedding_file_path = os.path.join(
        'structural_neighborhood', "outf_nodes_space_relation_{}_{}.txt".format(dataset_name, embedding_method))
    space_and_relation_type_to_idx_dict = {}

    with open(embedding_file_path) as embedding_file:
        for line in embedding_file:
            if line.rstrip() == 'node1,node2	space	relation_type':
                continue
            line = re.split(r'[\t,]', line.rstrip())
            assert (len(line) == 4)
            assert (int(line[0]) in G and int(line[1]) in G)
            if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                    space_and_relation_type_to_idx_dict)
            if G.has_edge(int(line[0]), int(line[1])):
                G.remove_edge(int(line[0]), int(line[1]))
            G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                (line[2], int(line[3]))])

    space_and_relation_type_to_idx_dict['self_loop'] = len(
        space_and_relation_type_to_idx_dict)
    for node in sorted(G.nodes()):
        if G.has_edge(node, node):
            G.remove_edge(node, node)
        G.add_edge(
            node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    g = DGLGraph(adj)
    for u, v, feature in G.edges(data='subgraph_idx'):
        g.edges[g.edge_id(u, v)].data['subgraph_idx'] = torch.tensor([feature])

    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    return g


def full_load_data(dataset_name, splits_file_path=None, use_raw_normalize=False, model_type=None,
    embedding_method=None, get_degree=False, augment=False, learn_feats=False, p=0.2, clip=True,
    directed=False, include_vnode_labels=False, khops=1, min_degree=0):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, non_valid_samples = full_load_citation(
            dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(
            'new_data', dataset_name, 'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    # Fix uint8 to uint16 for the following line
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    #################### remove the nodes that the label vectors are all zeros, they aren't assigned to any class ############
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        for n_i in non_valid_samples:
            if train_mask[n_i]:
                train_mask[n_i] = False
            elif test_mask[n_i]:
                test_mask[n_i] = False
            elif val_mask[n_i]:
                val_mask[n_i] = False

    num_features = features.shape[1]
    num_labels = len(np.unique(labels)) # NOTE: Original code but not sure if it's correct
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    # pi_avg = []
    # for i in range(num_labels):
    #     num_i = np.sum(labels==i)
    #     pi = num_i/(len(labels)-num_i)
    #     t = pi/(1+pi)
    #     pi_avg.append(t*num_i/len(labels))
    #     print("{}: {}".format(i, t))
    # print(np.sum(pi_avg))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    # Convert scipy sparse matrix to sparse tensor then to dense tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj, model_type=None) # TODO check that model_type=None is enough to handle the case for GPRGNN, when return value is actually edge_index
    adj = _binarize_tensor(adj)

    # Calculate num_labels # TODO Review effect of this on remaining code
    num_labels = torch.unique(labels).max().item() + 1

    # Augment the dataset with virtual nodes
    agg = 'mean' # TODO make this a parameter
    num_vnodes = 0
    vnode_feat_means = None
    if augment:
        adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_vnodes = augment_graph(
            adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels,
            p=p, agg=agg, clip=clip, directed=directed, include_vnode_labels=include_vnode_labels, khops=khops, min_degree=min_degree)
        if learn_feats:
            vnode_feat_means = features[-num_vnodes:, :]
    # Convert dense tensor back to scipy sparse matrix
    adj = dense_tensor_to_sparse_mx(adj)

    # TODO Also update G with these changes for the case of model_type GEOMGCN

    # Preprocessing
    features = preprocess_features(features)

    # Convert numpy arrays to tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    if augment and learn_feats:
        # # Create learnable features
        # vnode_feats = torch.rand((num_vnodes, num_features), dtype=torch.float)
        # features[-num_vnodes:, :] = torch.nn.Parameter(vnode_feats) # requires_grad=True
        # print(f'Features Augmented: {features[-2:, :]}')
        # Remove vnode features to make room for learnable features
        features = features[:-num_vnodes, :]
        pass

    # Normalizing etc
    if get_degree:
        deg_vec = np.array(adj.sum(1))
        deg_vec = deg_vec.flatten()
        raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        deg_vec = None
        raw_adj = None 
    if model_type == 'GEOMGCN':
        g = process_geom(G, dataset_name, embedding_method)
    else:
        g = adj
        if use_raw_normalize:
            # never actually used
            g = row_normalized_adjacency(g)
        else:
            # D^(-1/2) A D^(-1/2). Always used
            g = sys_normalized_adjacency(g)
        g = sparse_mx_to_torch_sparse_tensor(g, model_type)


    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj, num_vnodes, vnode_feat_means
