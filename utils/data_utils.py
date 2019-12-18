#Copyright 2018 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


"""Utils functions to load and process citation data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf
from third_party.gcn.gcn.utils import normalize_adj
from third_party.gcn.gcn.utils import parse_index_file
from third_party.gcn.gcn.utils import sample_mask
from third_party.gcn.gcn.utils import sparse_to_tuple
from third_party.gcn.gcn.utils import preprocess_features


def load_test_edge_mask(dataset_str, data_path, drop_edge_prop):
  """Remove test edges by loading edge masks."""
  edge_mask_path = os.path.join(
      data_path, 'emask.{}.remove{}.npz'.format(dataset_str, drop_edge_prop))
  with tf.gfile.Open(edge_mask_path) as f:
    mask = sp.load_npz(f)
  return mask


def load_edge_masks(dataset_str, data_path, adj_true, drop_edge_prop):
  """Loads adjacency matrix as sparse matrix and masks for val & test links.

  Args:
    dataset_str: dataset to use
    data_path: path to data folder
    adj_true: true adjacency matrix in dense format,
    drop_edge_prop: proportion of edges to remove.

  Returns:
    adj_matrix: adjacency matrix
    train_mask: mask for train edges
    val_mask: mask for val edges
    test_mask: mask for test edges
  """
  edge_mask_path = os.path.join(
      data_path, 'emask.{}.remove{}.'.format(dataset_str, drop_edge_prop))
  val_mask = sp.load_npz(edge_mask_path + 'val.npz')
  test_mask = sp.load_npz(edge_mask_path + 'test.npz')
  train_mask = 1. - val_mask.todense() - test_mask.todense()
  # remove val and test edges from true A
  adj_train = np.multiply(adj_true, train_mask)
  train_mask -= np.eye(train_mask.shape[0])
  return adj_train, sparse_to_tuple(val_mask), sparse_to_tuple(
      val_mask), sparse_to_tuple(test_mask)


def add_top_k_edges(data, edge_mask_path, gae_scores_path, topk, nb_nodes,
                    norm_adj):
  """Loads GAE scores and adds topK edges to train adjacency."""
  test_mask = sp.load_npz(os.path.join(edge_mask_path, 'test_mask.npz'))
  train_mask = 1. - test_mask.todense()
  # remove val and test edges from true A
  adj_train_curr = np.multiply(data['adj_true'], train_mask)
  # Predict test edges using precomputed scores
  scores = np.load(os.path.join(gae_scores_path, 'gae_scores.npy'))
  # scores_mask = 1 - np.eye(nb_nodes)
  scores_mask = np.zeros((nb_nodes, nb_nodes))
  scores_mask[:140, 140:] = 1.
  scores_mask[140:, :140] = 1.
  scores = np.multiply(scores, scores_mask).reshape((-1,))
  threshold = scores[np.argsort(-scores)[topk]]
  adj_train_curr += 1 * (scores > threshold).reshape((nb_nodes, nb_nodes))
  adj_train_curr = 1 * (adj_train_curr > 0)
  if norm_adj:
    adj_train_norm = normalize_adj(data['adj_train'])
  else:
    adj_train_norm = sp.coo_matrix(data['adj_train'])
  return adj_train_curr, sparse_to_tuple(adj_train_norm)


def process_adj(adj, model_name):
  """Symmetrically normalize adjacency matrix."""
  if model_name == 'Cheby':
    laplacian = sp.eye(adj.shape[0]) - normalize_adj(adj - sp.eye(adj.shape[0]))
    # TODO(chamii): compare with
    # adj)
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    laplacian_norm = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    return laplacian_norm
  else:
    return normalize_adj(adj)


def load_data(dataset_str, data_path):
  if dataset_str in ['cora', 'citeseer', 'pubmed']:
    return load_citation_data(dataset_str, data_path)
  else:
    return load_ppi_data(data_path)


def load_ppi_data(data_path):
  """Load PPI dataset."""
  with tf.gfile.Open(os.path.join(data_path, 'ppi.edges.npz')) as f:
    adj = sp.load_npz(f)

  with tf.gfile.Open(os.path.join(data_path, 'ppi.features.norm.npy')) as f:
    features = np.load(f)

  with tf.gfile.Open(os.path.join(data_path, 'ppi.labels.npz')) as f:
    labels = sp.load_npz(f).todense()

  train_mask = np.load(
      tf.gfile.Open(os.path.join(data_path, 'ppi.train_mask.npy'))) > 0
  val_mask = np.load(
      tf.gfile.Open(os.path.join(data_path, 'ppi.test_mask.npy'))) > 0
  test_mask = np.load(
      tf.gfile.Open(os.path.join(data_path, 'ppi.test_mask.npy'))) > 0

  return adj, features, labels, train_mask, val_mask, test_mask


def load_citation_data(dataset_str, data_path):
  """Load data."""
  names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
  objects = {}
  for name in names:
    with tf.gfile.Open(
        os.path.join(data_path, 'ind.{}.{}'.format(dataset_str, name)),
        'rb') as f:
      if sys.version_info > (3, 0):
        objects[name] = pkl.load(f)  # , encoding='latin1') comment to pass lint
      else:
        objects[name] = pkl.load(f)

  test_idx_reorder = parse_index_file(
      os.path.join(data_path, 'ind.{}.test.index'.format(dataset_str)))
  test_idx_range = np.sort(test_idx_reorder)

  if dataset_str == 'citeseer':
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(
        min(test_idx_reorder),
        max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full),
                                 objects['x'].shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = objects['tx']
    objects['tx'] = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full),
                            objects['y'].shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = objects['ty']
    objects['ty'] = ty_extended

  features = sp.vstack((objects['allx'], objects['tx'])).tolil()
  features[test_idx_reorder, :] = features[test_idx_range, :]
  adj = nx.adjacency_matrix(nx.from_dict_of_lists(objects['graph']))

  labels = np.vstack((objects['ally'], objects['ty']))
  labels[test_idx_reorder, :] = labels[test_idx_range, :]

  idx_test = test_idx_range.tolist()
  idx_train = range(len(objects['y']))
  idx_val = range(len(objects['y']), len(objects['y']) + 500)

  train_mask = sample_mask(idx_train, labels.shape[0])
  val_mask = sample_mask(idx_val, labels.shape[0])
  test_mask = sample_mask(idx_test, labels.shape[0])

  features = preprocess_features(features)
  return adj, features, labels, train_mask, val_mask, test_mask


def construct_feed_dict(adj_normalized, adj, features, placeholders):
  # construct feed dictionary
  feed_dict = dict()
  feed_dict.update({placeholders['features']: features})
  feed_dict.update({placeholders['adj']: adj_normalized})
  feed_dict.update({placeholders['adj_orig']: adj})
  return feed_dict


def mask_val_test_edges(adj, prop):
  """Function to mask test and val edges."""
  # NOTE: Splits are randomized and results might slightly
  # deviate from reported numbers in the paper.

  # Remove diagonal elements
  adj = adj - sp.dia_matrix(
      (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
  adj.eliminate_zeros()
  # Check that diag is zero:
  assert np.diag(adj.todense()).sum() == 0

  adj_triu = sp.triu(adj)
  adj_tuple = sparse_to_tuple(adj_triu)
  edges = adj_tuple[0]
  edges_all = sparse_to_tuple(adj)[0]
  num_test = int(np.floor(edges.shape[0] * prop))
  # num_val = int(np.floor(edges.shape[0] * 0.05))  # we keep 5% for validation
  # we keep 10% of training edges for validation
  num_val = int(np.floor((edges.shape[0] - num_test) * 0.05))

  all_edge_idx = range(edges.shape[0])
  np.random.shuffle(all_edge_idx)
  val_edge_idx = all_edge_idx[:num_val]
  test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
  test_edges = edges[test_edge_idx]
  val_edges = edges[val_edge_idx]
  train_edges = np.delete(
      edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

  def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)

  test_edges_false = []
  while len(test_edges_false) < len(test_edges):
    idx_i = np.random.randint(0, adj.shape[0])
    idx_j = np.random.randint(0, adj.shape[0])
    if idx_i == idx_j:
      continue
    if ismember([idx_i, idx_j], edges_all):
      continue
    if test_edges_false:
      if ismember([idx_j, idx_i], np.array(test_edges_false)):
        continue
      if ismember([idx_i, idx_j], np.array(test_edges_false)):
        continue
    test_edges_false.append([idx_i, idx_j])

  val_edges_false = []
  while len(val_edges_false) < len(val_edges):
    idx_i = np.random.randint(0, adj.shape[0])
    idx_j = np.random.randint(0, adj.shape[0])
    if idx_i == idx_j:
      continue
    if ismember([idx_i, idx_j], train_edges):
      continue
    if ismember([idx_j, idx_i], train_edges):
      continue
    if ismember([idx_i, idx_j], val_edges):
      continue
    if ismember([idx_j, idx_i], val_edges):
      continue
    if val_edges_false:
      if ismember([idx_j, idx_i], np.array(val_edges_false)):
        continue
      if ismember([idx_i, idx_j], np.array(val_edges_false)):
        continue
    val_edges_false.append([idx_i, idx_j])

  assert ~ismember(test_edges_false, edges_all)
  assert ~ismember(val_edges_false, edges_all)
  assert ~ismember(val_edges, train_edges)
  assert ~ismember(test_edges, train_edges)
  assert ~ismember(val_edges, test_edges)

  data = np.ones(train_edges.shape[0])

  # Re-build adj matrix
  adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                            shape=adj.shape)
  adj_train = adj_train + adj_train.T

  # NOTE: these edge lists only contain single direction of edge!
  num_nodes = adj.shape[0]
  val_mask = np.zeros((num_nodes, num_nodes))
  for i, j in val_edges:
    val_mask[i, j] = 1
    val_mask[j, i] = 1
  for i, j in val_edges_false:
    val_mask[i, j] = 1
    val_mask[j, i] = 1
  test_mask = np.zeros((num_nodes, num_nodes))
  for i, j in test_edges:
    test_mask[i, j] = 1
    test_mask[j, i] = 1
  for i, j in test_edges_false:
    test_mask[i, j] = 1
    test_mask[j, i] = 1
  return adj_train, sparse_to_tuple(val_mask), sparse_to_tuple(test_mask)


def mask_test_edges(adj, prop):
  """Function to mask test edges.

  Args:
    adj: scipy sparse matrix
    prop: proportion of edges to remove (float in [0, 1])

  Returns:
    adj_train: adjacency with edges removed
    test_edges: list of positive and negative test edges
  """
  # Remove diagonal elements
  adj = adj - sp.dia_matrix(
      (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
  adj.eliminate_zeros()
  # Check that diag is zero:
  assert np.diag(adj.todense()).sum() == 0

  adj_triu = sp.triu(adj)
  adj_tuple = sparse_to_tuple(adj_triu)
  edges = adj_tuple[0]
  edges_all = sparse_to_tuple(adj)[0]
  num_test = int(np.floor(edges.shape[0] * prop))

  all_edge_idx = range(edges.shape[0])
  np.random.shuffle(all_edge_idx)
  test_edge_idx = all_edge_idx[:num_test]
  test_edges = edges[test_edge_idx]
  train_edges = np.delete(edges, test_edge_idx, axis=0)

  def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)

  test_edges_false = []
  while len(test_edges_false) < len(test_edges):
    idx_i = np.random.randint(0, adj.shape[0])
    idx_j = np.random.randint(0, adj.shape[0])
    if idx_i == idx_j:
      continue
    if ismember([idx_i, idx_j], edges_all):
      continue
    if test_edges_false:
      if ismember([idx_j, idx_i], np.array(test_edges_false)):
        continue
      if ismember([idx_i, idx_j], np.array(test_edges_false)):
        continue
    test_edges_false.append([idx_i, idx_j])

  assert ~ismember(test_edges_false, edges_all)
  assert ~ismember(test_edges, train_edges)

  data = np.ones(train_edges.shape[0])

  # Re-build adj matrix
  adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                            shape=adj.shape)
  adj_train = adj_train + adj_train.T

  # NOTE: these edge lists only contain single direction of edge!
  num_nodes = adj.shape[0]
  test_mask = np.zeros((num_nodes, num_nodes))
  for i, j in test_edges:
    test_mask[i, j] = 1
    test_mask[j, i] = 1
  for i, j in test_edges_false:
    test_mask[i, j] = 1
    test_mask[j, i] = 1
  return adj_train, sparse_to_tuple(test_mask)
