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


"""Heuristics for link prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from data_utils import mask_test_edges
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sklearn.metrics as skm


flags.DEFINE_string('adj_path', '../data/cora.adj.npz', 'path to graph to use.')
flags.DEFINE_string('prop_drop', '10-30-50', 'proportion of edges to remove.')
flags.DEFINE_string('methods', 'svd-katz-common_neighbours',
                    'which methods to use')
FLAGS = flags.FLAGS


class LinkPredictionHeuristcs(object):
  """Link prediction heuristics."""

  def __init__(self, adj_matrix):
    self.adj_matrix = adj_matrix

  def common_neighbours(self):
    """Computes scores for each node pair based on common neighbours."""
    scores = self.adj_matrix.dot(self.adj_matrix)
    return scores

  def svd(self, rank=64):
    """Computes scores using low rank factorization with SVD."""
    adj_matrix = self.adj_matrix.asfptype()
    u, s, v = sp.linalg.svds(A=adj_matrix, k=rank)
    adj_low_rank = u.dot(np.diag(s).dot(v))
    return adj_low_rank

  def adamic_adar(self):
    """Computes adamic adar scores."""
    graph = nx.from_scipy_sparse_matrix(self.adj_matrix)
    scores = nx.adamic_adar_index(graph)
    return scores

  def jaccard_coeff(self):
    """Computes Jaccard coefficients."""
    graph = nx.from_scipy_sparse_matrix(self.adj_matrix)
    coeffs = nx.jaccard_coefficient(graph)
    return coeffs

  def katz(self, beta=0.001, steps=25):
    """Computes Katz scores."""
    coeff = beta
    katz_scores = beta * self.adj_matrix
    adj_power = self.adj_matrix
    for _ in range(2, steps + 1):
      adj_power = adj_power.dot(self.adj_matrix)
      katz_scores += coeff * adj_power
      coeff *= beta
    return katz_scores


def get_scores_from_generator(gen, nb_nodes=2708):
  """Helper function to get scores in numpy array format from generator."""
  adj = np.zeros((nb_nodes, nb_nodes))
  for i, j, score in gen:
    adj[i, j] = score
  return adj


def compute_lp_metrics(edges, true_adj, pred_adj):
  """Computes link prediction scores on test edges."""
  labels = np.array(true_adj[edges]).reshape((-1,))
  scores = np.array(pred_adj[edges]).reshape((-1,))
  roc = skm.roc_auc_score(labels, scores)
  ap = skm.average_precision_score(labels, scores)
  return roc, ap


if __name__ == '__main__':
  adj_true = sp.load_npz(FLAGS.adj_path).todense()
  lp = LinkPredictionHeuristcs(adj_true)
  for delete_prop in FLAGS.prop_drop.split('-'):
    for method in FLAGS.methods.split('-'):
      lp_func = getattr(lp, method)
      adj_train, test_edges = mask_test_edges(
          adj_true, float(delete_prop) * 0.01)
      adj_scores = lp_func(adj_train).todense()
      roc_score, ap_score = compute_lp_metrics(test_edges, adj_true, adj_scores)
      print('method={} | prop={} | roc_auc={} ap={}\n'.format(
          method, delete_prop, round(roc_score, 4), round(ap_score, 4)))
