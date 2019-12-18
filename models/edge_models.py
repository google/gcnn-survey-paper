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


"""Inference step for link prediction models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.base_models import EdgeModel
import tensorflow as tf
from utils.model_utils import compute_adj
from utils.model_utils import gat_module
from utils.model_utils import gcn_module
from utils.model_utils import mlp_module


class Gae(EdgeModel):
  """Graph Auto-Encoder (GAE) (Kipf & al) for link prediction.

  arXiv link: https://arxiv.org/abs/1611.07308
  """

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for GAE model."""
    sparse = self.sparse_features
    in_dim = self.input_dim
    with tf.variable_scope('edge-model'):
      h0 = gcn_module(node_features, adj_matrix, self.n_hidden, self.p_drop,
                      is_training, in_dim, sparse)
      adj_matrix_pred = compute_adj(h0, self.att_mechanism, self.p_drop,
                                    is_training)
    self.adj_matrix_pred = tf.nn.sigmoid(adj_matrix_pred)
    return adj_matrix_pred


class Egat(EdgeModel):
  """Edge-GAT for link prediction."""

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for GAE model."""
    sparse = self.sparse_features
    in_dim = self.input_dim
    with tf.variable_scope('edge-model'):
      h0 = gat_module(
          node_features,
          adj_matrix,
          self.n_hidden,
          self.n_att,
          self.p_drop,
          is_training,
          in_dim,
          sparse,
          average_last=True)
      adj_matrix_pred = compute_adj(h0, self.att_mechanism, self.p_drop,
                                    is_training)
    self.adj_matrix_pred = tf.nn.sigmoid(adj_matrix_pred)
    return adj_matrix_pred


class Vgae(EdgeModel):
  """Variational Graph Auto-Encoder (VGAE) (Kipf & al) for link prediction.

  arXiv link: https://arxiv.org/abs/1611.07308
  """

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for GAE model."""
    sparse = self.sparse_features
    in_dim = self.input_dim
    with tf.variable_scope('edge-model'):
      h0 = gcn_module(node_features, adj_matrix, self.n_hidden[:-1],
                      self.p_drop, is_training, in_dim, sparse)
      # N x F
      with tf.variable_scope('mean'):
        z_mean = gcn_module(h0, adj_matrix, self.n_hidden[-1:], self.p_drop,
                            is_training, self.n_hidden[-2], False)
        self.z_mean = z_mean
      with tf.variable_scope('std'):
        # N x F
        z_log_std = gcn_module(h0, adj_matrix, self.n_hidden[-1:], self.p_drop,
                               is_training, self.n_hidden[-2], False)
        self.z_log_std = z_log_std
      # add noise during training
      noise = tf.random_normal([self.nb_nodes, self.n_hidden[-1]
                               ]) * tf.exp(z_log_std)
      z = tf.cond(is_training, lambda: tf.add(z_mean, noise),
                  lambda: z_mean)
      # N x N
      adj_matrix_pred = compute_adj(z, self.att_mechanism, self.p_drop,
                                    is_training)
    self.adj_matrix_pred = tf.nn.sigmoid(adj_matrix_pred)
    return adj_matrix_pred

  def _compute_edge_loss(self, adj_pred, adj_train):
    """Overrides _compute_edge_loss to add Variational Inference objective."""
    log_lik = super(Vgae, self)._compute_edge_loss(adj_pred, adj_train)
    norm = self.nb_nodes**2 / float((self.nb_nodes**2 - self.nb_edges) * 2)
    kl_mat = 0.5 * tf.reduce_sum(
        1 + 2 * self.z_log_std - tf.square(self.z_mean) - tf.square(
            tf.exp(self.z_log_std)), 1)
    kl = tf.reduce_mean(kl_mat) / self.nb_nodes
    edge_loss = norm * log_lik - kl
    return edge_loss


class Emlp(EdgeModel):
  """Simple baseline for link prediction.

  Creates a tensorflow graph to train and evaluate EMLP on graph data.
  """

  def compute_inference(self, node_features, _, is_training):
    """Forward step for GAE model."""
    sparse = self.sparse_features
    in_dim = self.input_dim
    with tf.variable_scope('edge-model'):
      h0 = mlp_module(
          node_features,
          self.n_hidden,
          self.p_drop,
          is_training,
          in_dim,
          sparse,
          use_bias=False)
      adj_matrix_pred = compute_adj(h0, self.att_mechanism, self.p_drop,
                                    is_training)
    self.adj_matrix_pred = tf.nn.sigmoid(adj_matrix_pred)
    return adj_matrix_pred
