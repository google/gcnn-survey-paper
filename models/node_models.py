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


"""Inference step for node classification models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.base_models import NodeModel
import tensorflow as tf
from utils.model_utils import cheby_module
from utils.model_utils import compute_adj
from utils.model_utils import gat_module
from utils.model_utils import gcn_module
from utils.model_utils import gcn_pool_layer
from utils.model_utils import mlp_module
from utils.model_utils import sp_gat_layer
from utils.model_utils import sp_gcn_layer


class Gat(NodeModel):
  """Graph Attention (GAT) Model (Velickovic & al).

  arXiv link: https://arxiv.org/abs/1710.10903
  """

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for GAT model."""
    sparse = self.sparse_features
    in_dim = self.input_dim
    average_last = True
    with tf.variable_scope('node-model'):
      logits = gat_module(node_features, adj_matrix, self.n_hidden, self.n_att,
                          self.p_drop, is_training, in_dim, sparse,
                          average_last)
    return logits


class Gcn(NodeModel):
  """Graph convolution network (Kipf & al).

  arXiv link: https://arxiv.org/abs/1609.02907
  """

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for graph convolution model."""
    with tf.variable_scope('node-model'):
      logits = gcn_module(node_features, adj_matrix, self.n_hidden, self.p_drop,
                          is_training, self.input_dim, self.sparse_features)
    return logits


class Mlp(NodeModel):
  """Multi-layer perceptron model."""

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for graph convolution model."""
    with tf.variable_scope('node-model'):
      logits = mlp_module(node_features, self.n_hidden, self.p_drop,
                          is_training, self.input_dim, self.sparse_features,
                          use_bias=True)
    return logits


class SemiEmb(NodeModel):
  """Deep Learning via Semi-Supervised Embedding (Weston & al).

  paper: http://icml2008.cs.helsinki.fi/papers/340.pdf
  """

  def __init__(self, config):
    super(SemiEmb, self).__init__(config)
    self.semi_emb_k = config.semi_emb_k

  def compute_inference(self, node_features, adj_matrix, is_training):
    with tf.variable_scope('node-model'):
      hidden_repr = mlp_module(node_features, self.n_hidden, self.p_drop,
                               is_training, self.input_dim,
                               self.sparse_features, use_bias=True,
                               return_hidden=True)
      logits = hidden_repr[-1]
      hidden_repr_reg = hidden_repr[self.semi_emb_k]
      l2_scores = compute_adj(hidden_repr_reg, self.att_mechanism, self.p_drop,
                              is_training=False)
      self.l2_scores = tf.gather_nd(l2_scores, adj_matrix.indices)
    return logits

  def _compute_node_loss(self, logits, labels):
    supervised_loss = super(SemiEmb, self)._compute_node_loss(logits, labels)
    # supervised_loss = tf.nn.softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logits)
    # supervised_loss = tf.reduce_sum(supervised_loss) / self.nb_nodes
    reg_loss = tf.reduce_mean(self.l2_scores)
    return supervised_loss + self.edge_reg * reg_loss


class Cheby(NodeModel):
  """Chebyshev polynomials for Spectral Graph Convolutions (Defferrard & al).

  arXiv link: https://arxiv.org/abs/1606.09375
  """

  def __init__(self, config):
    super(Cheby, self).__init__(config)
    self.cheby_k_loc = config.cheby_k_loc

  def compute_inference(self, node_features, normalized_laplacian, is_training):
    with tf.variable_scope('node-model'):
      dense_normalized_laplacian = tf.sparse_to_dense(
          sparse_indices=normalized_laplacian.indices,
          output_shape=normalized_laplacian.dense_shape,
          sparse_values=normalized_laplacian.values)
      cheby_polynomials = [tf.eye(self.nb_nodes), dense_normalized_laplacian]
      self.cheby = cheby_polynomials
      for _ in range(2, self.cheby_k_loc+1):
        cheby_polynomials.append(2 * tf.sparse_tensor_dense_matmul(
            normalized_laplacian, cheby_polynomials[-1]) - cheby_polynomials[-2]
                                )
      logits = cheby_module(node_features, cheby_polynomials, self.n_hidden,
                            self.p_drop, is_training, self.input_dim,
                            self.sparse_features)
    return logits


############################ EXPERIMENTAL MODELS #############################


class Hgat(NodeModel):
  """Hierarchical Graph Attention (GAT) Model."""

  def compute_inference(self, node_features, adj_matrix, is_training):
    """Forward step for GAT model."""
    in_dim = self.input_dim
    att = []
    for j in range(4):
      with tf.variable_scope('gat-layer1-att{}'.format(j)):
        att.append(
            sp_gat_layer(node_features, adj_matrix, in_dim, 8, self.p_drop,
                         is_training, True))
    hidden_2 = []
    hidden_2.append(tf.nn.elu(tf.concat(att[:2], axis=-1)))
    hidden_2.append(tf.nn.elu(tf.concat(att[2:], axis=-1)))
    att = []
    for j in range(2):
      with tf.variable_scope('gat-layer2-att{}'.format(j)):
        att.append(
            sp_gat_layer(hidden_2[j], adj_matrix, 16, 7, self.p_drop,
                         is_training, False))
    return tf.add_n(att) / 2.


class Pgcn(NodeModel):
  """Pooling Graph Convolution Network."""

  def compute_inference(self, node_features, adj_matrix, is_training):
    adj_matrix_dense = tf.sparse_to_dense(
        sparse_indices=adj_matrix.indices,
        output_shape=adj_matrix.dense_shape,
        sparse_values=adj_matrix.values,
        validate_indices=False)
    adj_matrix_dense = tf.cast(tf.greater(adj_matrix_dense, 0), tf.float32)
    adj_matrix_dense = tf.expand_dims(adj_matrix_dense, -1)  # N x N x 1
    in_dim = self.input_dim
    sparse = self.sparse_features
    for i, out_dim in enumerate(self.n_hidden[:-1]):
      if i > 0:
        sparse = False
      with tf.variable_scope('gcn-pool-{}'.format(i)):
        node_features = gcn_pool_layer(
            node_features,
            adj_matrix_dense,
            in_dim=in_dim,
            out_dim=out_dim,
            sparse=sparse,
            is_training=is_training,
            p_drop=self.p_drop)
        node_features = tf.reshape(node_features, (-1, out_dim))
        node_features = tf.contrib.layers.bias_add(node_features)
        node_features = tf.nn.elu(node_features)
        in_dim = out_dim
    with tf.variable_scope('gcn-layer-last'):
      logits = sp_gcn_layer(node_features, adj_matrix, in_dim,
                            self.n_hidden[-1], self.p_drop, is_training, False)
    return logits

