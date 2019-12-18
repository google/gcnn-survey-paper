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


"""Utils functions for GNN models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

WEIGHT_INIT = tf.contrib.layers.xavier_initializer()
BIAS_INIT = tf.zeros_initializer()


############################## LAYERS #############################


def sparse_dropout(tensor, p_drop, is_training):
  """Dropout with sparse tensor."""
  return tf.SparseTensor(
      indices=tensor.indices,
      values=tf.layers.dropout(
          inputs=tensor.values,
          rate=p_drop,
          training=is_training),
      dense_shape=tensor.dense_shape)


def dense(node_features,
          in_dim,
          out_dim,
          p_drop,
          is_training,
          sparse,
          use_bias=False):
  """Dense layer with sparse or dense tensor and dropout."""
  w_dense = tf.get_variable(
      initializer=WEIGHT_INIT,
      dtype=tf.float32,
      name='linear',
      shape=(in_dim, out_dim))
  if sparse:
    node_features = sparse_dropout(node_features, p_drop, is_training)
    node_features = tf.sparse_tensor_dense_matmul(node_features, w_dense)
  else:
    node_features = tf.layers.dropout(
        inputs=node_features, rate=p_drop, training=is_training)
    node_features = tf.matmul(node_features, w_dense)
  if use_bias:
    node_features = tf.contrib.layers.bias_add(node_features)
  return node_features


def sp_gcn_layer(node_features, adj_matrix, in_dim, out_dim, p_drop,
                 is_training, sparse):
  """Single graph convolution layer with sparse tensors AXW.

  Args:
    node_features: Tensor of shape (nb_nodes, in_dim) or SparseTensor.
    adj_matrix: Sparse Tensor, normalized adjacency matrix.
    in_dim: integer specifying the input feature dimension.
    out_dim: integer specifying the output feature dimension.
    p_drop: dropout probability.
    is_training: boolean, True if the model is being trained, False otherwise.
    sparse: True if node_features are sparse.

  Returns:
    node_features: tensor of shape (nb_nodes, out_dim). New node
        features obtained from applying one GCN layer.

  Raises:
  """
  node_features = dense(node_features, in_dim, out_dim, p_drop, is_training,
                        sparse)
  node_features = tf.layers.dropout(
      inputs=node_features, rate=p_drop, training=is_training)
  node_features = tf.sparse_tensor_dense_matmul(adj_matrix, node_features)
  return node_features


def gcn_layer(node_features, adj_matrix, in_dim, out_dim, p_drop, is_training,
              sparse):
  """Single graph convolution layer with dense A.

  Args:
    node_features: Tensor of shape (nb_nodes, in_dim) or SparseTensor.
    adj_matrix: Tensor, normalized adjacency matrix.
    in_dim: integer specifying the input feature dimension.
    out_dim: integer specifying the output feature dimension.
    p_drop: dropout probability.
    is_training: boolean, True if the model is being trained, False otherwise.
    sparse: True if node_features are sparse.

  Returns:
    node_features: tensor of shape (nb_nodes, out_dim). New node
        features obtained from applying one GCN layer.

  Raises:
  """
  node_features = dense(node_features, in_dim, out_dim, p_drop, is_training,
                        sparse)
  node_features = tf.layers.dropout(
      inputs=node_features, rate=p_drop, training=is_training)
  node_features = tf.matmul(adj_matrix, node_features)
  return node_features


def gcn_pool_layer(node_features, adj_matrix, in_dim, out_dim, sparse,
                   is_training, p_drop):
  """GCN with maxpooling over neighbours instead of avreaging."""
  node_features = dense(node_features, in_dim, out_dim, p_drop, is_training,
                        sparse)
  node_features = tf.expand_dims(node_features, 0)  # 1 x N x d
  # broadcasting (adj in N x N x 1 and features are 1 x N x d)
  node_features = tf.multiply(node_features, adj_matrix)
  node_features = tf.transpose(node_features, perm=[0, 2, 1])
  node_features = tf.reduce_max(node_features, axis=-1)  # N x d
  return node_features


def sp_gat_layer(node_features, adj_matrix, in_dim, out_dim, p_drop,
                 is_training, sparse):
  """Single graph attention layer using sparse tensors.

  Args:
    node_features: Sparse Tensor of shape (nb_nodes, in_dim) or SparseTensor.
    adj_matrix: Sparse Tensor.
    in_dim: integer specifying the input feature dimension.
    out_dim: integer specifying the output feature dimension.
    p_drop: dropout probability.
    is_training: boolean, True if the model is being trained, False otherwise
    sparse: True if node features are sparse.

  Returns:
    node_features: tensor of shape (nb_nodes, out_dim). New node
        features obtained from applying one head of attention to input.

  Raises:
  """
  # Linear transform
  node_features = dense(node_features, in_dim, out_dim, p_drop, is_training,
                        sparse)
  # Attention scores
  alpha = sp_compute_adj_att(node_features, adj_matrix)
  alpha = tf.SparseTensor(
      indices=alpha.indices,
      values=tf.nn.leaky_relu(alpha.values),
      dense_shape=alpha.dense_shape)
  alpha = tf.sparse_softmax(alpha)
  alpha = sparse_dropout(alpha, p_drop, is_training)
  node_features = tf.layers.dropout(
      inputs=node_features, rate=p_drop, training=is_training)
  # Compute self-attention features
  node_features = tf.sparse_tensor_dense_matmul(alpha, node_features)
  node_features = tf.contrib.layers.bias_add(node_features)
  return node_features


def gat_layer(node_features, adj_matrix, out_dim, p_drop, is_training, i, j):
  """Single graph attention layer.

  Args:
    node_features: Tensor of shape (nb_nodes, feature_dim)
    adj_matrix: adjacency matrix. Tensor of shape (nb_nodes, nb_nodes) and type
      float. There should be 1 if there is a connection between two nodes and 0
      otherwise.
    out_dim: integer specifying the output feature dimension.
    p_drop: dropout probability.
    is_training: boolean, True if the model is being trained, False otherwise
    i: layer index, used for naming variables
    j: attention mechanism index, used for naming variables

  Returns:
    node_features: tensor of shape (nb_nodes, out_dim). New node
        features obtained from applying one head of attention to input.

  Raises:
  """
  with tf.variable_scope('gat-{}-{}'.format(i, j)):
    node_features = tf.layers.dropout(
        inputs=node_features, rate=p_drop, training=is_training)
    # Linear transform of the features
    w_dense = tf.get_variable(
        initializer=WEIGHT_INIT,
        dtype=tf.float32,
        name='linear',
        shape=(node_features.shape[1], out_dim))
    node_features = tf.matmul(node_features, w_dense)
    alpha = compute_adj_att(node_features)
    alpha = tf.nn.leaky_relu(alpha)
    # Mask values before activation to inject the graph structure
    # Add -infinity to corresponding pairs before normalization
    bias_mat = -1e9 * (1. - adj_matrix)
    # multiply here if adjacency is weighted
    alpha = tf.nn.softmax(alpha + bias_mat, axis=-1)
    # alpha = tf.nn.softmax(alpha, axis=-1)
    alpha = tf.layers.dropout(inputs=alpha, rate=p_drop, training=is_training)
    node_features = tf.layers.dropout(
        inputs=node_features, rate=p_drop, training=is_training)
    # Compute self-attention features
    node_features = tf.matmul(alpha, node_features)
    node_features = tf.contrib.layers.bias_add(node_features)
    return node_features


def sp_egat_layer(node_features, adj_matrix, in_dim, out_dim, p_drop,
                  is_training, sparse):
  """Single graph attention layer using sparse tensors.

  Args:
    node_features: Tensor of shape (nb_nodes, in_dim) or SparseTensor.
    adj_matrix: Sparse Tensor.
    in_dim: integer specifying the input feature dimension.
    out_dim: integer specifying the output feature dimension.
    p_drop: dropout probability.
    is_training: boolean, True if the model is being trained, False otherwise
    sparse: True if node features are sparse.

  Returns:
    node_features: tensor of shape (nb_nodes, out_dim). New node
        features obtained from applying one head of attention to input.

  Raises:
  """
  # Linear transform
  node_features = dense(node_features, in_dim, out_dim, p_drop, is_training,
                        sparse)
  # Attention scores
  alpha = sp_compute_adj_att(node_features, adj_matrix)
  alpha = tf.SparseTensor(
      indices=alpha.indices,
      values=tf.nn.leaky_relu(alpha.values),
      dense_shape=alpha.dense_shape)
  alpha = tf.sparse_softmax(alpha)
  alpha = sparse_dropout(alpha, p_drop, is_training)
  node_features = tf.layers.dropout(
      inputs=node_features, rate=p_drop, training=is_training)
  # Compute self-attention features
  node_features = tf.sparse_tensor_dense_matmul(alpha, node_features)
  node_features = tf.contrib.layers.bias_add(node_features)
  return node_features


############################## MULTI LAYERS #############################


def mlp_module(node_features, n_hidden, p_drop, is_training, in_dim,
               sparse_features, use_bias, return_hidden=False):
  """MLP."""
  nb_layers = len(n_hidden)
  hidden_layers = [node_features]
  for i, out_dim in enumerate(n_hidden):
    with tf.variable_scope('mlp-{}'.format(i)):
      if i > 0:
        sparse_features = False
      if i == nb_layers - 1:
        use_bias = False
      h_i = dense(hidden_layers[-1], in_dim, out_dim, p_drop, is_training,
                  sparse_features, use_bias)
      if i < nb_layers - 1:
        h_i = tf.nn.relu(h_i)
        in_dim = out_dim
      hidden_layers.append(h_i)
  if return_hidden:
    return hidden_layers
  else:
    return hidden_layers[-1]


def gcn_module(node_features, adj_matrix, n_hidden, p_drop, is_training, in_dim,
               sparse_features):
  """GCN module with multiple layers."""
  nb_layers = len(n_hidden)
  for i, out_dim in enumerate(n_hidden):
    if i > 0:
      sparse_features = False
    with tf.variable_scope('gcn-{}'.format(i)):
      node_features = sp_gcn_layer(node_features, adj_matrix, in_dim, out_dim,
                                   p_drop, is_training, sparse_features)
    if i < nb_layers - 1:
      node_features = tf.nn.relu(node_features)
      in_dim = out_dim
  return node_features


def cheby_module(node_features, cheby_poly, n_hidden, p_drop, is_training,
                 in_dim, sparse_features):
  """GCN module with multiple layers."""
  nb_layers = len(n_hidden)
  for i, out_dim in enumerate(n_hidden):
    if i > 0:
      sparse_features = False
    feats = []
    for j, poly in enumerate(cheby_poly):
      with tf.variable_scope('cheb-{}-{}'.format(i, j)):
        sparse_poly = tf.contrib.layers.dense_to_sparse(poly)
        feats.append(sp_gcn_layer(node_features, sparse_poly, in_dim, out_dim,
                                  p_drop, is_training, sparse_features))
    node_features = tf.add_n(feats)
    if i < nb_layers - 1:
      node_features = tf.nn.relu(node_features)
      in_dim = out_dim
  return node_features


def gat_module(node_features, adj_matrix, n_hidden, n_att, p_drop, is_training,
               in_dim, sparse_features, average_last):
  """GAT module with muli-headed attention and multiple layers."""
  nb_layers = len(n_att)
  for i, k in enumerate(n_att):
    out_dim = n_hidden[i]
    att = []
    if i > 0:
      sparse_features = False
    for j in range(k):
      with tf.variable_scope('gat-layer{}-att{}'.format(i, j)):
        att.append(
            sp_gat_layer(node_features, adj_matrix, in_dim, out_dim, p_drop,
                         is_training, sparse_features))
    # intermediate layers, concatenate features
    if i < nb_layers - 1:
      in_dim = out_dim * k
      node_features = tf.nn.elu(tf.concat(att, axis=-1))
  if average_last:
    # last layer, average features instead of concatenating
    logits = tf.add_n(att) / n_att[-1]
  else:
    logits = tf.concat(att, axis=-1)
  return logits


def egat_module(node_features, adj_matrix, n_hidden, n_att, p_drop, is_training,
                in_dim, sparse_features, average_last):
  """Edge-GAT module with muli-headed attention and multiple layers."""
  nb_layers = len(n_att)
  for i, k in enumerate(n_att):
    out_dim = n_hidden[i]
    att = []
    if i > 0:
      sparse_features = False
    for j in range(k):
      with tf.variable_scope('egat-layer{}-att{}'.format(i, j)):
        att.append(
            sp_gat_layer(node_features, adj_matrix, in_dim, out_dim, p_drop,
                         is_training, sparse_features))
    # intermediate layers, concatenate features
    if i < nb_layers - 1:
      in_dim = out_dim * k
      node_features = tf.nn.elu(tf.concat(att, axis=-1))
  if average_last:
    # last layer, average features instead of concatenating
    logits = tf.add_n(att) / n_att[-1]
  else:
    logits = tf.concat(att, axis=-1)
  return logits


###################### EDGE SCORES FUNCTIONS #############################


def sp_compute_adj_att(node_features, adj_matrix_sp):
  """Self-attention for edges as in GAT with sparse adjacency."""
  out_dim = node_features.shape[-1]
  # Self-attention mechanism
  a_row = tf.get_variable(
      initializer=WEIGHT_INIT,
      dtype=tf.float32,
      name='selfatt-row',
      shape=(out_dim, 1))
  a_col = tf.get_variable(
      initializer=WEIGHT_INIT,
      dtype=tf.float32,
      name='selfatt-col',
      shape=(out_dim, 1))
  alpha_row = tf.matmul(node_features, a_row)
  alpha_col = tf.matmul(node_features, a_col)
  # Compute matrix with self-attention scores using broadcasting
  alpha = tf.sparse_add(adj_matrix_sp * alpha_row,
                        adj_matrix_sp * tf.transpose(alpha_col, perm=[1, 0]))
  return alpha


def compute_adj_att(node_features):
  """Self-attention for edges as in GAT."""
  out_dim = node_features.shape[-1]
  # Self-attention mechanism
  a_row = tf.get_variable(
      initializer=WEIGHT_INIT,
      dtype=tf.float32,
      name='selfatt-row',
      shape=(out_dim, 1))
  a_col = tf.get_variable(
      initializer=WEIGHT_INIT,
      dtype=tf.float32,
      name='selfatt-col',
      shape=(out_dim, 1))
  alpha_row = tf.matmul(node_features, a_row)
  alpha_col = tf.matmul(node_features, a_col)
  # Compute matrix with self-attention scores using broadcasting
  alpha = alpha_row + tf.transpose(alpha_col, perm=[1, 0])
  # alpha += alpha_col + tf.transpose(alpha_row, perm=[1, 0])
  return alpha


def compute_weighted_mat_dot(node_features, nb_dots=1):
  """Compute weighted dot with matrix multiplication."""
  adj_scores = []
  in_dim = node_features.shape[-1]
  for i in range(nb_dots):
    weight_mat = tf.get_variable(
        initializer=WEIGHT_INIT,
        dtype=tf.float32,
        name='w-dot-{}'.format(i),
        shape=(in_dim, in_dim))
    adj_scores.append(tf.matmul(node_features, tf.matmul(
        weight_mat, tf.transpose(node_features, perm=[1, 0]))))
  return tf.add_n(adj_scores)


def compute_weighted_dot(node_features, nb_dots=4):
  """Compute weighted dot product."""
  adj_scores = []
  in_dim = node_features.shape[-1]
  for i in range(nb_dots):
    weight_vec = tf.get_variable(
        initializer=WEIGHT_INIT,
        dtype=tf.float32,
        name='w-dot-{}'.format(i),
        shape=(1, in_dim))
    weight_vec = tf.nn.softmax(weight_vec, axis=-1)
    adj_scores.append(tf.matmul(tf.multiply(weight_vec, node_features),
                                tf.transpose(node_features, perm=[1, 0])))
  return tf.add_n(adj_scores)


def compute_l2_sim_matrix(node_features):
  """Compute squared-L2 distance between each pair of nodes."""
  # N x N
  # d_scores = tf.matmul(node_features, tf.transpose(node_features,perm=[1, 0]))
  # diag = tf.diag_part(d_scores)
  # d_scores *= -2.
  # d_scores += tf.reshape(diag, (-1, 1)) + tf.reshape(diag, (1, -1))
  l2_norm = tf.reduce_sum(tf.square(node_features), 1)
  na = tf.reshape(l2_norm, [-1, 1])
  nb = tf.reshape(l2_norm, [1, -1])
  # return pairwise euclidead difference matrix
  l2_scores = tf.maximum(
      na - 2*tf.matmul(node_features, node_features, False, True) + nb, 0.0)
  return l2_scores


def compute_dot_sim_matrix(node_features):
  """Compute edge scores with dot product."""
  sim = tf.matmul(node_features, tf.transpose(node_features, perm=[1, 0]))
  return sim


def compute_dot_norm(features):
  """Compute edge scores with normalized dot product."""
  features = tf.nn.l2_normalize(features, axis=-1)
  sim = tf.matmul(features, tf.transpose(features, perm=[1, 0]))
  return sim


def compute_asym_dot(node_features):
  """Compute edge scores with asymmetric dot product."""
  feat_left, feat_right = tf.split(node_features, 2, axis=-1)
  feat_left = tf.nn.l2_normalize(feat_left, axis=-1)
  feat_right = tf.nn.l2_normalize(feat_right, axis=-1)
  sim = tf.matmul(feat_left, tf.transpose(feat_right, perm=[1, 0]))
  return sim


def compute_adj(features, att_mechanism, p_drop, is_training):
  """Compute adj matrix given node features."""
  features = tf.layers.dropout(
      inputs=features, rate=p_drop, training=is_training)
  if att_mechanism == 'dot':
    return compute_dot_sim_matrix(features)
  elif att_mechanism == 'weighted-mat-dot':
    return compute_weighted_mat_dot(features)
  elif att_mechanism == 'weighted-dot':
    return compute_weighted_dot(features)
  elif att_mechanism == 'att':
    return compute_adj_att(features)
  elif att_mechanism == 'dot-norm':
    return compute_dot_norm(features)
  elif att_mechanism == 'asym-dot':
    return compute_asym_dot(features)
  else:
    return compute_l2_sim_matrix(features)


def get_sp_topk(adj_pred, sp_adj_train, nb_nodes, k):
  """Returns binary matrix with topK."""
  _, indices = tf.nn.top_k(tf.reshape(adj_pred, (-1,)), k)
  indices = tf.reshape(tf.cast(indices, tf.int64), (-1, 1))
  sp_adj_pred = tf.SparseTensor(
      indices=indices,
      values=tf.ones(k),
      dense_shape=(nb_nodes * nb_nodes,))
  sp_adj_pred = tf.sparse_reshape(sp_adj_pred,
                                  shape=(nb_nodes, nb_nodes, 1))
  sp_adj_train = tf.SparseTensor(
      indices=sp_adj_train.indices,
      values=tf.ones_like(sp_adj_train.values),
      dense_shape=sp_adj_train.dense_shape)
  sp_adj_train = tf.sparse_reshape(sp_adj_train,
                                   shape=(nb_nodes, nb_nodes, 1))
  sp_adj_pred = tf.sparse_concat(
      sp_inputs=[sp_adj_pred, sp_adj_train], axis=-1)
  return tf.sparse_reduce_max(sp_adj_pred, axis=-1)


@tf.custom_gradient
def mask_edges(scores, mask):
  masked_scores = tf.multiply(scores, mask)
  def grad(dy):
    return dy, None  # tf.multiply(scores, dy)
  return masked_scores, grad
