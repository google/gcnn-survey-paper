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


"""Inference step for joint node classification and link prediction models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from models.base_models import NodeEdgeModel
from models.edge_models import Gae
from models.node_models import Gat
from models.node_models import Gcn
import tensorflow as tf
from utils.model_utils import compute_adj
from utils.model_utils import gat_module
from utils.model_utils import gcn_module
from utils.model_utils import get_sp_topk
from utils.model_utils import mask_edges


class GaeGat(NodeEdgeModel):
  """GAE for link prediction and GAT for node classification."""

  def __init__(self, config):
    """Initializes EGCNGAT model."""
    super(GaeGat, self).__init__(config)
    self.edge_model = Gae(config)
    self.node_model = Gat(config)

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    adj_matrix_pred = self.edge_model.compute_inference(
        node_features_in, sp_adj_matrix, is_training)
    self.adj_matrix_pred = adj_matrix_pred
    adj_mask = get_sp_topk(adj_matrix_pred, sp_adj_matrix, self.nb_nodes,
                           self.topk)
    self.adj_mask = adj_mask
    # masked_adj_matrix_pred = tf.multiply(adj_mask,
    #                                      tf.nn.sigmoid(adj_matrix_pred))
    masked_adj_matrix_pred = mask_edges(tf.nn.sigmoid(adj_matrix_pred),
                                        adj_mask)
    sp_adj_pred = tf.contrib.layers.dense_to_sparse(masked_adj_matrix_pred)
    logits = self.node_model.compute_inference(node_features_in, sp_adj_pred,
                                               is_training)
    return logits, adj_matrix_pred


class GaeGcn(NodeEdgeModel):
  """GAE for link prediction and GCN for node classification."""

  def __init__(self, config):
    """Initializes EGCNGCN model."""
    super(GaeGcn, self).__init__(config)
    self.edge_model = Gae(config)
    self.node_model = Gcn(config)

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    adj_matrix_pred = self.edge_model.compute_inference(
        node_features_in, sp_adj_matrix, is_training)
    self.adj_matrix_pred = adj_matrix_pred
    adj_mask = get_sp_topk(adj_matrix_pred, sp_adj_matrix, self.nb_nodes,
                           self.topk)
    sp_adj_pred = tf.contrib.layers.dense_to_sparse(
        tf.multiply(adj_mask, tf.nn.leaky_relu(adj_matrix_pred)))
    sp_adj_pred = tf.sparse_softmax(sp_adj_pred)
    logits = self.node_model.compute_inference(node_features_in, sp_adj_pred,
                                               is_training)
    return logits, adj_matrix_pred


############################ EXPERIMENTAL MODELS #############################


class GatGraphite(NodeEdgeModel):
  """Gae for link prediction and GCN for node classification."""

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    with tf.variable_scope('edge-model'):
      z_latent = gat_module(
          node_features_in,
          sp_adj_matrix,
          self.n_hidden_edge,
          self.n_att_edge,
          self.p_drop_edge,
          is_training,
          self.input_dim,
          self.sparse_features,
          average_last=False)
      adj_matrix_pred = compute_adj(z_latent, self.att_mechanism,
                                    self.p_drop_edge, is_training)
      self.adj_matrix_pred = adj_matrix_pred
    with tf.variable_scope('node-model'):
      concat = True
      if concat:
        z_latent = tf.sparse_concat(
            axis=1,
            sp_inputs=[
                tf.contrib.layers.dense_to_sparse(z_latent), node_features_in
            ],
        )
        sparse_features = True
        input_dim = self.n_hidden_edge[-1] * self.n_att_edge[
            -1] + self.input_dim
      else:
        sparse_features = False
        input_dim = self.n_hidden_edge[-1] * self.n_att_edge[-1]
      logits = gat_module(
          z_latent,
          sp_adj_matrix,
          self.n_hidden_node,
          self.n_att_node,
          self.p_drop_node,
          is_training,
          input_dim,
          sparse_features=sparse_features,
          average_last=False)

    return logits, adj_matrix_pred


class GaeGatConcat(NodeEdgeModel):
  """EGCN for link prediction and GCN for node classification."""

  def __init__(self, config):
    """Initializes EGCN_GAT model."""
    super(GaeGatConcat, self).__init__(config)
    self.edge_model = Gae(config)
    self.node_model = Gat(config)

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    with tf.variable_scope('edge-model'):
      z_latent = gcn_module(node_features_in, sp_adj_matrix, self.n_hidden_edge,
                            self.p_drop_edge, is_training, self.input_dim,
                            self.sparse_features)
      adj_matrix_pred = compute_adj(z_latent, self.att_mechanism,
                                    self.p_drop_edge, is_training)
      self.adj_matrix_pred = adj_matrix_pred
    with tf.variable_scope('node-model'):
      z_latent = tf.sparse_concat(
          axis=1,
          sp_inputs=[
              tf.contrib.layers.dense_to_sparse(z_latent), node_features_in
          ])
      sparse_features = True
      input_dim = self.n_hidden_edge[-1] + self.input_dim
      sp_adj_train = tf.SparseTensor(
          indices=sp_adj_matrix.indices,
          values=tf.ones_like(sp_adj_matrix.values),
          dense_shape=sp_adj_matrix.dense_shape)
      logits = gat_module(
          z_latent,
          sp_adj_train,
          self.n_hidden_node,
          self.n_att_node,
          self.p_drop_node,
          is_training,
          input_dim,
          sparse_features=sparse_features,
          average_last=True)
    return logits, adj_matrix_pred


class GaeGcnConcat(NodeEdgeModel):
  """EGCN for link prediction and GCN for node classification."""

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    with tf.variable_scope('edge-model'):
      z_latent = gcn_module(node_features_in, sp_adj_matrix, self.n_hidden_edge,
                            self.p_drop_edge, is_training, self.input_dim,
                            self.sparse_features)
      adj_matrix_pred = compute_adj(z_latent, self.att_mechanism,
                                    self.p_drop_edge, is_training)
      self.adj_matrix_pred = adj_matrix_pred
    with tf.variable_scope('node-model'):
      z_latent = tf.sparse_concat(
          axis=1,
          sp_inputs=[
              tf.contrib.layers.dense_to_sparse(z_latent), node_features_in
          ])
      sparse_features = True
      input_dim = self.n_hidden_edge[-1] + self.input_dim
      logits = gcn_module(
          z_latent,
          sp_adj_matrix,
          self.n_hidden_node,
          self.p_drop_node,
          is_training,
          input_dim,
          sparse_features=sparse_features)
    return logits, adj_matrix_pred


class Gcat(NodeEdgeModel):
  """1 iteration Graph Convolution Attention Model."""

  def __init__(self, config):
    """Initializes GCAT model."""
    super(Gcat, self).__init__(config)
    self.edge_model = Gae(config)
    self.node_model = Gcn(config)

  def compute_inference(self, node_features_in, sp_adj_matrix, is_training):
    """Forward pass for GAT model."""
    adj_matrix_pred = self.edge_model.compute_inference(
        node_features_in, sp_adj_matrix, is_training)
    sp_adj_mask = tf.SparseTensor(
        indices=sp_adj_matrix.indices,
        values=tf.ones_like(sp_adj_matrix.values),
        dense_shape=sp_adj_matrix.dense_shape)
    sp_adj_att = sp_adj_mask * adj_matrix_pred
    sp_adj_att = tf.SparseTensor(
        indices=sp_adj_att.indices,
        values=tf.nn.leaky_relu(sp_adj_att.values),
        dense_shape=sp_adj_att.dense_shape)
    sp_adj_att = tf.sparse_softmax(sp_adj_att)
    logits = self.node_model.compute_inference(node_features_in, sp_adj_att,
                                               is_training)
    return logits, adj_matrix_pred
