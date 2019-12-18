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

"""Base models class.

Main functionnalities for node classification models, link prediction
models and joint node classification and link prediction models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BaseModel(object):
  """Base model class. Defines basic functionnalities for all models."""

  def __init__(self, config):
    """Initialize base model.

    Args:
      config: object of Config class defined in train.py,
              stores configuration parameters to build and train the model
    """
    self.input_dim = config.input_dim
    self.lr = config.lr
    self.edge_reg = config.edge_reg
    self.edge_l2_reg = config.edge_l2_reg
    self.node_l2_reg = config.node_l2_reg
    self.nb_nodes = config.nb_nodes
    self.nb_edges = config.nb_edges
    self.sparse_features = config.sparse_features
    self.edge_loss = config.edge_loss
    self.att_mechanism = config.att_mechanism
    self.multilabel = config.multilabel

  def _create_placeholders(self):
    raise NotImplementedError

  def compute_inference(self, features, adj_matrix, is_training):
    raise NotImplementedError

  def build_graph(self):
    raise NotImplementedError

  def _create_optimizer(self, loss):
    """Create train operation."""
    opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    train_op = opt.minimize(loss)
    return train_op

  def _compute_node_loss(self, logits, labels):
    """Node classification loss with sigmoid cross entropy."""
    if self.multilabel:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
    else:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    return tf.reduce_mean(loss)

  def _compute_node_l2_loss(self):
    """L2 regularization loss for parameters in node classification model."""
    all_variables = tf.trainable_variables()
    non_reg = ['bias', 'embeddings', 'beta', 'edge-model']
    node_l2_loss = tf.add_n([
        tf.nn.l2_loss(v)
        for v in all_variables
        if all([var_name not in v.name for var_name in non_reg])
    ])
    return node_l2_loss

  def _compute_edge_l2_loss(self):
    """L2 regularization loss for parameters in link prediction model."""
    all_variables = tf.trainable_variables()
    edge_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_variables if \
                             'edge-model' in v.name])
    return edge_l2_loss

  def _compute_edge_loss_neg_sampling(self, adj_pred, adj_true):
    """Link prediction CE loss with negative sampling."""
    keep_prob = self.nb_edges / (self.nb_nodes**2 - self.nb_edges)
    loss_mask = tf.nn.dropout(
        1 - adj_true, keep_prob=keep_prob) * keep_prob
    loss_mask += adj_true
    boolean_mask = tf.greater(loss_mask, 0.)
    masked_pred = tf.boolean_mask(adj_pred, boolean_mask)
    masked_true = tf.boolean_mask(adj_true, boolean_mask)
    edge_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=masked_true,
        logits=masked_pred,
    )
    return tf.reduce_mean(edge_loss)

  def _compute_edge_loss_weighted_ce(self, adj_pred, adj_true):
    """Link prediction loss with weighted sigmoid cross entropy."""
    pos_weight = float((self.nb_nodes**2) - self.nb_edges) / self.nb_edges
    edge_loss = tf.nn.weighted_cross_entropy_with_logits(
        targets=adj_true,
        logits=adj_pred,
        pos_weight=pos_weight)
    return tf.reduce_mean(edge_loss)

  def _compute_edge_loss(self, adj_pred, adj_true):
    if self.edge_loss == 'weighted':
      return self._compute_edge_loss_weighted_ce(adj_pred, adj_true)
    else:
      return self._compute_edge_loss_neg_sampling(adj_pred, adj_true)


class NodeModel(BaseModel):
  """Base model class for semi-supevised node classification."""

  def __init__(self, config):
    """Initializes NodeModel for semi-supervised node classification.

    Args:
      config: object of Config class defined in train.py,
              stores configuration parameters to build and train the model
    """
    super(NodeModel, self).__init__(config)
    self.p_drop = config.p_drop_node
    self.n_att = config.n_att_node
    self.n_hidden = config.n_hidden_node

  def _create_placeholders(self):
    """Create placeholders."""
    with tf.name_scope('input'):
      self.placeholders = {
          'adj_train':
              tf.sparse_placeholder(tf.float32),  # normalized
          'node_labels':
              tf.placeholder(tf.float32, shape=[None, self.n_hidden[-1]]),
          'node_mask':
              tf.placeholder(tf.float32, shape=[
                  None,
              ]),
          'is_training':
              tf.placeholder(tf.bool),
      }
      if self.sparse_features:
        self.placeholders['features'] = tf.sparse_placeholder(tf.float32)
      else:
        self.placeholders['features'] = tf.placeholder(
            tf.float32, shape=[None, self.input_dim])

  def make_feed_dict(self, data, split, is_training):
    """Build feed dictionnary to train the model."""
    feed_dict = {
        self.placeholders['adj_train']: data['adj_train_norm'],
        self.placeholders['features']: data['features'],
        self.placeholders['node_labels']: data['node_labels'],
        self.placeholders['node_mask']: data[split]['node_mask'],
        self.placeholders['is_training']: is_training
    }
    return feed_dict

  def build_graph(self):
    """Build tensorflow graph and create training, testing ops."""
    self._create_placeholders()
    logits = self.compute_inference(self.placeholders['features'],
                                    self.placeholders['adj_train'],
                                    self.placeholders['is_training'])
    boolean_mask = tf.greater(self.placeholders['node_mask'], 0.)
    masked_pred = tf.boolean_mask(logits, boolean_mask)
    masked_true = tf.boolean_mask(self.placeholders['node_labels'],
                                  boolean_mask)
    loss = self._compute_node_loss(masked_pred, masked_true)
    loss += self.node_l2_reg * self._compute_node_l2_loss()
    train_op = self._create_optimizer(loss)
    metric_op, metric_update_op = self._create_metrics(
        masked_pred, masked_true)
    return loss, train_op, metric_op, metric_update_op

  def _create_metrics(self, logits, node_labels):
    """Create evaluation metrics for node classification."""
    with tf.name_scope('metrics'):
      metrics = {}
      if self.multilabel:
        predictions = tf.cast(
            tf.greater(tf.nn.sigmoid(logits), 0.5), tf.float32)
        metrics['recall'], rec_op = tf.metrics.recall(
            labels=node_labels, predictions=predictions)
        metrics['precision'], prec_op = tf.metrics.precision(
            labels=node_labels, predictions=predictions)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
            )
        update_ops = [rec_op, prec_op]
      else:
        metrics['node_acc'], acc_op = tf.metrics.accuracy(
            labels=tf.argmax(node_labels, 1), predictions=tf.argmax(logits, 1))
        update_ops = [acc_op]
    return metrics, update_ops


class EdgeModel(BaseModel):
  """Base model class for link prediction."""

  def __init__(self, config):
    """Initializes Edge model for link prediction.

    Args:
      config: object of Config class defined in train.py,
              stores configuration parameters to build and train the model
    """
    super(EdgeModel, self).__init__(config)
    self.p_drop = config.p_drop_edge
    self.n_att = config.n_att_edge
    self.n_hidden = config.n_hidden_edge

  def _create_placeholders(self):
    """Create placeholders."""
    with tf.name_scope('input'):
      self.placeholders = {
          # to compute metrics
          'adj_true': tf.placeholder(tf.float32, shape=[None, None]),
          # to compute loss
          'adj_train': tf.placeholder(tf.float32, shape=[None, None]),
          # for inference step
          'adj_train_norm': tf.sparse_placeholder(tf.float32),  # normalized
          'edge_mask': tf.sparse_placeholder(tf.float32),
          'is_training': tf.placeholder(tf.bool),
      }
      if self.sparse_features:
        self.placeholders['features'] = tf.sparse_placeholder(tf.float32)
      else:
        self.placeholders['features'] = tf.placeholder(
            tf.float32, shape=[None, self.input_dim])

  def make_feed_dict(self, data, split, is_training):
    """Build feed dictionnary to train the model."""
    feed_dict = {
        self.placeholders['features']: data['features'],
        self.placeholders['adj_true']: data['adj_true'],
        self.placeholders['adj_train']: data['adj_train'],
        self.placeholders['adj_train_norm']: data['adj_train_norm'],
        self.placeholders['edge_mask']: data[split]['edge_mask'],
        self.placeholders['is_training']: is_training
    }
    return feed_dict

  def build_graph(self):
    """Build tensorflow graph and create training, testing ops."""
    self._create_placeholders()
    adj_pred = self.compute_inference(self.placeholders['features'],
                                      self.placeholders['adj_train_norm'],
                                      self.placeholders['is_training'])
    adj_train = tf.reshape(self.placeholders['adj_train'], (-1,))
    loss = self._compute_edge_loss(tf.reshape(adj_pred, (-1,)), adj_train)
    loss += self.edge_l2_reg * self._compute_edge_l2_loss()
    train_op = self._create_optimizer(loss)
    masked_true = tf.reshape(tf.gather_nd(
        self.placeholders['adj_true'], self.placeholders['edge_mask'].indices),
                             (-1,))
    masked_pred = tf.reshape(tf.gather_nd(
        adj_pred, self.placeholders['edge_mask'].indices), (-1,))
    metric_op, metric_update_op = self._create_metrics(masked_pred, masked_true)
    return loss, train_op, metric_op, metric_update_op

  def _create_metrics(self, adj_pred, adj_true):
    """Create evaluation metrics for node classification."""
    with tf.name_scope('metrics'):
      metrics = {}
      metrics['edge_roc_auc'], roc_op = tf.metrics.auc(
          labels=adj_true,
          predictions=tf.sigmoid(adj_pred),
          curve='ROC'
      )
      metrics['edge_pr_auc'], pr_op = tf.metrics.auc(
          labels=adj_true,
          predictions=tf.sigmoid(adj_pred),
          curve='PR'
      )
      update_ops = [roc_op, pr_op]
    return metrics, update_ops


class NodeEdgeModel(BaseModel):
  """Model class for semi-supevised node classification and link prediction."""

  def __init__(self, config):
    """Initializes model.

    Args:
      config: object of Config class defined in train.py,
              stores configuration parameters to build and train the model
    """
    super(NodeEdgeModel, self).__init__(config)
    self.n_att_edge = config.n_att_edge
    self.n_hidden_edge = config.n_hidden_edge
    self.p_drop_edge = config.p_drop_edge
    self.n_att_node = config.n_att_node
    self.n_hidden_node = config.n_hidden_node
    self.p_drop_node = config.p_drop_node
    self.topk = config.topk

  def _create_placeholders(self):
    """Create placeholders."""
    with tf.name_scope('input'):
      self.placeholders = {
          'adj_true': tf.placeholder(tf.float32, shape=[None, None]),
          # to compute loss
          'adj_train': tf.placeholder(tf.float32, shape=[None, None]),
          # for inference step
          'adj_train_norm': tf.sparse_placeholder(tf.float32),  # normalized
          'edge_mask': tf.sparse_placeholder(tf.float32),
          'node_labels':
              tf.placeholder(tf.float32, shape=[None, self.n_hidden_node[-1]]),
          'node_mask':
              tf.placeholder(tf.float32, shape=[
                  None,
              ]),
          'is_training':
              tf.placeholder(tf.bool),
      }
      if self.sparse_features:
        self.placeholders['features'] = tf.sparse_placeholder(tf.float32)
      else:
        self.placeholders['features'] = tf.placeholder(
            tf.float32, shape=[None, self.input_dim])

  def make_feed_dict(self, data, split, is_training):
    """Build feed dictionnary to train the model."""
    feed_dict = {
        self.placeholders['features']: data['features'],
        self.placeholders['adj_true']: data['adj_true'],
        self.placeholders['adj_train']: data['adj_train'],
        self.placeholders['adj_train_norm']: data['adj_train_norm'],
        self.placeholders['edge_mask']: data[split]['edge_mask'],
        self.placeholders['node_labels']: data['node_labels'],
        self.placeholders['node_mask']: data[split]['node_mask'],
        self.placeholders['is_training']: is_training
    }
    return feed_dict

  def build_graph(self):
    """Build tensorflow graph and create training, testing ops."""
    self._create_placeholders()
    logits, adj_pred = self.compute_inference(
        self.placeholders['features'],
        self.placeholders['adj_train_norm'],
        self.placeholders['is_training'])
    adj_train = tf.reshape(self.placeholders['adj_train'], (-1,))
    boolean_node_mask = tf.greater(self.placeholders['node_mask'], 0.)
    masked_node_pred = tf.boolean_mask(logits, boolean_node_mask)
    masked_node_true = tf.boolean_mask(self.placeholders['node_labels'],
                                       boolean_node_mask)
    loss = self._compute_node_loss(masked_node_pred,
                                   masked_node_true)
    loss += self.node_l2_reg * self._compute_node_l2_loss()
    loss += self.edge_reg * self._compute_edge_loss(
        tf.reshape(adj_pred, (-1,)), adj_train)
    loss += self.edge_l2_reg * self._compute_edge_l2_loss()
    self.grad = tf.gradients(loss, self.adj_matrix_pred)
    train_op = self._create_optimizer(loss)
    masked_adj_true = tf.reshape(tf.gather_nd(
        self.placeholders['adj_true'],
        self.placeholders['edge_mask'].indices), (-1,))
    masked_adj_pred = tf.reshape(tf.gather_nd(
        adj_pred, self.placeholders['edge_mask'].indices), (-1,))
    metric_op, metric_update_op = self._create_metrics(
        masked_adj_pred, masked_adj_true, masked_node_pred, masked_node_true)
    return loss, train_op, metric_op, metric_update_op

  def _create_metrics(self, adj_pred, adj_true, node_pred, node_labels):
    """Create evaluation metrics for node classification."""
    with tf.name_scope('metrics'):
      metrics = {}
      metrics['edge_roc_auc'], roc_op = tf.metrics.auc(
          labels=adj_true,
          predictions=tf.sigmoid(adj_pred),
          curve='ROC'
      )
      metrics['edge_pr_auc'], pr_op = tf.metrics.auc(
          labels=adj_true,
          predictions=tf.sigmoid(adj_pred),
          curve='PR'
      )
      metrics['node_acc'], acc_op = tf.metrics.accuracy(
          labels=tf.argmax(node_labels, 1),
          predictions=tf.argmax(node_pred, 1))
      update_ops = [roc_op, pr_op, acc_op]
    return metrics, update_ops
