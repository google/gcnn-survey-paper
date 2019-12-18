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


"""Training script for GNN models for link prediction/node classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags

import models.edge_models as edge_models
import models.node_edge_models as node_edge_models
import models.node_models as node_models

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from utils.data_utils import load_data
from utils.data_utils import mask_test_edges
from utils.data_utils import mask_val_test_edges
from utils.data_utils import process_adj
from utils.data_utils import sparse_to_tuple
from utils.train_utils import check_improve
from utils.train_utils import format_metrics
from utils.train_utils import format_params

flags.DEFINE_string('model_name', 'Gat', 'Which model to use.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train for.')
flags.DEFINE_integer('patience', 100, 'Patience for early stopping.')
flags.DEFINE_string('dataset', 'cora',
                    'Dataset to use: (cora - citeseer - pubmed).')
flags.DEFINE_string('datapath', 'data/',
                    'Path to directory with data files.')
flags.DEFINE_string('save_dir', '/tmp/models/cora/gat',
                    'Directory where to save model checkpoints and summaries.')
flags.DEFINE_float('lr', 0.005, 'Learning rate to use.')
flags.DEFINE_string(
    'model_checkpoint', '', 'Model checkpoint to load before'
    'training or for testing. If not specified the model will be trained from '
    'scratch.')
flags.DEFINE_float('drop_edge_prop', 0,
                   'Percentage of edges to remove (0 to keep all edges).')
flags.DEFINE_float('node_l2_reg', 0.0005, 'L2 regularization to use for node '
                   'model parameters.')
flags.DEFINE_float('edge_l2_reg', 0., 'L2 regularization to use for edge '
                   'model parameters.')
flags.DEFINE_float('edge_reg', 0., 'Regularization to use for the edge '
                   'loss.')
flags.DEFINE_integer(
    'cheby_k_loc', 1, 'K for K-localized filters in Chebyshev'
    'polynomials approximation.')
flags.DEFINE_integer(
    'semi_emb_k', -1, 'which layer to regularize for'
    'semi-supervised embedding model.')
flags.DEFINE_float('p_drop_node', 0.6, 'Dropout probability for node model.')
flags.DEFINE_float('p_drop_edge', 0., 'Dopout probability for edge model.')
flags.DEFINE_integer(
    'topk', 1000, 'Top k entries to keep in adjacency for'
    ' NodeEdge models.')
flags.DEFINE_string(
    'n_hidden_node', '8', 'Number of hidden units per layer in node model. '
    'The last layer has as many nodes as the number of classes '
    'in the dataset.')
flags.DEFINE_string(
    'n_att_node', '8-1',
    'Number of attentions heads per layer in for node model. '
    '(This is only for graph attention models).')
flags.DEFINE_string('n_hidden_edge', '32-16',
                    'Number of hidden units per layer in edge model.')
flags.DEFINE_string(
    'n_att_edge', '8-4',
    'Number of attentions heads per layer in for edge model. '
    '(This is only for edge graph attention models)')
flags.DEFINE_string(
    'att_mechanism', 'dot',
    'Attention mehcanism to use: dot product, asymmetric dot '
    'product or attention (dot - att - asym-dot).')
flags.DEFINE_string(
    'edge_loss', 'w_sigmoid_ce', 'edge loss (w_sigmoid_ce - neg_sampling_ce). '
    'w_sigmoid_ce for weighted sigmoid cross entropy and neg_sampling_ce for'
    'negative sampling.')
flags.DEFINE_boolean('sparse_features', True,
                     'True if node features are sparse.')
flags.DEFINE_boolean(
    'normalize_adj', True, 'Whether to normalize adjaceny or not (True for'
    'GCN models and False for GAT models).')
flags.DEFINE_integer('run_id', 0, 'Run id.')

FLAGS = flags.FLAGS
NODE_MODELS = ['Gat', 'Gcn', 'Mlp', 'Hgat', 'Pgcn', 'SemiEmb', 'Cheby']
NODE_EDGE_MODELS = [
    'GaeGat', 'GaeGcn', 'GatGraphite', 'GaeGatConcat', 'GaeGcnConcat', 'Gcat'
]
EDGE_MODELS = ['Gae', 'Egat', 'Emlp', 'Vgae']


class Config(object):
  """Gets config parameters from flags to train the GNN models."""

  def __init__(self):
    # Model parameters
    self.n_hidden_node = list(map(int, FLAGS.n_hidden_node.split('-')))
    self.n_att_node = list(map(int, FLAGS.n_att_node.split('-')))
    self.n_hidden_edge = list(map(int, FLAGS.n_hidden_edge.split('-')))
    self.n_att_edge = list(map(int, FLAGS.n_att_edge.split('-')))
    self.topk = FLAGS.topk
    self.att_mechanism = FLAGS.att_mechanism
    self.edge_loss = FLAGS.edge_loss
    self.cheby_k_loc = FLAGS.cheby_k_loc
    self.semi_emb_k = FLAGS.semi_emb_k

    # Dataset parameters
    self.sparse_features = FLAGS.sparse_features

    # Training parameters
    self.lr = FLAGS.lr
    self.epochs = FLAGS.epochs
    self.patience = FLAGS.patience
    self.node_l2_reg = FLAGS.node_l2_reg
    self.edge_l2_reg = FLAGS.edge_l2_reg
    self.edge_reg = FLAGS.edge_reg
    self.p_drop_node = FLAGS.p_drop_node
    self.p_drop_edge = FLAGS.p_drop_edge

  def set_num_nodes_edges(self, data):
    if self.sparse_features:
      self.nb_nodes, self.input_dim = data['features'][-1]
    else:
      self.nb_nodes, self.input_dim = data['features'].shape
    self.nb_classes = data['node_labels'].shape[-1]
    self.n_hidden_node += [int(self.nb_classes)]
    self.nb_edges = np.sum(data['adj_train'] > 0) - self.nb_nodes
    self.multilabel = np.max(np.sum(data['node_labels'], 1)) > 1

  def get_filename_suffix(self, run_id):
    """Formats all params in a string for log file suffix."""
    all_params = [
        self.lr, self.epochs, self.patience, self.node_l2_reg, self.edge_l2_reg,
        self.edge_reg, self.p_drop_node, self.p_drop_edge, '.'.join([
            str(x) for x in self.n_hidden_node
        ]), '.'.join([str(x) for x in self.n_att_node]),
        '.'.join([str(x) for x in self.n_hidden_edge]), '.'.join(
            [str(x) for x in self.n_att_edge]), self.topk, self.att_mechanism,
        self.edge_loss, self.cheby_k_loc, self.semi_emb_k, run_id
    ]
    file_suffix = '-'.join([str(x) for x in all_params])
    return file_suffix


class TrainTest(object):
  """Class to train node and edge classification models"""

  def __init__(self, model_name):
    # initialize global step
    self.global_step = 0
    self.model_name = model_name
    self.data = {'train': {}, 'test': {}, 'val': {}}

  def load_dataset(self, dataset, sparse_features, datapath):
    """Loads citation dataset."""
    dataset = load_data(dataset, datapath)
    adj_true = dataset[0] + sp.eye(dataset[0].shape[0])
    # adj_true to compute link prediction metrics
    self.data['adj_true'] = adj_true.todense()
    if sparse_features:
      self.data['features'] = sparse_to_tuple(dataset[1])
    else:
      self.data['features'] = dataset[1]
    self.data['node_labels'] = dataset[2]
    self.data['train']['node_mask'] = dataset[3]
    self.data['val']['node_mask'] = dataset[4]
    self.data['test']['node_mask'] = dataset[5]

  def mask_edges(self, adj_true, drop_edge_prop):
    """Load edge mask and remove edges for training adjacency."""
    # adj_train to compute loss
    if drop_edge_prop > 0:
      if self.model_name in NODE_MODELS:
        self.data['adj_train'], test_mask = mask_test_edges(
            sp.coo_matrix(adj_true), drop_edge_prop * 0.01)
      else:
        self.data['adj_train'], val_mask, test_mask = mask_val_test_edges(
            sp.coo_matrix(adj_true), drop_edge_prop * 0.01)
        self.data['val']['edge_mask'] = val_mask
        self.data['train']['edge_mask'] = val_mask  # unused
        self.data['test']['edge_mask'] = test_mask
      self.data['adj_train'] += sp.eye(adj_true.shape[0])
      self.data['adj_train'] = self.data['adj_train'].todense()
    else:
      self.data['adj_train'] = adj_true

  def process_adj(self, norm_adj):
    # adj_train_norm for inference
    if norm_adj:
      adj_train_norm = process_adj(self.data['adj_train'], self.model_name)
    else:
      adj_train_norm = sp.coo_matrix(self.data['adj_train'])
    self.data['adj_train_norm'] = sparse_to_tuple(adj_train_norm)

  def init_global_step(self):
    self.global_step = 0

  def create_saver(self, save_dir, filename_suffix):
    """Creates saver to save model checkpoints."""
    self.summary_writer = tf.summary.FileWriter(
        save_dir, tf.get_default_graph(), filename_suffix=filename_suffix)
    self.saver = tf.train.Saver()
    # logging file to print metrics and loss
    self.log_file = tf.gfile.Open(
        os.path.join(save_dir, '{}.log'.format(filename_suffix)), 'w')

  def _create_summary(self, loss, metrics, split):
    """Create summaries for tensorboard."""
    with tf.name_scope('{}-summary'.format(split)):
      tf.summary.scalar('loss', loss)
      for metric in metrics:
        tf.summary.scalar(metric, metrics[metric])
      summary_op = tf.summary.merge_all()
    return summary_op

  def _make_feed_dict(self, split):
    """Creates feed dictionnaries for edge models and node models."""
    if split == 'train':
      is_training = True
    else:
      is_training = False
    return self.model.make_feed_dict(self.data, split, is_training)

  def _get_model_and_targets(self, multilabel):
    """Define targets to select best model based on val metrics."""
    if self.model_name in NODE_MODELS:
      model_class = getattr(node_models, self.model_name)
      if multilabel:
        target_metrics = {'f1': 1, 'loss': 0}
      else:
        target_metrics = {'node_acc': 1, 'loss': 0}
      # target_metrics = {'node_acc': 1}
    elif self.model_name in NODE_EDGE_MODELS:
      model_class = getattr(node_edge_models, self.model_name)
      target_metrics = {'node_acc': 1}
    else:
      model_class = getattr(edge_models, self.model_name)
      target_metrics = {'edge_pr_auc': 1}  #, 'loss': 0}
    return model_class, target_metrics

  def build_model(self, config):
    """Build model graph."""
    model_class, self.target_metrics = self._get_model_and_targets(
        config.multilabel)
    self.model = model_class(config)
    all_ops = self.model.build_graph()
    loss, train_op, metric_op, metric_update_op = all_ops
    self.train_ops = [train_op]
    self.eval_ops = [loss, metric_update_op]
    self.metrics = metric_op
    self.train_summary = self._create_summary(loss, metric_op, 'train')
    self.val_summary = self._create_summary(loss, metric_op, 'val')

  def _eval_model(self, sess, split):
    """Evaluates model."""
    sess.run(tf.local_variables_initializer())
    if split == 'train':
      metrics = {}
      # tmp way to not eval on train for edge model
      metrics['loss'] = sess.run(
          self.eval_ops[0], feed_dict=self._make_feed_dict(split))
    else:
      loss, _ = sess.run(self.eval_ops, feed_dict=self._make_feed_dict(split))
      metrics = sess.run(self.metrics, feed_dict=self._make_feed_dict(split))
      metrics['loss'] = loss
    return metrics

  def _init_best_metrics(self):
    best_metrics = {}
    for metric in self.target_metrics:
      if self.target_metrics[metric] == 1:
        best_metrics[metric] = -1
      else:
        best_metrics[metric] = np.inf
    return best_metrics

  def _log(self, message):
    """Writes into train.log file."""
    time = datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')
    self.log_file.write(time + ' : ' + message + '\n')

  def init_model_weights(self, sess):
    self._log('Initializing model weights...')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

  def restore_checkpoint(self, sess, model_checkpoint=None):
    """Loads model checkpoint if found and computes evaluation metrics."""
    if model_checkpoint is None or not tf.train.checkpoint_exists(
        model_checkpoint):
      self.init_model_weights(sess)
    else:
      self._log('Loading existing model saved at {}'.format(model_checkpoint))
      self.saver.restore(sess, model_checkpoint)
      self.global_step = int(model_checkpoint.split('-')[-1])
      val_metrics = self._eval_model(sess, 'val')
      test_metrics = self._eval_model(sess, 'test')
      self._log(format_metrics(val_metrics, 'val'))
      self._log(format_metrics(test_metrics, 'test'))

  def train(self, sess, config):
    """Trains node classification model or joint node edge model."""
    self._log('Training {} model...'.format(self.model_name))
    self._log('Training parameters : \n ' + format_params(config))
    epochs = config.epochs
    lr = config.lr
    patience = config.patience
    # best_step = self.global_step
    # step for patience
    curr_step = 0
    # best metrics to select model
    best_val_metrics = self._init_best_metrics()
    best_test_metrics = self._init_best_metrics()
    # train the model
    for epoch in range(epochs):
      self.global_step += 1
      sess.run(self.train_ops, feed_dict=self._make_feed_dict('train'))
      train_metrics = self._eval_model(sess, 'train')
      val_metrics = self._eval_model(sess, 'val')
      self._log('Epoch {} : lr = {:.4f} | '.format(epoch, lr) +
                format_metrics(train_metrics, 'train') +
                format_metrics(val_metrics, 'val'))
      # write summaries
      train_summary = sess.run(self.train_summary,
                               self._make_feed_dict('train'))
      val_summary = sess.run(self.val_summary, self._make_feed_dict('val'))
      self.summary_writer.add_summary(
          train_summary, global_step=self.global_step)
      self.summary_writer.add_summary(val_summary, global_step=self.global_step)
      # save model checkpoint if val acc increased and val loss decreased
      comp = check_improve(best_val_metrics, val_metrics, self.target_metrics)
      if np.any(comp):
        if np.all(comp):
          # best_step = self.global_step
          # save_path = os.path.join(save_dir, 'model')
          # self.saver.save(sess, save_path, global_step=self.global_step)
          best_test_metrics = self._eval_model(sess, 'test')
        best_val_metrics = val_metrics
        curr_step = 0
      else:
        curr_step += 1
        if curr_step == patience:
          self._log('Early stopping')
          break

    self._log('\n' + '*' * 40 + ' Best model metrics ' + '*' * 40)
    # load best model to evaluate on test set
    # save_path = os.path.join(save_dir, 'model-{}'.format(best_step))
    # self.restore_checkpoint(sess, save_path)
    self._log(format_metrics(best_val_metrics, 'val'))
    self._log(format_metrics(best_test_metrics, 'test'))
    self._log('\n' + '*' * 40 + ' Training done ' + '*' * 40)

  def run(self, config, save_dir, file_prefix):
    """Build and train a model."""
    tf.reset_default_graph()
    self.init_global_step()
    # build model
    self.build_model(config)
    # create summary writer and save for model weights
    if not os.path.exists(save_dir):
      tf.gfile.MakeDirs(save_dir)
    self.create_saver(save_dir, file_prefix)
    # run sessions
    with tf.Session() as sess:
      self.init_model_weights(sess)
      self.train(sess, config)
    sess.close()
    self.log_file.close()


def main(_):
  # parse configuration parameters
  trainer = TrainTest(FLAGS.model_name)
  print('Loading dataset...')
  # load the dataset and process adjacency and node features
  trainer.load_dataset(FLAGS.dataset, FLAGS.sparse_features, FLAGS.datapath)
  trainer.mask_edges(trainer.data['adj_true'], FLAGS.drop_edge_prop)
  trainer.process_adj(FLAGS.normalize_adj)
  print('Dataset loaded...')
  config = Config()
  config.set_num_nodes_edges(trainer.data)
  filename_suffix = config.get_filename_suffix(FLAGS.run_id)
  trainer.run(config, FLAGS.save_dir, filename_suffix)


if __name__ == '__main__':
  app.run(main)
