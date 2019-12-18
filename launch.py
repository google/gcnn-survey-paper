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


"""Train models with different combinations of parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product

import os
from absl import app
from absl import flags

from train import Config
from train import TrainTest

flags.DEFINE_string('launch_save_dir', '/tmp/launch',
                    'Where to save the results.')
flags.DEFINE_string('launch_model_name', 'Gcn', 'Model to train.')
flags.DEFINE_string('launch_dataset', 'cora', 'Dataset to use.')
flags.DEFINE_string('launch_datapath',
                    'data/',
                    'Path to data folder.')
flags.DEFINE_boolean('launch_sparse_features', True,
                     'True if node features are sparse.')
flags.DEFINE_boolean('launch_normalize_adj', True,
                     'True to normalize adjacency matrix')
flags.DEFINE_integer('launch_n_runs', 5,
                     'number of runs for each combination of parameters.')
FLAGS = flags.FLAGS


def get_params():
  ############################### CHANGE PARAMS HERE ##########################
  return {
      # training parameters
      'lr': [0.01],
      'epochs': [10000],
      'patience': [10],
      'node_l2_reg': [0.001, 0.0005],
      'edge_l2_reg': [0.],
      'edge_reg': [0],
      'p_drop_node': [0.5],
      'p_drop_edge': [0],

      # model parameters
      'n_hidden_node': ['128', '64'],
      'n_att_node': ['8-8'],
      'n_hidden_edge': ['128-64'],
      'n_att_edge': ['8-1'],
      'topk': [0],
      'att_mechanism': ['l2'],
      'edge_loss': ['w_sigmoid_ce'],
      'cheby_k_loc': [1],
      'semi_emb_k': [-1],

      # data parameters
      'drop_edge_prop': [0, 50],
      'normalize_adj': [True]
  }
  #############################################################################


def get_config(run_params, data):
  """Parse configuration parameters for training."""
  config = Config()
  for param in run_params:
    if 'n_hidden' in param or 'n_att' in param:
      # Number of layers and att are defined as string so we parse
      # them differently
      setattr(config, param, list(map(int, run_params[param].split('-'))))
    else:
      setattr(config, param, run_params[param])
  config.set_num_nodes_edges(data)
  return config


def main(_):
  params = get_params()
  trainer = TrainTest(FLAGS.launch_model_name)
  print('Loading dataset...')
  trainer.load_dataset(FLAGS.launch_dataset, FLAGS.launch_sparse_features,
                       FLAGS.launch_datapath)
  print('Dataset loaded!')
  # iterate over all combination of parameters
  all_params = product(*params.values())
  for run_params in all_params:
    run_params = dict(zip(params, run_params))
    # load the dataset and process adjacency and node features
    trainer.mask_edges(trainer.data['adj_true'], run_params['drop_edge_prop'])
    trainer.process_adj(FLAGS.launch_normalize_adj)
    config = get_config(run_params, trainer.data)
    # multilple runs
    save_dir = os.path.join(FLAGS.launch_save_dir, FLAGS.launch_dataset,
                            str(run_params['drop_edge_prop']),
                            FLAGS.launch_model_name)
    for run_id in range(FLAGS.launch_n_runs):
      filename_suffix = config.get_filename_suffix(run_id)
      trainer.run(config, save_dir, filename_suffix)


if __name__ == '__main__':
  app.run(main)
