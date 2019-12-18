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


"""Averages validation metric over multiple runs and returns best model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import numpy as np
import scipy.stats as stats
import tensorflow as tf

flags.DEFINE_string('dir', '/tmp/launch', 'path were models are saved.')
flags.DEFINE_string('target', 'node_acc', 'target metric to use.')
flags.DEFINE_string('datasets', 'cora', 'datasets to use.')
flags.DEFINE_string('drop_prop', '0-10-20-30-40-50-60-70-80-90',
                    'proportion of edges dropped')
flags.DEFINE_string('save_file', 'best_params', 'name of files to same the'
                    'results.')
flags.DEFINE_string('models', 'Gcn', 'name of model directories to parse.')
FLAGS = flags.FLAGS


def get_val_test_acc(data):
  """Parses log file to retrieve test and val accuracy."""
  data = [x.split() for x in data if len(x.split()) > 1]
  val_acc_idx = data[-4].index('val_{}'.format(FLAGS.target))
  test_acc_idx = data[-3].index('test_{}'.format(FLAGS.target))
  val_acc = data[-4][val_acc_idx + 2]
  test_acc = data[-3][test_acc_idx + 2]
  return float(val_acc) * 100, float(test_acc) * 100


def main(_):
  log_file = tf.gfile.Open(os.path.join(FLAGS.dir, FLAGS.save_file), 'w')
  for dataset in FLAGS.datasets.split('-'):
    for prop in FLAGS.drop_prop.split('-'):
      dir_path = os.path.join(FLAGS.dir, dataset, prop)
      if tf.gfile.IsDirectory(dir_path):
        print(dir_path)
        for model_name in tf.gfile.ListDirectory(dir_path):
          if model_name in FLAGS.models.split('-'):
            model_dir = os.path.join(dir_path, model_name)
            train_log_files = [
                filename for filename in tf.gfile.ListDirectory(model_dir)
                if 'log' in filename
            ]
            eval_stats = {}
            for filename in train_log_files:
              data = tf.gfile.Open(os.path.join(model_dir,
                                                filename)).readlines()
              nb_lines = len(data)
              if nb_lines > 0:
                if 'Training done' in data[-1]:
                  val_acc, test_acc = get_val_test_acc(data)
                  params = '-'.join(filename.split('-')[:-1])
                  if params in eval_stats:
                    eval_stats[params]['val'].append(val_acc)
                    eval_stats[params]['test'].append(test_acc)
                  else:
                    eval_stats[params] = {'val': [val_acc], 'test': [test_acc]}
            best_val_metric = -1
            best_params = None
            for params in eval_stats:
              val_metric = np.mean(eval_stats[params]['val'])
              if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_params = params
            # print(eval_stats)
            log_file.write('\n' + model_dir + '\n')
            log_file.write('Best params: {}\n'.format(best_params))
            log_file.write('val_{}: {} +- {}\n'.format(
                FLAGS.target, round(np.mean(eval_stats[best_params]['val']), 2),
                round(stats.sem(eval_stats[best_params]['val']), 2)))
            log_file.write('test_{}: {} +- {}\n'.format(
                FLAGS.target, round(
                    np.mean(eval_stats[best_params]['test']), 2),
                round(stats.sem(eval_stats[best_params]['test']), 2)))


if __name__ == '__main__':
  app.run(main)
