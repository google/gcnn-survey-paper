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

"""Helper functions for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def format_metrics(metrics, mode):
  """Format metrics for logging."""
  result = ''
  for metric in metrics:
    result += '{}_{} = {:.4f} | '.format(mode, metric, float(metrics[metric]))
  return result


def format_params(config):
  """Format training parameters for logging."""
  result = ''
  for key, value in config.__dict__.items():
    result += '{}={} \n '.format(key, str(value))
  return result


def check_improve(best_metrics, metrics, targets):
  """Checks if any of the target metrics improved."""
  return [
      compare(metrics[target], best_metrics[target], targets[target])
      for target in targets
  ]


def compare(x1, x2, increasing):
  if increasing == 1:
    return x1 >= x2
  else:
    return x1 <= x2
