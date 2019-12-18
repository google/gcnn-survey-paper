# Graph Convolutional Neural Networks (GCNN) models

This repository contains a tensorflow implementation of GCNN models for node
classification, link predicition and joint node classification and link
prediction to supplement the survey paper by Chami et al.

NOTE: This is not an officially supported Google product.

## Code organization

* `train.py`: trains a model with FLAGS parameters. `train --helpshort` for more information.
.
* `launch.py`: trains several model with varied combinations of parameters. Specify parameters in `launch.py` file. `launch --helpshort` for more information.


* `best_model.py`: Parse the logs for multiple training with `launch.py` and finds best model parameters based on validation accuracy. `best_model --helpshort` for more information.

* `models/`
  * `base_models.py`: base model functionnalities (data utils, loss function, metrics etc)

  * `node_models.py`: forward pass implementation of node classification models (including Gat, Gcn, Mlp and SemiEmb)

  * `edge_models.py`: forward pass implementation of link prediction models (including Gae and Vgae)

  * `node_edge_models.py`: forward pass implementation of joint node classification and link prediction

* `utils/`

  * `model_utils.py`: layers implementation.

  * `link_prediction_utils.py`: implementation of some link prediction heuristics such as common neighbours or adamic adar

  * `data_utils.py`: data processing utils functions

  * `train_utils.py` train utils functions

* `data/`: contains data files for citation data (cora, citeseer, pubmed) and PPI

## Code usage

0. Install required libraries.

1. Set environment variables
  `GCNN_HOME=$(pwd)`
  `export PATH="$GCNN_HOME:$PATH"`

2. Put datasets the data folder.

3. Train GAT on cora with default parameters

  `SAVE_DIRECTORY="/tmp/models/cora/Gat"`
  `python train.py --save_dir=$SAVE_DIRECTORY --dataset=cora --model_name=Gat`

4. Check results

  `cat $SAVE_DIRECTORY/*.log`

  This model should give approximately 83% test accuracy.

5. Launch multiple experiments

  To launch multiple experiments for hyper-parameter search use the `launch.py` script. Update the parameters to search over in the `launch.py` file. For instance to train Gcn on cora with multiple parameters:

  `LAUNCH_DIR="/tmp/launch"`

  `python launch.py --launch_save_dir=$LAUNCH_DIR --launch_model_name=Gcn --launch_dataset=cora --launch_n_runs=3`

  This will create subdirectories `$LAUNCH_DIR/dataset_name/prop_edges_removed` where the log files will be saved.

6. Retrieve best model parameters

  `python best_model.py --dir=$LAUNCH_DIR --models=Gcn --target=node_acc --datasets=cora`

  This will create a `best_params` file in `$LAUNCH_DIR` with the best parameters for each (dataset-model-proportion_edges_dropped) combination based on validation metrics.

  `cat $LAUNCH_DIR/best_params`

## More examples

* Reproduce Gat results on cora (83.5% average test accuracy):

`python train.py --model_name=Gat --lr=0.005 --node_l2_reg=0.0005 --dataset=cora --p_drop_node=0.6 --n_att_node=8,1 --n_hidden_node=8 --save_dir=/tmp/models/cora/gat
--epochs=10000 --patience=100 --normalize_adj=False --sparse_features=True`

* Reproduce Gcn results on cora (81.5% average test accuracy):

`python train.py --model_name=Gcn --epochs=200 --patience=10 --lr=0.01 --node_l2_reg=0.0005
--dataset=cora --p_drop_node=0.5 --n_hidden_node=16
--save_dir=/tmp/models/cora/gcn --normalize_adj=True --sparse_features=True`

* Better Gcn results on cora (83.1% average test accuracy):

`python train.py --model_name=Gcn --epochs=10000 --patience=100 --lr=0.005 --node_l2_reg=0.0005
--dataset=cora --p_drop_node=0.6 --input_dim=1433 --n_hidden_node=128
--save_dir=/tmp/models/cora/gcn_best --normalize_adj=True --sparse_features=True`

* Train Gae on Cora with 10% of edges removed

`python train.py --model_name=Gae --epochs=10000 --patience=50 --lr=0.005 --p_drop_edge=0. --n_hidden_edge=256-128 --save_dir=/tmp/models/cora/Gae --edge_l2_reg=0 --att_mechanism=dot --normalize_adj=True --edge_loss=w_sigmoid_ce --dataset=cora --sparse_features=True --drop_edge_prop=10`

## Implementing a new model

To add a new model:

* Create a model class inheriting from one of the base class (NodeModel, EdgeModel or NodeEdgeModel) and implement the inference step in the correspoding file (`node_models.py`, `edge_models.py` or `node_edge_models.py`)

* Add the model name to the list of models in `train.py`

## Adding another dataset

To add another dataset:

* Write a `load_${dataset_str}_data()` function and add it to the load_data(dataset_str, data_path) function. the dataset_str will be the FLAG for this dataset.

* Save the data files in the `data/` folder.

## References

[GAT original code](https://github.com/PetarV-/GAT)

[GCN original code](https://github.com/tkipf/gcn/tree/master/gcn)

[GAE original code](https://github.com/tkipf/gae/blob/master/gae)
