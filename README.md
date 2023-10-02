Various utilities for using GNNs with Hydra

## Installation

Make a virtual environment and install `requirements.txt` for a minimal set of functionality. Some code may require the hydra python bindings (see [here](https://github.mit.edu:SPARK/hydra_python) for details).

The minimal requirement to run training code with pickled dataset file is installing [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) with [pyg_lib](https://github.com/pyg-team/pyg-lib) for Heterogeneous GNN operators, and the following dependencies:
```
pip install pandas plotly networkx tensorboardX pyyaml
```
This code is tested on PyTorch 1.12.1, PyTorch Geometric 2.2.0 with Cuda 11.3, and PyTorch 1.8.1, PyTorch Geometric 2.0.4 with Cuda 10.2.

## H-tree Construction
by Rajat Talak

### Dataset

Put the dataset and pre-trained word2vec model in the folder ./data. It is organized as follows:

- data
  - house_files
  - mp3d_old
    - hydra_mp3d_dataset
    - hydra_mp3d_non_gt_dataset
    - mp3d_with_agents
  - tro_graphs_2022_09_24
  - [GoogleNews-vectors-negative300.bin](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)


### Dataset nomenclature

File ./datasets/mp3d.py contains the MP3D dataset code.
MP3D(complete=True) will generate a dataset of full MP3D scenes.
MP3D(complete=False) will generate a dataset of MP3D trajectory scenes.


### Constructing H-tree

```bash
cd neural_tree

python -W ignore construct.py
```

This will generate H-tree for all MP3D scenes.
By default, it will use the MP3D(complete=True) dataset. This can be changed in the construct.py code.
