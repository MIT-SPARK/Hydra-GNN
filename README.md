Various utilities for using GNNs with Hydra

## Installation

Make a virtual environment:
```
# if you don't have virtualenv already
# pip3 install --user virtualenv
cd path/to/env
python3 -m virtualenv --download -p $(which python3) hydra_gnn
```

Activate the virtual environment and install:
```
cd path/to/installation
git clone git@github.com:MIT-SPARK/Hydra-GNN.git
cd Hydra-GNN
source path/to/env/hydra_gnn/bin/activate
pip install -e .
```

The training code primarily relies on
  - [PyTorch](https://pytorch.org/get-started/locally/),
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
  - [pyg_lib](https://github.com/pyg-team/pyg-lib) for Heterogeneous GNN operators

While a default install **should** provide everything necessary, you may need to make sure the versions align correctly for these packages.

This code has been tested with:
  - PyTorch 2.0.1, PyTorch Geometric 2.3.1, and Cuda 11.7
  - PyTorch 1.12.1, PyTorch Geometric 2.2.0, and Cuda 11.3
  - PyTorch 1.8.1, PyTorch Geometric 2.0.4, and Cuda 10.2

### Dataset Organization

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

#### Authorship

  - H-tree Construction was written by Rajat Talak

  - Training code was written by Siyi Hu

  - Example inference server with Hydra was written by Nathan Hughes
