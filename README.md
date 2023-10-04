This repository contains code to train room-classification networks using 3D scene graphs as input.
It is based on the papers:
  - ["Neural Trees for Learning on Graphs"](https://proceedings.neurips.cc/paper/2021/file/ddf88ea64eaed0f3de5531ac964a0a1a-Paper.pdf)
  - ["Foundations of Spatial Perception for Robotics: Hierarchical Representations and Real-time Systems"](https://arxiv.org/abs/2305.07154)

If you find this code relevant for your work, please consider citing one or both of these papers. Bibtex entries are provided below:

```
@inproceedings{talak2021neuraltree,
               author = {Talak, Rajat and Hu, Siyi and Peng, Lisa and Carlone, Luca},
               booktitle = {Advances in Neural Information Processing Systems},
               title = {Neural Trees for Learning on Graphs},
               year = {2021}
}

@article{hughes2023foundations,
         title={Foundations of Spatial Perception for Robotics: Hierarchical Representations and Real-time Systems},
         author={Nathan Hughes and Yun Chang and Siyi Hu and Rajat Talak and Rumaisa Abdulhai and Jared Strader and Luca Carlone},
         year={2023},
         eprint={2305.07154},
         archivePrefix={arXiv},
         primaryClass={cs.RO}
}
```

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

## Dataset Organization

All datasets and resoruces (such as the pre-trained word2vec model) live in the `./data` folder. It is organized as follows:

- data
  - [GoogleNews-vectors-negative300.bin](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)
  - Stanford3DSceneGraph
    - [tiny](https://github.com/StanfordVL/3DSceneGraph)
  - house_files (can be obtained from the habitat mp3d dataset following the download instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset))
  - mp3d_benchmark
  - mpcat40.tsv
  - tro_graphs_2022_09_24

Steps to get started for training:
1) Obtain the word2vec model from [here](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)
2) Obtain the house files for each mp3d scene from the MP3D dataset and extract them to the house_files directory
3) Obtain the Hydra-produced scene graphs from [here](https://drive.google.com/drive/folders/1OgQOLYKUg5nRdZnfWQsFspBd7HEV5ZyW?usp=sharing)
4) (optional) Obtain the Stanford3D tiny split from [here](https://github.com/StanfordVL/3DSceneGraph)

## Training

Before training, you must construct the relevant pytorch-geometric dataset. For Stanford3D, you can do that via
```
python scripts/prepare_Stanford3DSG_for_training.py
```
and for MP3D you can do that via
```
python scripts/prepare_Stanford3DSG_for_training.py
```

You can examine both scripts with `--help` to view possible arguments.

Training for a specific dataset can be run via
```
python scripts/train_Stanford.py
```
or
```
python scripts/train_mp3d.py
```

## Running with Hydra

We provide pre-trained models [here](https://drive.google.com/drive/folders/1OgQOLYKUg5nRdZnfWQsFspBd7HEV5ZyW?usp=sharing)

First, start the GNN model via
```
./bin/room_classification_server server path/to/pretrained/model path/to/hydra/label/space
```

For the uhumans2 office, this would look like
```
./bin/room_classification_server server path/to/pretrained/model path/to/hydra/config/uhumans2/uhumans2_office_typology.yaml
```

Then, start Hydra with the `use_zmq_interface:=true` argument. For the uhumans2 office scene, this would look like:
```
roslaunch hydra_ros uhumans2.launch use_zmq_interface:=true
```

## Authorship

  - Primary author is Siyi Hu

  - H-tree Construction was written by Rajat Talak

  - Example inference server for Hydra was written by Nathan Hughes
