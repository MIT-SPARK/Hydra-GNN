# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: venv_hydra_gnn
#     language: python
#     name: python3
# ---

# %%
import spark_dsg as dsg
from spark_dsg.mp3d import load_mp3d_info
from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, Hydra_mp3d_htree_data
from hydra_gnn.models import (
    HomogeneousNetwork,
    HeterogeneousNetwork,
    HomogeneousNeuralTreeNetwork,
    HeterogeneousNeuralTreeNetwork,
)
from hydra_gnn.preprocess_dsgs import dsg_node_converter, hydra_object_feature_converter
from hydra_gnn.utils import COLORMAP_DATA_PATH, WORD2VEC_MODEL_PATH
import gensim
import torch
import yaml
import numpy as np
import pandas as pd

# %%
# model files
model_dir = "./output/pretrained_models/data_gt60_model_noSemantics"
with_word2vec = False
hyper_param_path = f"{model_dir}/model.yaml"
model_weight_path = f"{model_dir}/model_weights.pth"

# example dsg
example_dsg_path = "./tests/test_data/17DRP5sb8fy_0_gt_partial_dsg_1414.json"
example_house_file_path = "./tests/test_data/17DRP5sb8fy.house"

# %%
# room labels filtering function -- keep rooms that have more than 1 children (objects) or have siblings (rooms)
room_removal_func = lambda room: not (len(room.children()) > 1 or room.has_siblings())

# dsg construction/conversion params
threshold_near = 1.5
max_near = 2.0
max_on = 0.2
object_synonyms = []
room_synonyms = [("a", "t"), ("z", "Z", "x", "p", "\x15")]
min_iou = 0.6

# %% [markdown]
# ## Load pretrained model

# %%
with open(hyper_param_path, "r") as input_file:
    model_param = yaml.safe_load(input_file)

network_type, graph_type = model_param["network_type"], model_param["graph_type"]
print(f"network type: {network_type}")
print(f"graph type: {graph_type}")
print(f"model hyper params: {model_param['network_params']}")

# %%
# initialize model and load weights
if model_param["graph_type"] == "homogeneous":
    if network_type == "baseline":
        model = HomogeneousNetwork(**model_param["network_params"])
    else:
        model = HomogeneousNeuralTreeNetwork(**model_param["network_params"])
else:
    if network_type == "baseline":
        model = HeterogeneousNetwork(**model_param["network_params"])
    else:
        model = HeterogeneousNeuralTreeNetwork(**model_param["network_params"])

model.load_state_dict(torch.load(model_weight_path))


# %% [markdown]
# ## Prepare data

# %%
# dsg node attributes to PyG node feature converter
colormap_data = pd.read_csv(COLORMAP_DATA_PATH, delimiter=",")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    WORD2VEC_MODEL_PATH, binary=True
)
object_feature_converter = hydra_object_feature_converter(colormap_data, word2vec_model)
room_feature_converter = lambda i: np.zeros(300)

# %%
# load example graph for testing
if model_param["network_type"] == "baseline":
    data = Hydra_mp3d_data(
        scene_id=0, trajectory_id=0, num_frames=0, file_path=example_dsg_path
    )
else:
    data = Hydra_mp3d_htree_data(
        scene_id=0, trajectory_id=0, num_frames=0, file_path=example_dsg_path
    )

# skip dsg without room node or without object node
if (
    data.get_room_object_dsg().num_nodes() == 0
    or data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_nodes() == 0
    or data.get_room_object_dsg().get_layer(dsg.DsgLayers.ROOMS).num_nodes() == 0
):
    raise RuntimeError("Input dsg does not satisfy minimum node number requirement.")

# parepare torch data
data.add_dsg_room_labels(
    load_mp3d_info(example_house_file_path),
    angle_deg=-90,
    room_removal_func=room_removal_func,
    min_iou_threshold=min_iou,
    repartition_rooms=True,
)
if data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_nodes() == 0:
    raise RuntimeError(
        "Input dsg does not contain any object node after room repartitioning."
    )

data.add_object_edges(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
data.compute_torch_data(
    use_heterogeneous=True,
    node_converter=dsg_node_converter(object_feature_converter, room_feature_converter),
    object_synonyms=object_synonyms,
    room_synonyms=room_synonyms,
)
data.clear_dsg()  # remove hydra dsg

if model_param["network_params"]["conv_block"] == "GAT_edge":
    data.compute_relative_pos()
if graph_type == "homogeneous":
    data.to_homogeneous()
if not with_word2vec:
    data.remove_last_features(300)

# get PyG data
data = data.get_torch_data()

# %% [markdown]
# # Pass data through pre-trained model

# %%
# pass prepared data through model - this will run on cuda if gpu is available
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

with torch.no_grad():
    pred = model(data.to(device)).argmax(dim=1)
    if graph_type == "homogeneous":
        label = data.y[data.room_mask]
    else:
        if network_type == "baseline":
            label = data["rooms"].y
        else:
            label = data["room_virtual"].y
    mask = label != 25  # ignore label 25, which is the unknown/filtered label
    pred = pred[mask]
    label = label[mask]
print(f"Predicted room labels: {pred}")
print(f"Ground truth room labels: {label}")
