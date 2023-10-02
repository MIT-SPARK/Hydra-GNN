# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: hydra-gnn
#     language: python
#     name: hydra-gnn
# ---

# %%
import os
import numpy as np
import spark_dsg as dsg
from spark_dsg.mp3d import load_mp3d_info, repartition_rooms, add_gt_room_label
from hydra_gnn.utils import plot_heterogeneous_graph
from hydra_gnn.preprocess_dsgs import (
    convert_label_to_y,
    add_object_connectivity,
    get_room_object_dsg,
)

# %%
# dataset file paths
hydra_dataset_dir = "./data/tro_graphs_2022_09_24"
mp3d_housefile_dir = "./data/house_files"
colormap_data_path = "./data/colormap.csv"
mp3d_label_data_path = "./data/mpcat40.tsv"
word2vec_model_path = "./data/GoogleNews-vectors-negative300.bin"

# %% [markdown]
# ## Load DSG

# %%
# test dataset file paths
# sparse room
test_json_file = (
    "./data/tro_graphs_2022_09_24/2t7WUuJeko7_trajectory_0/gt_partial_dsg_1330.json"
)
gt_house_file = "./data/house_files/2t7WUuJeko7.house"

# # single room
# # test_json_file = "./data/tro_graphs_2022_09_24/YVUC4YcDtcY_trajectory_0/gt_partial_dsg_1000.json"
# # gt_house_file = "./data/house_files/YVUC4YcDtcY.house"

assert os.path.exists(test_json_file)
assert os.path.exists(gt_house_file)

# %%
# Load hydra scene graph
G = dsg.DynamicSceneGraph.load(test_json_file)
print(
    "Number of nodes separated by layer: {} ({} total).".format(
        [layer.num_nodes() for layer in G.layers], G.num_nodes()
    )
)
gt_house_info = load_mp3d_info(gt_house_file)
# dsg.render_to_open3d(G)

# %%
data = G.to_torch(use_heterogeneous=True)
fig = plot_heterogeneous_graph(data)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)

fig.show()

# %%
G_rep0 = repartition_rooms(G, gt_house_info, min_iou_threshold=0.0, verbose=True)
data0 = G_rep0.to_torch(use_heterogeneous=True)
fig = plot_heterogeneous_graph(data0)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)

fig.show()

# %%
G_rep50 = repartition_rooms(G, gt_house_info, min_iou_threshold=0.5, verbose=True)
data50 = G_rep50.to_torch(use_heterogeneous=True)
fig = plot_heterogeneous_graph(data50)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)

fig.show()

# %% [markdown]
# ## Add room label from GT

# %%
# room labels are based on maximum IoU between hydra room bbox and mp3d room segmentation
dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
add_gt_room_label(G, gt_house_info, angle_deg=-90)

# %% [markdown]
# ## Get room-object graph and convert to torch_data

# %%
import gensim
import pandas as pd
from hydra_gnn.preprocess_dsgs import hydra_object_feature_converter, dsg_node_converter

# %%
# extract room-object dsg and add object connectivity
G_ro = get_room_object_dsg(G, verbose=False)
add_object_connectivity(G_ro, threshold_near=2.0, max_on=0.2, max_near=2.0)

# %%
# remove isolated room nodes
room_removal_func = lambda room: not (room.has_children() or room.has_siblings())

for room in G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes:
    if room_removal_func(room):
        room.attributes.semantic_label = ord("\x15")

# %%
colormap_data = pd.read_csv(colormap_data_path, delimiter=",")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    word2vec_model_path, binary=True
)

# %%
# data conversion
data_heterogeneous = G_ro.to_torch(
    use_heterogeneous=True,
    node_converter=dsg_node_converter(
        object_feature_converter=hydra_object_feature_converter(
            colormap_data, word2vec_model
        ),
        room_feature_converter=lambda i: np.empty(0),
    ),
)
# data_homogeneous = G_ro.to_torch(use_heterogeneous=False,
#                                  node_converter=dsg_node_converter(
#                                   object_feature_converter=hydra_object_feature_converter(colormap_data, word2vec_model),
#                                   room_feature_converter=lambda i: np.empty(300)))

# %%
# get label index data_*.y from hydra node labels
synonym_objects = []
synonym_rooms = [("a", "t"), ("z", "Z", "x", "p", "\x15")]

object_label_dict, room_label_dict = convert_label_to_y(
    data_heterogeneous, object_synonyms=synonym_objects, room_synonyms=synonym_rooms
)
# object_label_dict, room_label_dict = convert_label_to_y(data_homogeneous, object_synonyms=synonym_objects, room_synonyms=synonym_rooms)

print(object_label_dict)
print(room_label_dict)

# %%
fig = plot_heterogeneous_graph(data_heterogeneous)
fig.show()

# %%
data_heterogeneous["rooms"].y

# %% [markdown]
# ## Debug

# %%
# hydra labels
labels = [
    data_heterogeneous[node_type].label for node_type in data_heterogeneous.node_types
]
print("object mp3d labels:", labels[0].tolist())
print("room mp3d labels:", [chr(l) for l in labels[1].tolist()])

# %%
# training labels
print(data_heterogeneous["objects"].y)
print(data_heterogeneous["rooms"].y)

# %%
data_heterogeneous["rooms", "rooms_to_objects", "objects"].edge_index
