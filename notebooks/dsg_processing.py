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
from hydra_gnn.utils import (
    MP3D_HOUSE_DIR,
    HYDRA_TRAJ_DIR,
    COLORMAP_DATA_PATH,
    WORD2VEC_MODEL_PATH,
    plot_heterogeneous_graph,

)
from hydra_gnn.preprocess_dsgs import (
    convert_label_to_y,
    add_object_connectivity,
    get_room_object_dsg,
)

# %% [markdown]
# ## Load DSG

# %%
# test dataset file paths
test_json_file = os.path.join(
    HYDRA_TRAJ_DIR, "759xd9YjKW5_trajectory_0/gt_partial_dsg_1640.json"
)
gt_house_file = os.path.join(MP3D_HOUSE_DIR, "759xd9YjKW5.house")

# sparse room
# test_json_file = os.path.join(
#     HYDRA_TRAJ_DIR, "2t7WUuJeko7_trajectory_1/gt_partial_dsg_1352.json"
# )
# gt_house_file = os.path.join(MP3D_HOUSE_DIR, "2t7WUuJeko7.house")

# # single room
# test_json_file = os.path.join(
#   HYDRA_TRAJ_DIR, "YVUC4YcDtcY_trajectory_0/gt_partial_dsg_1000.json"
# )
# gt_house_file = os.path.join(MP3D_HOUSE_DIR, "YVUC4YcDtcY.house")

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
dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
add_gt_room_label(G, gt_house_info, angle_deg=-90)
# dsg.render_to_open3d(G)

# Convert raw input graph to heterogeneous torch data
data = G.to_torch(use_heterogeneous=True)
fig = plot_heterogeneous_graph(data)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)
fig.show()

# %%
# Repartition rooms using ground-truth room geometry
G_rep = repartition_rooms(G, gt_house_info, min_iou_threshold=0.6, verbose=True)

# Convert repartitioned graph to heterogeneous torch data
data_rep = G_rep.to_torch(use_heterogeneous=True)
fig = plot_heterogeneous_graph(data_rep)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)
fig.show()

# %% [markdown]
# ## Get room-object graph and convert to torch_data

# %%
import gensim
import pandas as pd
from hydra_gnn.preprocess_dsgs import hydra_object_feature_converter, dsg_node_converter

# %%
# extract room-object dsg and add object connectivity
G_ro = get_room_object_dsg(G_rep, verbose=False)
add_object_connectivity(G_ro, threshold_near=2.0, max_on=0.2, max_near=2.0)

# %%
# remove isolated room nodes
room_removal_func = lambda room: not (len(room.children()) > 1 or room.has_siblings())

for room in G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes:
    if room_removal_func(room):
        room.attributes.semantic_label = ord("\x15")

# %%
# object features
colormap_data = pd.read_csv(COLORMAP_DATA_PATH, delimiter=",")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    WORD2VEC_MODEL_PATH, binary=True
)

# %%
# data conversion
data_ro = G_ro.to_torch(
    use_heterogeneous=True,
    node_converter=dsg_node_converter(
        object_feature_converter=hydra_object_feature_converter(
            colormap_data, word2vec_model
        ),
        room_feature_converter=lambda i: np.empty(0),
    ),
)

# %%
# get label index y from hydra node labels
synonym_objects = []
synonym_rooms = [("a", "t"), ("z", "Z", "x", "p", "\x15")]

object_label_dict, room_label_dict = convert_label_to_y(
    data_ro, object_synonyms=synonym_objects, room_synonyms=synonym_rooms
)

# %%
fig = plot_heterogeneous_graph(data_ro)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)
fig.show()

# %% [markdown]
# ## Print hydra labels and training indices

# %%
# Hydra/MP3D labels in input graphs
labels = [
    data_ro[node_type].label for node_type in data_ro.node_types
]
print("object hydra/mp3d labels:", labels[0].tolist())
print("room hydra/mp3d labels:", [chr(l) for l in labels[1].tolist()])

# %%
# training labels
print("object label indices:", data_ro["objects"].y)
print("room label indices", data_ro["rooms"].y)
