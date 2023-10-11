# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: 'Python 3.7.5 (''venv_hydra_gnn'': venv)'
#     language: python
#     name: python3
# ---

# %%
"""Compare room-object occurences in hydra graphs and ground-truth mp3d graphs"""
from collections import Counter
import os
import numpy as np
import spark_dsg as dsg
from spark_dsg.mp3d import (
    load_mp3d_info,
    add_gt_room_label,
    get_rooms_from_mp3d_info,
    repartition_rooms,
)
from hydra_gnn.preprocess_dsgs import get_room_object_dsg, OBJECT_LABELS
from hydra_gnn.utils import HYDRA_TRAJ_DIR, MP3D_HOUSE_DIR
from hydra_gnn.debug_utils import (
    plot_room_object_heatmap,
    plot_room_barchart,
    plot_object_barchart,
    object_label_list,
    mp3d_object_data,
    room_label_dict,
    room_label_list,
)
from ipywidgets import IntProgress

# default grouped room labels (for learning):
#   room_synonyms = [('a', 't'), ('z', 'Z', 'x', 'p')]
# The first tuple is toilet/bathroom, the second is outdoor/unlabeled space. Change this setting in debug_utils.py if needed.

# dataset offset
hydra_angle_deg = -90.0  # rotate mp3d rooms by this angle to match hydra graphs

# %% [markdown]
# ## Object and room labels

# %%
num_object_labels = len(object_label_list)
print("object semantic labels:\n  ", object_label_list)

# %%
# hydra/mp3d label index to training label index mapping
hydra_object_label_dict = dict(zip(OBJECT_LABELS, range(len(OBJECT_LABELS))))
print("Hydra object label to training label:\n  ", hydra_object_label_dict)

mp3d_object_label = [
    int(mp3d_object_data[mp3d_object_data["mpcat40"] == label]["mpcat40index"])
    for label in object_label_list
]
mp3d_object_label_dict = dict(zip(mp3d_object_label, range(len(mp3d_object_label))))
print("Mp3d object label to training label:\n  ", mp3d_object_label_dict)


# %%
num_room_labels = len(room_label_list)
print("room semantic labels:\n  ", room_label_list)

# %%
# mp3d room semantic label to training label index mapping
print("room semantic label to training label:\n  ", room_label_dict)

# %% [markdown]
# ## Load dataset files

# %%
trajectory_dirs = os.listdir(HYDRA_TRAJ_DIR)

scene_counter = Counter(full_name.split("_")[0] for full_name in trajectory_dirs)
scene_names = list(scene_counter)
print("Found {} scenes.".format(len(scene_names)))

# %% [markdown]
# ## Get mp3d room-object co-occurrences

# %%
progress_bar = IntProgress(
    value=0, min=0, max=len(scene_names), description="Progress:", bar_stlye="info"
)
display(progress_bar)

gt_room_object_count = np.zeros((num_room_labels, num_object_labels))
gt_room_count = np.zeros(num_room_labels)
for scene_name in scene_names:
    # Load gt house segmentation for room labeling
    gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_name}.house"
    gt_house_info = load_mp3d_info(gt_house_file)
    mp3d_rooms = get_rooms_from_mp3d_info(gt_house_info, angle_deg=0.0)
    print("GT segmentation file:", gt_house_file)

    for object_i in gt_house_info["O"]:
        object_category_index = object_i["category_index"]
        if object_category_index == -1:
            continue

        # object label
        mpcat40_index = gt_house_info["C"][object_category_index]["mpcat40_index"]
        assert (
            gt_house_info["C"][object_category_index]["category_index"]
            == object_category_index
        ), print(
            object_i,
            gt_house_info["C"][object_category_index]["category_index"],
            object_category_index,
        )  # sanity check
        if mpcat40_index not in mp3d_object_label_dict.keys():
            continue

        # room label
        object_region_index = object_i["region_index"]
        room_char = gt_house_info["R"][object_region_index]["label"]
        assert mp3d_rooms[object_region_index].semantic_label == ord(
            room_char
        )  # sanity check

        ro_geometry_output_str = f"{mp3d_rooms[object_region_index].get_id()} ({gt_house_info['R'][object_region_index]['bbox_min']}, {gt_house_info['R'][object_region_index]['bbox_max']})-{room_char} ({object_i['pos']})-{mpcat40_index}"
        # assert mp3d_rooms[object_region_index].pos_inside_room(object_i["pos"]), error_output_str
        # assert np.all(object_i['pos'] <= gt_house_info['R'][object_region_index]['bbox_max']) and np.all(object_i['pos'] >= gt_house_info['R'][object_region_index]['bbox_min']), error_output_str
        # assert mp3d_rooms[object_region_index].pos_on_same_floor(object_i["pos"]), error_output_str
        # assert np.sum(np.maximum(object_i["pos"] - gt_house_info['R'][object_region_index]['bbox_max'], 0) + \
        #      np.maximum(gt_house_info['R'][object_region_index]['bbox_min'] - object_i["pos"], 0)) <= 0.1:
        if not mp3d_rooms[object_region_index].pos_on_same_floor(object_i["pos"]):
            for mp3d_room_j in mp3d_rooms:
                if mp3d_room_j.pos_inside_room(object_i["pos"]):
                    room_char = chr(
                        mp3d_room_j.semantic_label
                    )  # update room_char if another room contains object_i
                    jj = mp3d_room_j.get_id().category_id
                    print(
                        f"   O({mpcat40_index})-{object_label_list[mp3d_object_label_dict[mpcat40_index]]} "
                        f"{mp3d_rooms[object_region_index].get_id()}-{chr(mp3d_rooms[object_region_index].semantic_label)}"
                        f" -> {mp3d_room_j.get_id()}-{room_char}"
                    )
                    break

        # update room-object co-occurences
        gt_room_object_count[
            room_label_dict[room_char], mp3d_object_label_dict[mpcat40_index]
        ] += 1

    # update room count
    for mp3d_room_j in mp3d_rooms:
        room_char = chr(mp3d_room_j.semantic_label)
        gt_room_count[room_label_dict[room_char]] += 1

    progress_bar.value += 1

# %%
fig = plot_room_object_heatmap(
    gt_room_object_count, title="MP3D ground truth room-object co-occurrences"
)
fig.show()
# fig.write_html('./output/mp3d_heatmap.html')

# %% [markdown]
# ## Get hydra room-object co-occurrences from longest scans of each trajectory

# %%
progress_bar = IntProgress(
    value=0, min=0, max=len(trajectory_dirs), description="Progress:", bar_stlye="info"
)
display(progress_bar)

hydra_room_object_count = np.zeros((num_room_labels, num_object_labels))
hydra_room_count = np.zeros(num_room_labels)
for trajectory_name in trajectory_dirs:
    trajectory_path = os.path.join(HYDRA_TRAJ_DIR, trajectory_name)
    scene_name = trajectory_name[:11]

    # Load gt house segmentation for room labeling
    gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_name}.house"
    gt_house_info = load_mp3d_info(gt_house_file)
    # print("GT segmentation file:", gt_house_file)

    # Find the longest GT trajectories of all scenes
    max_frame = max(
        int(traj_name[:-5].split("_")[3]) for traj_name in os.listdir(trajectory_path)
    )
    max_frame_data_path = f"{trajectory_path}/gt_partial_dsg_{max_frame}.json"

    # Load hydra scene graph
    G = dsg.DynamicSceneGraph.load(max_frame_data_path)
    # print("Number of nodes separated by layer: {} ({} total).\n".format([layer.num_nodes() for layer in G.layers], G.num_nodes()))

    # add room bounding box and label
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
    add_gt_room_label(
        G, gt_house_info, angle_deg=hydra_angle_deg, use_hydra_polygon=False
    )

    # extract room-object graph
    G_ro = get_room_object_dsg(G)

    # count hydra room-object co-occurences
    for object in G_ro.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        room = G_ro.get_node(object.get_parent())
        hydra_room_object_count[
            room_label_dict[chr(room.attributes.semantic_label)],
            hydra_object_label_dict[object.attributes.semantic_label],
        ] += 1

    # count hydra room occurences
    for room in G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes:
        hydra_room_count[room_label_dict[chr(room.attributes.semantic_label)]] += 1

    progress_bar.value += 1


# %%
fig = plot_room_object_heatmap(
    hydra_room_object_count / 5, title="Hydra room-object co-occurrences"
)
fig.show()
# fig.write_html('./output/hydra_heatmap.html')

# %% [markdown]
# ## Sanity Check: hydra room-object co-occurrences with gt room resegmentation (same trajectories as above)

# %%
progress_bar = IntProgress(
    value=0, min=0, max=len(trajectory_dirs), description="Progress:", bar_stlye="info"
)
display(progress_bar)

hydra_reseg_room_object_count = np.zeros((num_room_labels, num_object_labels))
hydra_reseg_room_count = np.zeros(num_room_labels)
for trajectory_name in trajectory_dirs:
    trajectory_path = os.path.join(HYDRA_TRAJ_DIR, trajectory_name)
    scene_name = trajectory_name[:11]

    # Load gt house segmentation for room labeling
    gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_name}.house"
    gt_house_info = load_mp3d_info(gt_house_file)
    # print("GT segmentation file:", gt_house_file)

    # Find the longest GT trajectories of all scenes
    max_frame = max(
        int(traj_name[:-5].split("_")[3]) for traj_name in os.listdir(trajectory_path)
    )
    max_frame_data_path = f"{trajectory_path}/gt_partial_dsg_{max_frame}.json"

    # Load hydra scene graph
    G = dsg.DynamicSceneGraph.load(max_frame_data_path)
    # print("Number of nodes separated by layer: {} ({} total).\n".format([layer.num_nodes() for layer in G.layers], G.num_nodes()))

    # Resegment rooms with ground-truth mp3d room geometry
    G = repartition_rooms(G, gt_house_info)

    # add room bounding box and label
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
    add_gt_room_label(
        G, gt_house_info, angle_deg=hydra_angle_deg, use_hydra_polygon=False
    )

    # extract room-object graph
    G_ro = get_room_object_dsg(G)

    # count hydra room-object co-occurences
    for object in G_ro.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        room = G_ro.get_node(object.get_parent())
        hydra_reseg_room_object_count[
            room_label_dict[chr(room.attributes.semantic_label)],
            hydra_object_label_dict[object.attributes.semantic_label],
        ] += 1

    # count hydra room occurences
    for room in G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes:
        hydra_reseg_room_count[
            room_label_dict[chr(room.attributes.semantic_label)]
        ] += 1

    progress_bar.value += 1

# %%
fig = plot_room_object_heatmap(
    hydra_reseg_room_object_count / 5,
    title="Hydra resegmented room-object co-occurrences",
)
fig.show()
# fig.write_html('./output/hydraReseg_heatmap.html')

# %% [markdown]
# ## Comparisons

# %%
# save room-object count data
import pickle

count_data = dict(
    objects=object_label_list,
    rooms=room_label_list,
    gt_room_object_count=gt_room_object_count,
    hydra_room_object_count=hydra_room_object_count,
    hydra_reseg_room_object_count=hydra_reseg_room_object_count,
)
with open("count_data.pkl", "wb") as output_file:
    pickle.dump(count_data, output_file)

# normalize across rooms for each object
hydra_room_object_count_normalized = hydra_room_object_count / np.linalg.norm(
    hydra_room_object_count, ord=1, axis=0
)
hydra_reseg_room_object_count_normalized = (
    hydra_reseg_room_object_count
    / np.linalg.norm(hydra_reseg_room_object_count, ord=1, axis=0)
)
gt_room_object_count_normalized = gt_room_object_count / np.linalg.norm(
    gt_room_object_count, ord=1, axis=0
)

# %%
fig = plot_room_object_heatmap(
    np.abs(hydra_room_object_count_normalized - gt_room_object_count_normalized),
    title="Normalized (over rooms) Hydra to GT co-occurrence differences",
)
fig.show()
# fig.write_html('./output/comp_hydra_mp3d_heatmap.html')

# %%
fig = plot_room_object_heatmap(
    np.abs(hydra_reseg_room_object_count_normalized - gt_room_object_count_normalized),
    title="Normalized (over rooms) Hydra-reseg to GT co-occurrence differences",
)
fig.show()
# fig.write_html('./output/comp_hydraReseg_mp3d_heatmap.html')

# %%
fig = plot_object_barchart(
    {
        "Hydra": np.sum(hydra_room_object_count / 5, axis=0),
        "Hydra-reseg": np.sum(hydra_reseg_room_object_count / 5, axis=0),
        "Mp3d-gt": np.sum(gt_room_object_count, axis=0),
    },
    title="Number of object occurrences",
)
fig.show()
# fig.write_html('./output/objects_barchart.html')

# %%
fig = plot_object_barchart(
    {
        "Hydra / Mp3d-gt": np.sum(hydra_room_object_count / 5, axis=0)
        / np.sum(gt_room_object_count, axis=0),
        "Hydra-reseg / Mp3d-gt": np.sum(hydra_reseg_room_object_count / 5, axis=0)
        / np.sum(gt_room_object_count, axis=0),
    },
    title="Hydra / GT object occurrences",
)
fig.show()
# fig.write_html('./output/comp_objects_barchart.html')

# %%
# compare room occurences
fig = plot_room_barchart(
    {
        "Hydra": hydra_room_count / 5,
        "Hydra-reseg": hydra_reseg_room_count / 5,
        "Mp3d-gt": gt_room_count,
    },
    title="Number of room occurrences",
)
fig.show()
# fig.write_html('./output/rooms_barchart.html')

# %%
fig = plot_room_barchart(
    {
        "Hydra / Mp3d-gt": (hydra_room_count / 5) / gt_room_count,
        "Hydra-reseg / Mp3d-gt": (hydra_reseg_room_count / 5) / gt_room_count,
    },
    title="Hydra / GT room occurrences",
)
fig.show()
# fig.write_html('./output/comp_rooms_barchart.html')

# %%
# compare number of objects by room type
fig = plot_room_barchart(
    {
        "Hydra": np.sum(hydra_room_object_count / 5, axis=1),
        "Hydra-reseg": np.sum(hydra_reseg_room_object_count / 5, axis=1),
        "Mp3d-gt": np.sum(gt_room_object_count, axis=1),
    },
    title="Number of objects by room type",
)
fig.show()
# fig.write_html('./output/num_objects_by_room_barchart.html')
