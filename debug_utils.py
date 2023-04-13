from hydra_gnn.utils import HYDRA_TRAJ_DIR, MP3D_HOUSE_DIR, COLORMAP_DATA_PATH, WORD2VEC_MODEL_PATH, MP3D_OBJECT_LABEL_DATA_PATH
from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, Hydra_mp3d_htree_data
from hydra_gnn.preprocess_dsgs import hydra_object_feature_converter, dsg_node_converter, OBJECT_LABELS, ROOM_LABELS, _get_label_dict
from spark_dsg.mp3d import load_mp3d_info
import spark_dsg as dsg
import torch_geometric.utils as pyg_utils
import torch
import os
import numpy as np
import gensim
import pandas as pd

import plotly.graph_objects as go


# default data params
threshold_near=1.5
max_near=2.0
max_on=0.2
object_synonyms=[]
room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')]

# data files
colormap_data = pd.read_csv(COLORMAP_DATA_PATH, delimiter=',')
mp3d_object_data = pd.read_csv(MP3D_OBJECT_LABEL_DATA_PATH, delimiter='\t')
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
object_feature_converter=hydra_object_feature_converter(colormap_data, word2vec_model)

# objects we care about
hydra_object_labels = OBJECT_LABELS
object_label_list = [colormap_data['name'][i] for i in hydra_object_labels]
num_object_labels = len(object_label_list)

# rooms we care about
room_label_dict = _get_label_dict(ROOM_LABELS, room_synonyms)
num_room_labels = max(room_label_dict.values()) + 1
room_label_list = [''] * num_room_labels
for mp3d_label, label in room_label_dict.items():
    room_label_list[label] += str(mp3d_label) + ', ' 
room_label_list = [label_str[:-2] for label_str in room_label_list]


# helper function to visualize GNN model
def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


# helper function to regenerate mp3d data
def generate_mp3d_datalist(expand_rooms=False, save_htree=False):
    # room feature converter
    if save_htree:
        room_feature_converter = lambda i: np.zeros(300)
    else:
        room_feature_converter = lambda i: np.zeros(0)

    trajectory_dirs = os.listdir(HYDRA_TRAJ_DIR)
    
    skipped_json_files = {'none': [], 'no room': [], 'no object': []}
    data_list = []
    for i, trajectory_name in enumerate(trajectory_dirs):
        trajectory_dir = os.path.join(HYDRA_TRAJ_DIR, trajectory_name)
        scene_id, _, trajectory_id = trajectory_name.split('_')
        # Load gt house segmentation for room labeling
        gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_id}.house"
        gt_house_info = load_mp3d_info(gt_house_file)
        
        json_file_names = os.listdir(trajectory_dir)
        for json_file_name in json_file_names:
            if json_file_name[0:3] == 'est':
                continue
            num_frames = json_file_name[15:-5]
            file_path = os.path.join(HYDRA_TRAJ_DIR, trajectory_name, json_file_name)
            assert os.path.exists(file_path)

            if save_htree:
                data = Hydra_mp3d_htree_data(scene_id=scene_id, trajectory_id=trajectory_id, \
                    num_frames=num_frames, file_path=file_path, expand_rooms=expand_rooms)
            else:
                data = Hydra_mp3d_data(scene_id=scene_id, trajectory_id=trajectory_id, \
                    num_frames=num_frames, file_path=file_path, expand_rooms=expand_rooms)

            # skip dsg without room node or without object node
            if data.get_room_object_dsg().num_nodes() == 0:
                skipped_json_files['none'].append(os.path.join(trajectory_name, json_file_name))
                continue
            if data.get_room_object_dsg().get_layer(dsg.DsgLayers.ROOMS).num_nodes() == 0:
                skipped_json_files['no room'].append(os.path.join(trajectory_name, json_file_name))
                continue
            if data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_nodes() == 0:
                skipped_json_files['no object'].append(os.path.join(trajectory_name, json_file_name))
                continue

            # parepare torch data
            data.add_room_labels(gt_house_info, angle_deg=-90)
            data.add_object_edges(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
            data.compute_torch_data(
                use_heterogeneous=True,
                node_converter=dsg_node_converter(object_feature_converter, room_feature_converter),
                object_synonyms=object_synonyms, 
                room_synonyms=room_synonyms)
            data.clear_dsg()    # remove hydra dsg for output
            data_list.append(data)
        
        if i == 0:
            print(f"Number of node features: {data.num_node_features()}")

        progress_bar.value = i + 1

    return data_list


# helper function to get room count and room-object co-ocurrences count given a list of Hydra_mp3d_data
def get_room_object_counts(datalist):
    label_dict = datalist[0].get_label_dict()
    num_object_labels = max(label_dict['objects'].values()) + 1
    num_room_labels = max(label_dict['rooms'].values()) + 1
    
    room_object_count = np.zeros((num_room_labels, num_object_labels), dtype=np.int32)
    room_count = np.zeros(num_room_labels, dtype=np.int32)
    for mp3d_data in datalist:
        torch_data = mp3d_data.get_torch_data()
        for r_index, o_index in torch_data[('rooms', 'rooms_to_objects', 'objects')].edge_index.t().tolist():
            room_train_label = torch_data['rooms'].y[r_index].item()
            object_train_label = torch_data['objects'].y[o_index].item()
            room_object_count[room_train_label, object_train_label] += 1
        
        for room_train_label in torch_data['rooms'].y.tolist():
            room_count[room_train_label] += 1
    return room_object_count, room_count


# helper function to get a list of objects per room type
def get_object_per_room(datalist):
    label_dict = datalist[0].get_label_dict()
    num_room_labels = max(label_dict['rooms'].values()) + 1
    
    output_dict = {l: [] for l in range(num_room_labels)}
    for mp3d_data in datalist:
        torch_data = mp3d_data.get_torch_data()
        room_degrees = pyg_utils.degree(torch_data['objects', 'objects_to_rooms', 'rooms'].edge_index[1, :],
                                        num_nodes=torch_data['rooms'].num_nodes, dtype=int)
        for i, deg in enumerate(room_degrees.tolist()):
            output_dict[torch_data['rooms'].y[i].item()].append(deg)
            
    return output_dict


# helper function to log number of nodes and intra-layer edges for each Hydra_mp3d_data
def get_dataset_stat(data_list):
    dataset_stat = dict()
    for data in data_list:
        data_info = data.get_data_info()
        scene_id = data_info['scene_id']
        trajectory_id = data_info['trajectory_id']
        num_frames = data_info['num_frames']
        if scene_id not in dataset_stat:
            dataset_stat[scene_id] = dict()
        if trajectory_id not in dataset_stat[scene_id]:
            dataset_stat[scene_id][trajectory_id] = {'num_rooms': [], 'num_objects': [], 'num_room_edges': [], 'num_object_edges': []}

        dataset_stat[scene_id][trajectory_id]['num_rooms'].append(data.get_torch_data()['rooms'].num_nodes)
        dataset_stat[scene_id][trajectory_id]['num_objects'].append(data.get_torch_data()['objects'].num_nodes)
        dataset_stat[scene_id][trajectory_id]['num_room_edges'].append(data.get_torch_data()['rooms', 'rooms_to_rooms', 'rooms'].num_edges)
        dataset_stat[scene_id][trajectory_id]['num_object_edges'].append(data.get_torch_data()['objects', 'objects_to_objects', 'objects'].num_edges)
    return dataset_stat


# helper function to get maximum data stat (over frames) for each trajectory
def get_max_data_stat(dataset_stat, columns=['max_rooms', 'max_objects', 'max_room_edges', 'max_object_edges'], \
    column_key=['num_rooms', 'num_objects', 'num_room_edges', 'num_object_edges'], func=max):
    df = pd.DataFrame(columns=['trajectory'] + columns)
    for scene_id in dataset_stat:
        for trajectory_id in dataset_stat[scene_id]:
            stat_dict = dataset_stat[scene_id][trajectory_id]
            # save largest number in stat_dict[key] list
            df.loc[len(df.index)] = [f'{scene_id}_{trajectory_id}'] + [func(stat_dict[key]) for key in column_key] 
    return df.sort_values('trajectory')


# helper functions to plot data stats
def plot_histogram(data_dict, opacity=0.5):
    fig = go.Figure()
    for key, value in data_dict.items():
        fig.add_trace(go.Histogram(x=value, name=key))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=opacity)
    return fig


def plot_room_object_heatmap(room_object_data, object_label_list=object_label_list, 
                             room_label_list=room_label_list, title='Room-object Heatmap'):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            y=object_label_list,
            x=room_label_list[:room_object_data.shape[0]],
            z=room_object_data.transpose(),
            colorscale='blues',
            showscale=True,
            ))
    fig.update_layout(
        title=title,
        height=len(object_label_list) * 30,
        width=len(room_label_list) * 30 + 200,)
    return fig


def plot_room_room_heatmap(room_room_data, room_label_list=room_label_list, title='Room-Room Heatmap'):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            y=room_label_list[:room_room_data.shape[1]],
            x=room_label_list[:room_room_data.shape[0]],
            z=room_room_data.transpose(),
            colorscale='blues',
            showscale=True,
            ))
    fig.update_layout(
        title=title,
        xaxis_title='True labels', 
        yaxis_title='Predicted labels',
        height=len(room_label_list) * 30,
        width=len(room_label_list) * 30 + 200,)
    return fig


def plot_object_barchart(data_dict, object_label_list=object_label_list, title='Object occurrences'):
    fig = go.Figure()
    for data_name, data in data_dict.items():
        fig.add_trace(
            go.Bar(name=data_name, 
                   y=object_label_list, 
                   x=data,
                   orientation='h'))

    fig.update_layout(
        barmode='group',
        height=len(object_label_list) * 30,
        title=title)
    return fig


def plot_room_barchart(data_dict, room_label_list=room_label_list, title='Room occurrences'):
    fig = go.Figure()
    for data_name, data in data_dict.items():
        fig.add_trace(
            go.Bar(name=data_name, 
                   x=room_label_list,
                   y=data,
                   orientation='v'))
    fig.update_layout(
        barmode='group',
        height=len(object_label_list) * 30,
        title=title)
    return fig