from hydra_gnn.preprocess_dsgs import hydra_node_converter, hydra_object_feature_converter, \
    OBJECT_LABELS, ROOM_LABELS
from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, Hydra_mp3d_dataset, EDGE_TYPES
from spark_dsg.mp3d import load_mp3d_info
import spark_dsg as dsg
import torch
import numpy as np
import warnings
import os.path
from torch_geometric.data import HeteroData
import pytest


def test_Hydra_mp3d_data(test_data_dir):
    test_json_file = test_data_dir / "x8F5xyUWy9e_0_gt_partial_dsg_1447.json"
    gt_house_file = test_data_dir / "x8F5xyUWy9e.house"
    if not (os.path.exists(test_json_file) and os.path.exists(gt_house_file)):
        warnings.warn(UserWarning("test data file missing. -- skip test"))
        return

    data = Hydra_mp3d_data(scene_id='x8F5xyUWy9e', trajectory_id=0, num_frames=1447, \
        file_path=str(test_json_file))
    assert data.get_data_info()['scene_id'] == 'x8F5xyUWy9e'
    assert data.get_data_info()['trajectory_id'] == 0
    assert data.get_data_info()['num_frames'] == 1447
    assert data.get_data_info()['file_path'] == str(test_json_file)
    assert data.is_heterogeneous() is None

    G = data.get_full_dsg()
    G_ro = data.get_room_object_dsg()
    assert G.get_layer(dsg.DsgLayers.ROOMS).num_nodes() == G_ro.get_layer(dsg.DsgLayers.ROOMS).num_nodes()
    assert G.get_layer(dsg.DsgLayers.ROOMS).num_edges() == G_ro.get_layer(dsg.DsgLayers.ROOMS).num_edges()
    assert G.get_layer(dsg.DsgLayers.OBJECTS).num_nodes() >= G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
    assert G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_edges() == 0
    assert G_ro.num_nodes() == G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_nodes() + \
            G_ro.get_layer(dsg.DsgLayers.ROOMS).num_nodes()

    # add room label and add object edges to room-object graph
    gt_house_info = load_mp3d_info(gt_house_file)
    data.add_room_labels(gt_house_info, angle_deg=-90)

    assert data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_edges() == 0
    data.add_object_edges(threshold_near=2.0, threshold_on=1.0, max_near=2.0)
    assert data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_edges() > 0

    # load data info
    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
        return
    object_feature_converter=hydra_object_feature_converter(
        pytest.colormap_data, pytest.word2vec_model)

    # expected results
    object_synonyms = []
    room_synonyms = [('a', 't'), ('z', 'Z', 'x', 'p', '\x15')]
    room_y_exp = torch.tensor([12, 23, 3, 23, 23])

    # setup 1: convert room-object graph to homogeneous torch data
    data.compute_torch_data(use_heterogeneous=False,
        node_converter=hydra_node_converter(object_feature_converter, lambda i: np.empty(300)),
        object_synonyms=object_synonyms, room_synonyms=room_synonyms)
    torch_data = data.get_torch_data()
    # check y is computed
    assert torch.numel(torch_data.y) == torch_data.num_nodes
    assert torch.all(torch_data.y[torch_data.node_masks[4]] == room_y_exp)
    #  check label dict is computed
    object_label_dict = data.get_label_dict()['objects']
    room_label_dict = data.get_label_dict()['rooms']
    assert len(object_label_dict) == 28
    assert all(hydra_label in OBJECT_LABELS for hydra_label in object_label_dict.keys())
    assert all(hydra_label in range(28) for hydra_label in object_label_dict.values())
    assert len(room_label_dict) == 31
    assert all(hydra_label in ROOM_LABELS for hydra_label in room_label_dict.keys())
    assert all(hydra_label in range(26) for hydra_label in room_label_dict.values())
    # check numbers
    assert data.num_node_features() == (306, 306)
    assert data.num_room_labels() == 26
    assert data.num_object_labels() == 28

    # setup 2: convert room-object graph to heterogeneous torch data
    data.compute_torch_data(use_heterogeneous=True,
        node_converter=hydra_node_converter(object_feature_converter, lambda i: np.empty(0)),
        object_synonyms=object_synonyms, room_synonyms=room_synonyms)
    torch_data = data.get_torch_data()
    # check y is computed
    assert torch.numel(torch_data['objects'].y) == \
        G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
    assert torch.numel(torch_data['rooms'].y) == \
        G_ro.get_layer(dsg.DsgLayers.ROOMS).num_nodes()
    assert torch.all(torch_data['rooms'].y == room_y_exp)
    #  check label dict is computed
    object_label_dict = data.get_label_dict()['objects']
    room_label_dict = data.get_label_dict()['rooms']
    assert len(object_label_dict) == 28
    assert all(hydra_label in OBJECT_LABELS for hydra_label in object_label_dict.keys())
    assert all(hydra_label in range(28) for hydra_label in object_label_dict.values())
    assert len(room_label_dict) == 31
    assert all(hydra_label in ROOM_LABELS for hydra_label in room_label_dict.keys())
    assert all(hydra_label in range(26) for hydra_label in room_label_dict.values())
    # check numbers
    assert data.num_node_features() == (6, 306)
    assert data.num_room_labels() == 26
    assert data.num_object_labels() == 28


def test_fill_missing_edge_index():
    # data with 1 room
    data = HeteroData()
    data['rooms'].x = torch.rand((1, 6))
    data['objects'].x = torch.rand((3, 306))
    object_edge = torch.tensor([[0, 1, 1, 2, 2, 0],
                                [1, 0, 2, 1, 0, 2]])
    room_object_edge = torch.tensor([[0, 0, 0],
                                     [0, 1, 2]])
    data['objects', 'objects_to_objects', 'objects'].edge_index = object_edge
    data['rooms', 'rooms_to_objects', 'objects'].edge_index = room_object_edge
    
    Hydra_mp3d_data.fill_missing_edge_index(data, edge_types=EDGE_TYPES)
    assert ('rooms', 'rooms_to_rooms', 'rooms') in data.edge_index_dict
    assert data[('rooms', 'rooms_to_rooms', 'rooms')].num_edges == 0
    assert ('objects', 'objects_to_rooms', 'rooms') in data.edge_index_dict
    assert torch.all(data[('objects', 'objects_to_rooms', 'rooms')].edge_index == \
        data['rooms', 'rooms_to_objects', 'objects'].edge_index.flip([0])), data[('object', 'objects_to_rooms', 'rooms')].edge_index

    # data with 2 room
    data = HeteroData()
    data['rooms'].x = torch.rand((2, 6))
    data['objects'].x = torch.rand((4, 306))
    room_edge = torch.tensor([[0, 1],
                              [1, 0]])
    object_edge = torch.tensor([[0, 1, 1, 2, 2, 0],
                                [1, 0, 2, 1, 0, 2]])
    room_object_edge = torch.tensor([[0, 0, 0, 1],
                                     [0, 1, 2, 3]])
    data['rooms', 'rooms_to_rooms', 'rooms'].edge_index = room_edge
    data['objects', 'objects_to_objects', 'objects'].edge_index = object_edge
    data['rooms', 'rooms_to_objects', 'objects'].edge_index = room_object_edge
    
    Hydra_mp3d_data.fill_missing_edge_index(data, edge_types=EDGE_TYPES)
    assert data[('rooms', 'rooms_to_rooms', 'rooms')].num_edges == 2
    assert ('objects', 'objects_to_rooms', 'rooms') in data.edge_index_dict
    assert torch.all(data[('objects', 'objects_to_rooms', 'rooms')].edge_index == \
        data['rooms', 'rooms_to_objects', 'objects'].edge_index.flip([0]))


def test_Hydra_mp3d_dataset(test_data_dir):
    test_json_file1 = test_data_dir / "x8F5xyUWy9e_0_gt_partial_dsg_1447.json"
    test_json_file2 = test_data_dir / "17DRP5sb8fy_0_gt_partial_dsg_1414.json"
    gt_house_file1 = test_data_dir / "x8F5xyUWy9e.house"
    gt_house_file2 = test_data_dir / "17DRP5sb8fy.house"
    if not (os.path.exists(test_json_file1) and os.path.exists(test_json_file2)
        and os.path.exists(gt_house_file1)):
        warnings.warn(UserWarning("test data file missing. -- skip test"))
        return

    # sample data
    data1 = Hydra_mp3d_data(scene_id='x8F5xyUWy9e', trajectory_id=0, num_frames=1447, \
        file_path=str(test_json_file1))
    data1_copy = Hydra_mp3d_data(scene_id='x8F5xyUWy9e', trajectory_id=1, num_frames=1447, \
        file_path=str(test_json_file1))
    data2 = Hydra_mp3d_data(scene_id='17DRP5sb8fy', trajectory_id=0, num_frames=1414, \
        file_path=str(test_json_file1))

    # 1. empty dataset
    dataset = Hydra_mp3d_dataset('train')
    assert dataset.split == 'train'
    assert len(dataset) == 0
    assert dataset.num_scenes() == 0

    # 2. add sample data -- without torch data
    dataset.add_data(data1)
    dataset.add_data(data1_copy)
    dataset.add_data(data2)
    assert dataset.split == 'train'
    assert len(dataset) == 3
    assert dataset.num_scenes() == 2
    assert dataset[1] is None
    assert dataset.get_data(2).is_heterogeneous() is None
    assert dataset.get_data(2).get_data_info()['num_frames'] ==1414

    # 3. clear data
    dataset.clear_dataset()
    assert dataset.split == 'train'
    assert len(dataset) == 0
    assert dataset.num_scenes() == 0

    # 4. add sample data -- with torch data
    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
    else:
        object_feature_converter=hydra_object_feature_converter(
            pytest.colormap_data, pytest.word2vec_model)
        gt_house_info1 = load_mp3d_info(gt_house_file1)
        data1.add_room_labels(gt_house_info1, angle_deg=-90)
        data1.add_object_edges(threshold_near=2.0, threshold_on=1.0, max_near=2.0)
        data1.compute_torch_data(use_heterogeneous=True,
            node_converter=hydra_node_converter(object_feature_converter, lambda i: np.empty(300)))
        gt_house_info2 = load_mp3d_info(gt_house_file2)
        data2.add_room_labels(gt_house_info2, angle_deg=-90)
        data2.add_object_edges(threshold_near=2.0, threshold_on=1.0, max_near=2.0)
        data2.compute_torch_data(use_heterogeneous=True,
            node_converter=hydra_node_converter(object_feature_converter, lambda i: np.empty(300)))
        dataset.add_data(data1)
        dataset.add_data(data2)
        assert dataset.split == 'train'
        assert len(dataset) == 2
        assert dataset.num_scenes() == 2
        assert isinstance(dataset[1], HeteroData)
