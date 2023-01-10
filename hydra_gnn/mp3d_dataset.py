import spark_dsg as dsg
from spark_dsg.mp3d import add_gt_room_label
from hydra_gnn.preprocess_dsgs import get_room_object_dsg, convert_label_to_y, add_object_connectivity
import os.path
import torch.utils
from torch_geometric.data import HeteroData


EDGE_TYPES = [('objects', 'objects_to_objects', 'objects'),
              ('rooms', 'rooms_to_rooms', 'rooms'),
              ('objects', 'objects_to_rooms', 'rooms'),
              ('rooms', 'rooms_to_objects', 'objects')]


class Hydra_mp3d_data:
    def __init__(self, scene_id, trajectory_id, num_frames, file_path) -> None:
        assert os.path.exists(file_path)
        self._scene_id = scene_id
        self._trajectory_id = trajectory_id
        self._num_frames = num_frames
        self._file_path = file_path
        
        # extract complete dsg (for book-keeping) and room-object graph
        self._G = dsg.DynamicSceneGraph.load(file_path)
        dsg.add_bounding_boxes_to_layer(self._G, dsg.DsgLayers.ROOMS)
        self._G_ro = get_room_object_dsg(self._G, verbose=False)

        # place-holder for torch room-object torch graphs
        self._torch_data = None
        # hydra semantic label attribute to training label conversion
        self._object_label_dict = None
        self._room_label_dict = None

    def add_room_labels(self, mp3d_info, angle_deg=-90):
        add_gt_room_label(self._G_ro, mp3d_info, angle_deg=angle_deg)

    def add_object_edges(self, threshold_near=2.0, threshold_on=1.0, max_near=2.0):
        add_object_connectivity(self._G_ro, threshold_near=threshold_near,
            threshold_on=threshold_on, max_near=max_near)

    @staticmethod
    def fill_missing_edge_index(torch_data, edge_types, dtype=torch.int64):
        for source_type, edge_name, target_type in edge_types:
            if (source_type, edge_name, target_type) in torch_data.edge_index_dict.keys():
                continue
            if source_type == target_type:
                print(source_type)
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch.empty((2, 0), dtype=dtype)
            elif (target_type, '_'.join(edge_name.split('_')[::-1]), source_type) in \
                torch_data.edge_index_dict.keys():
                print('here', edge_name)
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch_data[target_type, '_'.join(edge_name.split('_')[::-1]), source_type].edge_index.flip([0])
            else:
                print(edge_name)
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch.empty((2, 0), dtype=dtype)

    def compute_torch_data(self, use_heterogeneous: bool, node_converter, object_synonyms=[],
                           room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')] ):
        # convert room-object dsg to torch graph
        self._torch_data = self._G_ro.to_torch(use_heterogeneous=use_heterogeneous,
            node_converter=node_converter)
        # for HeteroData, make sure all edges are there
        if use_heterogeneous:
            self.fill_missing_edge_index(self._torch_data, EDGE_TYPES)

        # convert hydra semantic label to torch training label
        self._object_label_dict, self._room_label_dict = \
            convert_label_to_y(self._torch_data, object_synonyms=object_synonyms,
                room_synonyms=room_synonyms)
    
    def num_node_features(self):
        if self._torch_data is None:
            return None
        elif self.is_heterogeneous():
            return (self._torch_data['rooms'].num_node_features, self._torch_data['objects'].num_node_features)
        else:
            return (self._torch_data.num_node_features, self._torch_data.num_node_features)
    
    def num_room_labels(self):
        return max(self._room_label_dict.values()) + 1

    def num_object_labels(self):
        return max(self._object_label_dict.values()) + 1

    def is_heterogeneous(self):
        if self._torch_data is None:
            return None
        else:
            return isinstance(self._torch_data, HeteroData)

    def get_label_dict(self):
        return {'objects': self._object_label_dict,
                'rooms': self._room_label_dict}

    def get_data_info(self):
        return {'scene_id': self._scene_id,
                'trajectory_id': self._trajectory_id,
                'num_frames': self._num_frames,
                'file_path': self._file_path}
        
    def get_full_dsg(self):
        return self._G

    def get_room_object_dsg(self):
        return self._G_ro

    def get_torch_data(self):
        return self._torch_data

    def clear_dsg(self):
        self._G = None
        self._G_ro = None


class Hydra_mp3d_dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        assert split in ('train', 'val', 'test'), "Invalid data split."
        self.split = split
        self._data_list = []

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, item):
        return self._data_list[item].get_torch_data()

    def get_data(self, item):
        return self._data_list[item]

    def num_scenes(self):
        return len(set([data.get_data_info()['scene_id'] \
            for data in self._data_list]))

    def add_data(self, data: Hydra_mp3d_data):
        if len(self._data_list) > 0:
            assert data.is_heterogeneous() == \
                self._data_list[-1].is_heterogeneous(), "Invalid torch data type."
        self._data_list.append(data)

    def clear_dataset(self):
        self._data_list = []
    