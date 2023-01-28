from hydra_gnn.preprocess_dsgs import get_spark_dsg, get_room_object_dsg, convert_label_to_y, add_object_connectivity
from neural_tree.construct import generate_htree, add_virtual_nodes_to_htree, nx_htree_to_torch
import os.path
import torch.utils
from torch_geometric.data import HeteroData


EDGE_TYPES = [('objects', 'objects_to_objects', 'objects'),
              ('rooms', 'rooms_to_rooms', 'rooms'),
              ('objects', 'objects_to_rooms', 'rooms'),
              ('rooms', 'rooms_to_objects', 'objects')]


class Hydra_mp3d_data:
    """
    data class that takes in a hydra-mp3d trajectory, converts and stores torch data for training.
    """
    def __init__(self, scene_id, trajectory_id, num_frames, file_path):
        assert os.path.exists(file_path)
        self._scene_id = scene_id
        self._trajectory_id = trajectory_id
        self._num_frames = num_frames
        self._file_path = file_path
        
        # extract complete dsg (for book-keeping) and room-object graph
        dsg = get_spark_dsg()
        self._G = dsg.DynamicSceneGraph.load(file_path)
        dsg.add_bounding_boxes_to_layer(self._G, dsg.DsgLayers.ROOMS)
        self._G_ro = get_room_object_dsg(self._G, verbose=False)

        # place-holder for torch room-object torch graphs
        self._torch_data = None
        # hydra semantic label attribute to training label conversion
        self._object_label_dict = None
        self._room_label_dict = None

    def add_room_labels(self, mp3d_info, angle_deg=-90):
        """add room labels using ground-truth mp3d house segmentation"""
        dsg = get_spark_dsg()
        dsg.mp3d.add_gt_room_label(self._G_ro, mp3d_info, angle_deg=angle_deg)

    def add_object_edges(self, threshold_near=2.0, max_near=2.0, max_on=0.2):
        """add object connectivity to self._G_ro"""
        add_object_connectivity(self._G_ro, threshold_near=threshold_near, max_near=max_near,
            max_on=max_on)

    @staticmethod
    def fill_missing_edge_index(torch_data, edge_types):
        """helper function to fill missing edge types in torch_data to ensure all training data have the same edge_types"""
        for source_type, edge_name, target_type in edge_types:
            if (source_type, edge_name, target_type) in torch_data.edge_index_dict.keys():
                continue

            # missing intra-type edges, fill this type with empty edge_index
            if source_type == target_type:
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch.empty((2, 0), dtype=torch.int64)
            # missing inter-type edges but can find edges in the other direction, fill with flipped edge_index
            elif (target_type, '_'.join(edge_name.split('_')[::-1]), source_type) in \
                torch_data.edge_index_dict.keys():
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch_data[target_type, '_'.join(edge_name.split('_')[::-1]), source_type].edge_index.flip([0])
            # missing inter-type edges without edges in the other direction, fill with empty edge_index
            else:
                torch_data[source_type, edge_name, target_type].edge_index = \
                    torch.empty((2, 0), dtype=torch.int64)

    def compute_torch_data(self, use_heterogeneous: bool, node_converter, object_synonyms=[],
                           room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')], 
                           double_precision=False):
        """compute self._torch data by converting self._G_ro to torch data"""
        # convert room-object dsg to torch graph
        self._torch_data = self._G_ro.to_torch(use_heterogeneous=use_heterogeneous,
            node_converter=node_converter, double_precision=double_precision)
        
        if use_heterogeneous:   # for HeteroData, make sure all edges are there
            self.fill_missing_edge_index(self._torch_data, EDGE_TYPES)
        else:
            self._torch_data.room_mask = self._torch_data['node_masks'][4]
            delattr(self._torch_data, 'node_masks')

        # convert hydra semantic label to torch training label
        self._object_label_dict, self._room_label_dict = \
            convert_label_to_y(self._torch_data, object_synonyms=object_synonyms,
                room_synonyms=room_synonyms)

    def compute_relative_pos(self):
        """remove first three elements (3d pos) in x for each node and fill edge feature with relative pos"""
        if self._torch_data.edge_attr_dict:
            raise Warning("Cannot compute relative pos as edge_attr -- edge_attr is not empty.")

        # remove first three elements in x
        for node_type in self._torch_data.x_dict:
            self._torch_data[node_type].x = self._torch_data[node_type].x[:, 3:]

        # fill edge feature with relative pos
        for edge_type, edge_index in self._torch_data.edge_index_dict.items():
            source, _, target = edge_type
            self._torch_data[edge_type].edge_attr = \
                self._torch_data[target].pos[edge_index[1]] - self._torch_data[source].pos[edge_index[0]]
        
    def to_homogeneous(self):
        if isinstance(self._torch_data, HeteroData):
            self._torch_data = self._torch_data.to_homogeneous(add_edge_type=False)
            self._torch_data.room_mask = (self._torch_data.node_type == 1)
    
    def num_node_features(self):
        if self._torch_data is None:
            return None
        elif self.is_heterogeneous():
            return {node_type: self._torch_data[node_type].num_node_features \
                for node_type in self._torch_data.x_dict}
        else:
            return self._torch_data.num_node_features
    
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


class Hydra_mp3d_htree_data(Hydra_mp3d_data):
    def __init__(self, scene_id, trajectory_id, num_frames, file_path):
        super(Hydra_mp3d_htree_data, self).__init__(
            scene_id, trajectory_id, num_frames, file_path)

    def compute_torch_data(self, use_heterogeneous: bool, node_converter, object_synonyms=[], \
        room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')], double_precision=False,
        remove_room_semantic_feature=True, remove_clique_semantic_feature=True):
        """compute self._torch data by converting self._G_ro to htree in torch data format"""
        if not use_heterogeneous:
            raise NotImplemented

        # convert room-object dsg to torch dsg using parent class method
        Hydra_mp3d_data.compute_torch_data(self, use_heterogeneous=True, node_converter=node_converter, \
            object_synonyms=object_synonyms, room_synonyms=room_synonyms, double_precision=double_precision)
        
        # generate heterogeneous networkx htree and add virtual nodes (for training)
        htree_nx = generate_htree(self._torch_data, verbose=False)
        htree_aug_nx = add_virtual_nodes_to_htree(htree_nx)

        # update self._torch_data by coverting networkx htree to torch data
        self._torch_data = nx_htree_to_torch(htree_aug_nx, double_precision=double_precision)

        # remove redundant zero semantic features (assume it is a 300-dim and at then end of x)
        if remove_room_semantic_feature:
            self._torch_data['room'].x = self._torch_data['room'].x[:, :-300]
            self._torch_data['room_virtual'].x = self._torch_data['room_virtual'].x[:, :-300]
        if remove_clique_semantic_feature:
            self._torch_data['object-room'].x = self._torch_data['object-room'].x[:, :-300]
            self._torch_data['room-room'].x = self._torch_data['room-room'].x[:, :-300]

        # convert hydra semantic label to torch training label
        object_y = [self._object_label_dict[l] for l in self._torch_data['object_virtual'].label.tolist()]
        room_y = [self._room_label_dict[chr(l)] for l in self._torch_data['room_virtual'].label.tolist()]

        self._torch_data['object_virtual'].y = torch.tensor(object_y)
        self._torch_data['room_virtual'].y = torch.tensor(room_y)

    def to_homogeneous(self):
        raise NotImplemented


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
    