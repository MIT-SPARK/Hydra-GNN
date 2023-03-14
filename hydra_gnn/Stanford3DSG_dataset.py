from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, EDGE_TYPES, heterogeneous_htree_to_homogeneous
from hydra_gnn.preprocess_dsgs import get_spark_dsg
from neural_tree.construct import generate_htree, add_virtual_nodes_to_htree, nx_htree_to_torch
import os.path
import numpy as np
import torch.utils
import torch.nn.functional
from torch_geometric.data import HeteroData
import random


# room and object labels in Stanford3DSG tiny/verified split
room_semantic_labels = ['bathroom', 'bedroom', 'corridor', 'dining_room', 'home_office', 'kitchen', 'living_room', 'storage_room', 
                        'utility_room', 'lobby', 'playroom', 'staircase', 'closet', 'exercise_room', 'garage'] 
object_semantic_labels = ['apple', 'backpack', 'bed', 'bench', 'bicycle', 'book', 'bottle', 'bowl', 'cell phone', 'chair', 'clock', 'couch', 
                          'cup', 'dining table', 'handbag', 'keyboard', 'knife', 'laptop', 'microwave', 'orange', 'oven', 'potted plant', 
                          'refrigerator', 'remote', 'sink', 'skateboard', 'sports ball', 'suitcase', 'teddy bear', 'tie', 'toaster', 'toilet', 
                          'tv', 'vase', 'wine glass']
num_room_labels = len(room_semantic_labels)
num_object_labels = len(object_semantic_labels)
STANFORD3D_ROOM_SEMANTIC_LABEL_DICT = dict(zip(room_semantic_labels, range(num_room_labels)))
STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT = dict(zip(object_semantic_labels, range(num_object_labels)))
STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['pantry_room'] = STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['kitchen']
STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['television_room'] = STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['living_room']
STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['childs_room'] = STANFORD3D_ROOM_SEMANTIC_LABEL_DICT['playroom']


def Stanford3DSG_object_feature_converter(word2vec_model, object_semantic_dict=STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT):
    """
    Returns a function that takes in label index and returns word2vec semantic embedding vector 
    according to given semantic label to index mapping. Multi-word object semantic label are split by " ".
    """
    return lambda i: np.mean(
        [word2vec_model[s] for s in next(label for label, idx in object_semantic_dict.items() if idx==i).split(" ")], axis=0,)


def Stanford3DSG_room_feature_converter(word2vec_model, room_label_dict=STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT):
    """
    Returns a function that takes in label index and returns word2vec semantic embedding vector 
    according to given semantic label to index mapping. Multi-word room semantic label are split by " ".
    """
    return lambda i: np.mean(
        [word2vec_model[s] for s in next(label for label, idx in room_label_dict.items() if idx==i).split("_")], axis=0,)


class Stanford3DSG_data(Hydra_mp3d_data):
    """
    Data class that converts and stores Stanford 3D Scene Graph for training. This can be initialized with an 
    original .npz scene graph file or a pre-processsed data dictionary.
    """
    def __init__(self, file_path=None, data_dict=None,
                 object_semantic_dict=STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT, 
                 room_semantic_dict=STANFORD3D_ROOM_SEMANTIC_LABEL_DICT):
        if file_path is not None:
            assert os.path.exists(file_path)
            graph_data = np.load(file_path, allow_pickle=True)['output'].item()
            self._scene_name = graph_data['building']['name']
            self._scene_id = graph_data['building']['id']
            self._gibson_split = graph_data['building']['gibson_split']
            self._file_path = file_path
            
            # extract room-object graph
            self._G_ro = self.get_room_object_dsg_from_gt(graph_data, 
                                                          object_semantic_dict=object_semantic_dict,
                                                          room_semantic_dict=room_semantic_dict)
        else:
            self._scene_name = 'N/A'
            self._scene_id = 'N/A'
            self._gibson_split = 'N/A'
            self._file_path = 'N/A'

            # extract room-object graph
            self._G_ro = self.get_room_object_dsg_from_preprocessed_data(data_dict)

        # place-holder for torch room-object torch graphs
        self._torch_data = None
        # semantic label to training label index conversion
        self._object_semantic_dict = object_semantic_dict
        self._room_semantic_dict = room_semantic_dict
    
    @staticmethod
    def get_room_object_dsg_from_gt(graph_data, object_semantic_dict=STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT,
                                    room_semantic_dict=STANFORD3D_ROOM_SEMANTIC_LABEL_DICT):
        # drop object if 
        #   (1) object has no parent room; or 
        #   (2) object label not in object_label_dict.keys(); or 
        #   (3) parent room label not in room_label_dict.keys()
        # drop room if
        #   (1) room label not in room_label_dict.keys(); or
        #   (2) room does not contain objects

        dsg = get_spark_dsg()
        G_ro = dsg.DynamicSceneGraph()
        for key, object_data in graph_data['object'].items():
            if object_data['parent_room'] is None:
                continue
            if object_data['class_'] not in object_semantic_dict.keys():
                continue
            if graph_data['room'][object_data['parent_room']]['scene_category'] not in room_semantic_dict.keys():
                continue

            assert key == object_data['id']
            object_id = dsg._dsg_bindings.NodeSymbol("O", object_data['id'])
            object_attr = dsg._dsg_bindings.ObjectNodeAttributes()
            object_attr.name = str(object_id)
            object_attr.position = object_data['location']
            object_attr.semantic_label = object_semantic_dict[object_data['class_']]
            object_attr.bounding_box = dsg._dsg_bindings.BoundingBox(object_attr.position - object_data['size']/2, 
                                                                    object_attr.position + object_data['size']/2)
            object_attr.bounding_box.world_P_center = object_attr.position
            G_ro.add_node(dsg.DsgLayers.OBJECTS, object_id.value, object_attr)
            
            # add room node if not already in G_ro
            room_id = dsg._dsg_bindings.NodeSymbol("R", object_data['parent_room'])
            if not G_ro.has_node(room_id.value):
                room_data = graph_data['room'][object_data['parent_room']]
                room_attr = dsg._dsg_bindings.RoomNodeAttributes()
                room_attr.name = str(room_id)
                room_attr.position = room_data['location']
                room_attr.semantic_label = room_semantic_dict[room_data['scene_category']]
                room_attr.bounding_box = dsg._dsg_bindings.BoundingBox(room_attr.position - room_data['size']/2, 
                                                                       room_attr.position + room_data['size']/2)
                room_attr.bounding_box.world_P_center = room_attr.position
                G_ro.add_node(dsg.DsgLayers.ROOMS, room_id.value, room_attr)

            # insert room-object edge
            G_ro.insert_edge(object_id.value, room_id.value)
        return G_ro

    @staticmethod
    def get_room_object_dsg_from_preprocessed_data(data_dict):
        # assert there is only one room node and it is node 0 in the input data_dict
        assert data_dict['room_mask'][0]
        assert data_dict['room_mask'].sum().item() == 1

        # parse data_dict
        x_room = data_dict['x'][data_dict['room_mask']][0]
        x_objects = data_dict['x'][~data_dict['room_mask']]
        y_room = data_dict['y'][data_dict['room_mask']][0]
        y_objects = data_dict['y'][~data_dict['room_mask']]
        edge_list = data_dict['edge_index'].T.tolist()

        dsg = get_spark_dsg()
        G_ro = dsg.DynamicSceneGraph()

        # add room node
        room_id = dsg._dsg_bindings.NodeSymbol("R", 0)
        room_attr = dsg._dsg_bindings.RoomNodeAttributes()
        room_attr.name = '0'
        room_attr.position = x_room[:3]
        room_attr.semantic_label = y_room
        room_attr.bounding_box = dsg._dsg_bindings.BoundingBox(room_attr.position - x_room[3:]/2, 
                                                            room_attr.position + x_room[3:]/2)
        room_attr.bounding_box.world_P_center = room_attr.position
        G_ro.add_node(dsg.DsgLayers.ROOMS, room_id.value, room_attr)

        # add object nodes
        num_objects = len(y_objects)
        for i in range(num_objects):
            object_id = dsg._dsg_bindings.NodeSymbol("O", i + 1)
            object_attr = dsg._dsg_bindings.ObjectNodeAttributes()
            object_attr.name = str(i + 1)
            object_attr.position = x_objects[i, :3]
            object_attr.semantic_label = y_objects[i]
            object_attr.bounding_box = dsg._dsg_bindings.BoundingBox(object_attr.position - x_objects[i, 3:]/2, 
                                                                    object_attr.position + x_objects[i, 3:]/2)
            object_attr.bounding_box.world_P_center = object_attr.position
            G_ro.add_node(dsg.DsgLayers.OBJECTS, object_id.value, object_attr)
            assert G_ro.insert_edge(object_id.value, room_id.value)

        # add object edges
        for i, j in edge_list:
            if i == 0 or j == 0:
                continue
            object_id_i = dsg._dsg_bindings.NodeSymbol("O", i)
            object_id_j = dsg._dsg_bindings.NodeSymbol("O", j)
            G_ro.insert_edge(object_id_i.value, object_id_j.value)

        return G_ro
    
    def add_dsg_room_labels(self, **kwargs):
        pass

    def compute_torch_data(self, use_heterogeneous: bool, node_converter, double_precision=False):
        """compute self._torch data by converting self._G_ro to torch data"""
        # convert room-object dsg to torch graph
        self._torch_data = self._G_ro.to_torch(use_heterogeneous=use_heterogeneous,
                                               node_converter=node_converter,
                                               double_precision=double_precision)
        
        if use_heterogeneous:   # for HeteroData, make sure all edges are there
            self.fill_missing_edge_index(self._torch_data, EDGE_TYPES)
            # training label y is the same as label
            for node_type in self._torch_data.x_dict:
                self._torch_data[node_type].y = self._torch_data[node_type].label.long()
                # delattr(self._torch_data[node_type], 'label')
            assert self._torch_data['rooms'].num_node_features == self._torch_data['objects'].num_node_features
        else:
            self._torch_data.room_mask = self._torch_data['node_masks'][4]
            delattr(self._torch_data, 'node_masks')
            # training label y is the same as label
            self._torch_data.y = self._torch_data.label.long()
            # delattr(self._torch_data, 'label')
    
    def num_room_labels(self):
        return max(self._room_semantic_dict.values()) + 1

    def num_object_labels(self):
        return max(self._object_semantic_dict.values()) + 1
    
    def get_label_dict(self):
        return {'objects': self._object_semantic_dict,
                'rooms': self._room_semantic_dict}
    
    def get_data_info(self):
        return {'scene_id': self._scene_id,
                'scene_name': self._scene_name,
                'gibson_split': self._gibson_split,
                'file_path': self._file_path}
        
    def get_full_dsg(self):
        return None
    
    def clear_dsg(self):
        self._G_ro = None


class Stanford3DSG_htree_data(Stanford3DSG_data):
    def __init__(self, file_path=None, data_dict=None,
                 object_semantic_dict=STANFORD3D_OBJECT_SEMANTIC_LABEL_DICT, 
                 room_semantic_dict=STANFORD3D_ROOM_SEMANTIC_LABEL_DICT):
        super(Stanford3DSG_htree_data, self).__init__(
            file_path, data_dict, object_semantic_dict, room_semantic_dict)

    def compute_torch_data(self, use_heterogeneous: bool, node_converter, double_precision=False):
        """compute self._torch data by converting self._G_ro to htree in torch data format"""
        # convert room-object dsg to torch dsg using parent class method
        Stanford3DSG_data.compute_torch_data(self, use_heterogeneous=True, node_converter=node_converter, \
            double_precision=double_precision)
        
        # generate heterogeneous networkx htree and add virtual nodes (for training)
        htree_nx = generate_htree(self._torch_data, verbose=False)
        htree_aug_nx = add_virtual_nodes_to_htree(htree_nx)

        # update self._torch_data by coverting networkx htree to torch data
        self._torch_data = nx_htree_to_torch(htree_aug_nx, double_precision=double_precision)

        # copy label attributes to y
        self._torch_data['object_virtual'].y = self._torch_data['object_virtual'].label.long()
        self._torch_data['room_virtual'].y = self._torch_data['room_virtual'].label.long()
        
        if not use_heterogeneous:
            self.to_homogeneous()
    
    def num_graph_nodes(self):
        if self.is_heterogeneous():
            return self._torch_data['room_virtual'].num_nodes + self._torch_data['object_virtual'].num_nodes
        else:
            return self._torch_data.room_mask.sum().item() + self._torch_data.object_mask.sum().item()
    
    def to_homogeneous(self):
        if isinstance(self._torch_data, HeteroData):
            self._torch_data = heterogeneous_htree_to_homogeneous(self._torch_data)


class Stanford3DSG_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self._data_list = []

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, item):
        return self._data_list[item].get_torch_data()

    def get_data(self, item):
        return self._data_list[item]
    
    def data_type(self):
        if isinstance(self._data_list[0], Stanford3DSG_htree_data):
            if self._data_list[0].is_heterogeneous():
                return 'heterogeneous_htree'
            else:
                return 'homogeneous_htree'
        else:
            if self._data_list[0].is_heterogeneous():
                return 'heterogeneous'
            else:
                return 'homogeneous'

    def num_scenes(self):
        return len(set([data.get_data_info()['scene_id'] \
            for data in self._data_list]))

    def add_data(self, data: Stanford3DSG_data):
        if len(self._data_list) > 0:
            assert data.is_heterogeneous() == \
                self._data_list[-1].is_heterogeneous(), "Invalid torch data type."
        self._data_list.append(data)

    def generate_node_split(self, train_node_ratio, val_node_ratio, test_node_ratio, seed=0):
        random.seed(seed)
        assert train_node_ratio > 0
        assert val_node_ratio >= 0
        if test_node_ratio is None:
            test_node_ratio = 1 - train_node_ratio - val_node_ratio
        assert test_node_ratio >= 0
        assert train_node_ratio + val_node_ratio + test_node_ratio <= 1

        # find number of nodes in each split
        total_num_nodes = sum(data.num_graph_nodes() for data in self._data_list)
        num_train_nodes = round(total_num_nodes*train_node_ratio)
        num_val_nodes = round(total_num_nodes*val_node_ratio)
        if train_node_ratio + val_node_ratio + test_node_ratio == 1:
            num_test_nodes = total_num_nodes - num_train_nodes - num_val_nodes
            random_assignment_list = [1] * num_train_nodes + [2] * num_val_nodes + [3] * num_test_nodes
        else:
            num_test_nodes = round(total_num_nodes*test_node_ratio)
            num_unassigned_nodes = total_num_nodes - num_train_nodes - num_val_nodes - num_test_nodes
            random_assignment_list = [1] * num_train_nodes + [2] * num_val_nodes + [3] * num_test_nodes + [0] * num_unassigned_nodes
        
        # randomly permute a list of 0's (unassinged), 1's (train), 2's (val), and 3's (test)
        random.shuffle(random_assignment_list)

        # add train/val/test masks to torch_data
        data_type = self.data_type()
        for data in self._data_list:
            # get split assignment
            num_graph_nodes = data.num_graph_nodes()
            node_split = torch.tensor(random_assignment_list[:num_graph_nodes], dtype=torch.int)
            random_assignment_list = random_assignment_list[num_graph_nodes:]

            if data_type == 'homogeneous':
                data.get_torch_data().train_mask = (node_split == 1)
                data.get_torch_data().val_mask = (node_split == 2)
                data.get_torch_data().test_mask = (node_split == 3)
            elif data_type == 'heterogeneous':
                num_objects = data.get_torch_data()['objects'].num_nodes
                data.get_torch_data()['objects'].train_mask = (node_split[:num_objects] == 1)
                data.get_torch_data()['rooms'].train_mask = (node_split[num_objects:] == 1)
                data.get_torch_data()['objects'].val_mask = (node_split[:num_objects] == 2)
                data.get_torch_data()['rooms'].val_mask = (node_split[num_objects:] == 2)
                data.get_torch_data()['objects'].test_mask = (node_split[:num_objects] == 3)
                data.get_torch_data()['rooms'].test_mask = (node_split[num_objects:] == 3)
            elif data_type == 'homogeneous_htree':
                room_mask = data.get_torch_data().room_mask
                object_mask = data.get_torch_data().object_mask
                num_objects = object_mask.sum().item()
                data.get_torch_data().train_mask = torch.zeros(data.get_torch_data().num_nodes, dtype=torch.bool)
                data.get_torch_data().train_mask[object_mask] = (node_split == 1)[:num_objects]
                data.get_torch_data().train_mask[room_mask] = (node_split == 1)[num_objects:]
                data.get_torch_data().val_mask = torch.zeros(data.get_torch_data().num_nodes, dtype=torch.bool)
                data.get_torch_data().val_mask[object_mask] = (node_split == 2)[:num_objects]
                data.get_torch_data().val_mask[room_mask] = (node_split == 2)[num_objects:]
                data.get_torch_data().test_mask = torch.zeros(data.get_torch_data().num_nodes, dtype=torch.bool)
                data.get_torch_data().test_mask[object_mask] = (node_split == 3)[:num_objects]
                data.get_torch_data().test_mask[room_mask] = (node_split == 3)[num_objects:]
            elif data_type == 'heterogeneous_htree':
                num_objects = data.get_torch_data()['object_virtual'].num_nodes
                data.get_torch_data()['object_virtual'].train_mask = (node_split[:num_objects] == 1)
                data.get_torch_data()['room_virtual'].train_mask = (node_split[num_objects:] == 1)
                data.get_torch_data()['object_virtual'].val_mask = (node_split[:num_objects] == 2)
                data.get_torch_data()['room_virtual'].val_mask = (node_split[num_objects:] == 2)
                data.get_torch_data()['object_virtual'].test_mask = (node_split[:num_objects] == 3)
                data.get_torch_data()['room_virtual'].test_mask = (node_split[num_objects:] == 3)
            else:
                raise NotImplemented
        
        assert len(random_assignment_list) == 0

    def clear_dataset(self):
        self._data_list = []
    
