import warnings
import numpy as np
import torch
from torch_geometric.data import (Data, HeteroData)
import spark_dsg as dsg

OBJECT_LABELS = [
    3,  # chair
    5,  # table
    6,  # picture
    7,  # cabinet
    8,  # cushion
    10,  # sofa
    11,  # bed
    12,  # curtain
    13,  # chest of drawers
    14,  # plant
    15,  # sink
    18,  # toiled
    19,  # stool
    20,  # towel
    21,  # mirror
    22,  # tv_monitor
    23,  # shower
    25,  # bathtub
    27,  # fireplace
    28,  # lighting
    30,  # shelving
    31,  # blinds
    32,  # seating
    33,  # board_panel
    34,  # furniture
    35,  # clothes
    36,  # objects
    37,  # misc
]

ROOM_LABELS = [
    'a',  # bathroom (should have a toilet and a sink)
    'b',  # bedroom
    'c',  # closet
    'd',  # dining room (includes “breakfast rooms” other rooms people mainly eat in)
    'e',  # entryway/foyer/lobby (should be the front door, not any door)
    'f',  # familyroom (should be a room that a family hangs out in, not any area with couches)
    'g',  # garage
    'h',  # hallway
    'i',  # library (should be room like a library at a university, not an individual study)
    'j',  # laundryroom/mudroom (place where people do laundry, etc.)
    'k',  # kitchen
    'l',  # living room (should be the main “showcase” living room in a house, not any area with couches)
    'm',  # meetingroom/conferenceroom
    'n',  # lounge (any area where people relax in comfy chairs/couches that is not the family room or living room
    'o',  # office (usually for an individual, or a small set of people)
    'p',  # porch/terrace/deck/driveway (must be outdoors on ground level)
    'r',  # rec/game (should have recreational objects, like pool table, etc.)
    's',  # stairs
    't',  # toilet (should be a small room with ONLY a toilet)
    'u',  # utilityroom/toolroom
    'v',  # tv (must have theater-style seating)
    'w',  # workout/gym/exercise
    'x',  # outdoor areas containing grass, plants, bushes, trees, etc.
    'y',  # balcony (must be outside and must not be on ground floor)
    'z',  # other room (it is clearly a room, but the function is not clear)
    'B',  # bar
    'C',  # classroom
    'D',  # dining booth
    'S',  # spa/sauna
    'Z',  # junk (reflections of mirrors, random points floating in space, etc.)
    '\x15',  # Hydra-DSG unlabeled room
]


def _is_on(G, n1, n2, threshold_on=1.0):
    """
    Check whether n1 is "on" n2 or n2 is "on" n1
    Requires that the n1 center is inside n2 on xy plane, and n1 is above
    n2 on z-axis within a threshold (or vice-versa).
    """
    pos1 = G.get_position(n1.id.value)
    pos2 = G.get_position(n2.id.value)
    size1 = _get_size(n1)
    size2 = _get_size(n2)

    xy_dist = np.abs(pos1[0:2] - pos2[0:2])
    z_dist = np.abs(pos1[2] - pos2[2])
    n1_above_n2 = pos1[2] > pos2[2]
    new_thresh = threshold_on * (size1[2] + size2[2]) / 2

    if all(xy_dist <= size2[0:2] / 2) and n1_above_n2 and z_dist <= new_thresh:
        return True
    elif all(xy_dist <= size1[0:2] / 2) and not n1_above_n2 and z_dist <= new_thresh:
        return True
    else:
        return False


def _is_above(G, n1, n2, threshold_near=2.0, threshold_on=1.0):
    """
    Check whether n1 is "above" n2 or n2 is "above" n1
    Requires that the n1 center and n2 are nearby on the xy plane, and n1 is above
    n2 on z-axis by an amount greater than a provided threshold.
    """
    pos1 = G.get_position(n1.id.value)
    pos2 = G.get_position(n2.id.value)
    size1 = _get_size(n1)
    size2 = _get_size(n2)
    near_thresh = (size1[0:2] + size2[0:2]) / 2.0 * threshold_near

    xy_dist = np.abs(pos1[0:2] - pos2[0:2])
    z_dist = np.abs(pos1[2] - pos2[2])
    n1_above_n2 = pos1[2] > pos2[2]
    dist_thresh = threshold_on * (size1[2] + size2[2]) / 2

    if all(xy_dist <= near_thresh):
        if n1_above_n2 and z_dist > dist_thresh:
            return True
        if not n1_above_n2 and z_dist > dist_thresh:
            return True

    return False


def _is_under(G, n1, n2):
    """
    Check whether n1 is "under" n2 or n2 is "under" n1
    Requires that either n1 or n2 is inside the other node on the xy place and
    that the positions on the z-axis are distinct.
    """
    pos1 = G.get_position(n1.id.value)
    pos2 = G.get_position(n2.id.value)
    size1 = _get_size(n1)
    size2 = _get_size(n2)

    xy_dist = np.abs(pos1[0:2] - pos2[0:2])

    if all(xy_dist <= size1[0:2] / 2) or all(xy_dist <= size2[0:2] / 2):
        if pos1[2] < pos2[2]:
            return True
        if pos2[2] < pos1[2]:
            return True

    return False


def _is_near(G, n1, n2, threshold_near=2.0, max_near=2.0):
    """
    Check whether n1 is "near" n2 or n2 is "near" n1
    Requires that either n1 or n2 is inside the other node on the xy place and
    that the positions on the z-axis are distinct.
    """
    pos1 = G.get_position(n1.id.value)
    pos2 = G.get_position(n2.id.value)
    size1 = _get_size(n1)
    size2 = _get_size(n2)

    avg_size = (size1 + size2) / 2.0
    near_thresh = avg_size * threshold_near

    dist = np.abs(pos1 - pos2)

    # [LocatedNear]
    if all(dist <= near_thresh) and all(dist - avg_size <= max_near * np.ones(3)):
        return True

    return False


def _get_size(node):
    return node.attributes.bounding_box.max - node.attributes.bounding_box.min


def _dist(G, n1, n2):
    return np.linalg.norm(G.get_position(n1) - G.get_position(n2))


def add_object_connectivity(G, threshold_near=2.0, threshold_on=1.0, max_near=2.0):
    room_to_objects = dict()
    for node in G.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        room_id = node.get_parent()
        if room_id is None:
            print(f"skipping invalid object {node.id}")
            continue

        if room_id not in room_to_objects:
            room_to_objects[room_id] = [node]
            continue

        cmp_nodes = room_to_objects[room_id]
        for cmp_node in cmp_nodes:
            is_on = _is_on(G, node, cmp_node, threshold_on=threshold_on)
            is_above = _is_above(
                G,
                node,
                cmp_node,
                threshold_near=threshold_near,
                threshold_on=threshold_on,
            )
            is_under = _is_under(G, node, cmp_node)
            is_near = _is_near(
                G, node, cmp_node, threshold_near=threshold_near, max_near=max_near
            )

            if is_on or is_above or is_under or is_near:
                # TODO(nathan) consider getting direction
                assert G.insert_edge(node.id.value, cmp_node.id.value)

        room_to_objects[room_id].append(node)


def get_room_object_dsg(G, verbose=False):
    """Create a room-object DSG by copying and connecting room and object nodes from the input DSG."""

    # create an empty DSG and copy all room nodes
    G_room_object = dsg.DynamicSceneGraph()
    sibling_map = {}
    for room_node in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
        G_room_object.add_node(
            room_node.layer, room_node.id.value, room_node.attributes)
        sibling_map[room_node.id.value] = room_node.siblings()

    # add edges between rooms
    for room_id, sibling_ids in sibling_map.items():
        for i in sibling_ids:
            G_room_object.insert_edge(room_id, i)

    invalid_objects = []
    for object_node in G.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        if not object_node.has_parent():
            invalid_objects.append(object_node)
            warnings.warn(
                f"{object_node.id} has no parent node in the input DSG.")
            continue

        # insert edge through object -> place -> room edges in G
        parent_place = G.get_node(object_node.get_parent())
        if parent_place.has_parent():
            G_room_object.add_node(
                object_node.layer, object_node.id.value, object_node.attributes)
            G_room_object.insert_edge(
                object_node.id.value, parent_place.get_parent())
            if verbose:
                print(f"{object_node.id} - direct edge")
            continue

        # insert edge through object -> place - (neighboring place) -> room edges in G
        neighboring_places = parent_place.siblings()
        for i in neighboring_places:
            neighboring_place = G.get_node(i)
            if neighboring_place.has_parent():
                G_room_object.add_node(
                    object_node.layer, object_node.id.value, object_node.attributes)
                G_room_object.insert_edge(
                    object_node.id.value, neighboring_place.get_parent())
                if verbose:
                    print(f"{object_node.id} - indirect edge")
                break
        else:
            invalid_objects.append(object_node)
            if verbose:
                print(f"Drop {object_node.id}.")

    return G_room_object


def _get_label_dict(labels, synonyms=None):
    """Get mapping from (Hydra) labels to integer label index while grouping synonym labels. """

    if synonyms is None or len(synonyms) == 0:
        return dict(zip(labels, range(len(labels))))

        all_labels_to_combine = [l for syn in synonyms for l in syn]
        num_labels = len(labels) - len(all_labels_to_combine) + len(synonyms)

        # label to index mapping - unique labels
        label_dict = dict(zip([l for l in labels if l not in all_labels_to_combine] + \
                              [syn[0] for syn in synonyms], range(num_labels)))

        # label to index mapping - synonym labels
        label_index_offset = len(labels) - len(all_labels_to_combine)
        for i, syn in enumerate(synonyms):
            for l in syn:
                label_dict[l] = i + label_index_offset

    return label_dict


def convert_label_to_y(torch_data, object_labels=OBJECT_LABELS, room_labels=ROOM_LABELS,
                       object_synonyms=[], room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')]):
    """
    Convert labels
    """

    # converting object labels from mp3d integer label to filtered integer label
    object_label_dict = _get_label_dict(object_labels, object_synonyms)

    # converting room labels from unicode to integer
    room_label_dict = _get_label_dict(room_labels, room_synonyms)

    # add training label as y attribute to torch data
    if isinstance(torch_data, Data):
        assert 2 in torch_data.node_masks.keys()
        assert 4 in torch_data.node_masks.keys()

        object_y = [object_label_dict[l] for l in torch_data.label[torch_data.node_masks[2]].tolist()]
        room_y = [room_label_dict[chr(l)] for l in torch_data.label[torch_data.node_masks[4]].tolist()]

        torch_data.y = torch.zeros(torch_data.label.shape, dtype=torch.int64)
        torch_data.y[torch_data.node_masks[2]] = torch.tensor(object_y)
        torch_data.y[torch_data.node_masks[4]] = torch.tensor(room_y)

    elif isinstance(torch_data, HeteroData):
        assert len(torch_data.node_types) == 2
        assert 'objects' in torch_data.node_types
        assert 'rooms' in torch_data.node_types

        object_y = [object_label_dict[l] for l in torch_data['objects'].label.tolist()]
        room_y = [room_label_dict[chr(l)] for l in torch_data['rooms'].label.tolist()]

        torch_data['objects'].y = torch.tensor(object_y)
        torch_data['rooms'].y = torch.tensor(room_y)

    else:
        raise NotImplemented

    return object_label_dict, room_label_dict