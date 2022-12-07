import torch
from torch_geometric.data import (Data, HeteroData)
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
    'a', # bathroom (should have a toilet and a sink)
    'b', # bedroom
    'c', # closet
    'd', # dining room (includes “breakfast rooms” other rooms people mainly eat in)
    'e', # entryway/foyer/lobby (should be the front door, not any door)
    'f', # familyroom (should be a room that a family hangs out in, not any area with couches)
    'g', # garage
    'h', # hallway
    'i', # library (should be room like a library at a university, not an individual study)
    'j', # laundryroom/mudroom (place where people do laundry, etc.)
    'k', # kitchen
    'l', # living room (should be the main “showcase” living room in a house, not any area with couches)
    'm', # meetingroom/conferenceroom
    'n', # lounge (any area where people relax in comfy chairs/couches that is not the family room or living room
    'o', # office (usually for an individual, or a small set of people)
    'p', # porch/terrace/deck/driveway (must be outdoors on ground level)
    'r', # rec/game (should have recreational objects, like pool table, etc.)
    's', # stairs
    't', # toilet (should be a small room with ONLY a toilet)
    'u', # utilityroom/toolroom 
    'v', # tv (must have theater-style seating)
    'w', # workout/gym/exercise
    'x', # outdoor areas containing grass, plants, bushes, trees, etc.
    'y', # balcony (must be outside and must not be on ground floor)
    'z', # other room (it is clearly a room, but the function is not clear)
    'B', # bar
    'C', # classroom
    'D', # dining booth
    'S', # spa/sauna
    'Z', # junk (reflections of mirrors, random points floating in space, etc.)
    '\x15', # Hydra-DSG unlabeled room
]


]
def convert_label_to_y(torch_data, room_labels=ROOM_LABELS, object_labels=OBJECT_LABELS):
    """
    Convert labels
    """
    # converting room labels from unicode to integer
    num_room_labels = len(room_labels)
    room_label_dict = dict(zip(room_labels, range(num_room_labels)))
    room_label_dict['\x15'] = num_room_labels - 1  # '\x15' is a hydra room without gt mp3d label

    # converting object labels from mp3d integer label to filtered integer label
    num_object_labels = len(object_labels)
    object_label_dict = dict(zip(object_labels, range(num_object_labels)))

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
    
    return room_label_dict, object_label_dict
