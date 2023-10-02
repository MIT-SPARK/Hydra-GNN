"""Test graph pre-processing."""
from hydra_gnn.preprocess_dsgs import (
    get_room_object_dsg,
    add_object_connectivity,
    dsg_node_converter,
    hydra_object_feature_converter,
    convert_label_to_y,
    OBJECT_LABELS,
    ROOM_LABELS,
)
from spark_dsg.mp3d import load_mp3d_info, add_gt_room_label
import spark_dsg as dsg
import numpy as np
import warnings
import os.path
import pytest

tol = 1e-5


def test_get_room_object_dst(test_data_dir, tol=tol):
    """Test that room-object graphs make sense."""
    test_json_file = test_data_dir / "x8F5xyUWy9e_0_gt_partial_dsg_1447.json"
    if not os.path.exists(test_json_file):
        warnings.warn(UserWarning("test data file missing. -- skip test"))
        return

    # read test hydra scene graph and construct room bounding box
    G = dsg.DynamicSceneGraph.load(str(test_json_file))
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)

    # extract room-object graph
    G_ro = get_room_object_dsg(G, verbose=False)

    # check number of nodes and edges
    assert (
        G.get_layer(dsg.DsgLayers.ROOMS).num_nodes()
        == G_ro.get_layer(dsg.DsgLayers.ROOMS).num_nodes()
    )
    assert (
        G.get_layer(dsg.DsgLayers.ROOMS).num_edges()
        == G_ro.get_layer(dsg.DsgLayers.ROOMS).num_edges()
    )
    assert (
        G.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
        >= G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
    )
    assert G_ro.get_layer(dsg.DsgLayers.OBJECTS).num_edges() == 0

    # check node attributes
    def _check_node_attributes(node):
        assert (
            G_ro.get_node(node.id.value).attributes.semantic_label
            == G.get_node(node.id.value).attributes.semantic_label
        )
        assert (
            np.linalg.norm(
                G_ro.get_node(node.id.value).attributes.position
                - G.get_node(node.id.value).attributes.position
            )
            < tol
        )
        assert (
            np.linalg.norm(
                G_ro.get_node(node.id.value).attributes.bounding_box.min
                - G.get_node(node.id.value).attributes.bounding_box.min
            )
            < tol
        )
        assert (
            np.linalg.norm(
                G_ro.get_node(node.id.value).attributes.bounding_box.max
                - G.get_node(node.id.value).attributes.bounding_box.max
            )
            < tol
        )

    # check position is inside of a room
    def _is_inside(pos, room_node):
        bbx_min = G_ro.get_node(room_node.id.value).attributes.bounding_box.min
        bbx_max = G_ro.get_node(room_node.id.value).attributes.bounding_box.max
        assert np.all(pos >= bbx_min)
        assert np.all(pos <= bbx_max)

    for room_node in G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes:
        _check_node_attributes(room_node)
    for object_node in G_ro.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        _check_node_attributes(object_node)
        if G.get_node(object_node.id.value).has_parent():
            place_node_in_G = G.get_node(G.get_node(object_node.id.value).get_parent())
            _is_inside(
                place_node_in_G.attributes.position,
                G_ro.get_node(object_node.get_parent()),
            )


def test_hydra_object_feature_converter(tol=tol):
    """Test that converted object features make sense."""
    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
        return
    else:
        colormap_data = pytest.colormap_data
        word2vec_model = pytest.word2vec_model

    # feature converter should convert hydra integer label to the corresponding 300-dim
    # word2vec feature vector
    feature_converter = hydra_object_feature_converter(colormap_data, word2vec_model)
    assert np.linalg.norm(feature_converter(3) - word2vec_model["chair"]) < tol
    assert (
        np.linalg.norm(
            feature_converter(13)
            - (word2vec_model["chest"] + word2vec_model["drawers"]) / 2
        )
        < tol
    )
    assert (
        np.linalg.norm(
            feature_converter(22)
            - (word2vec_model["tv"] + word2vec_model["monitor"]) / 2
        )
        < tol
    )
    assert np.linalg.norm(feature_converter(36) - word2vec_model["clothes"]) < tol


def test_full_torch_feature_conversion(test_data_dir, tol=tol):
    """Test that features get constructed correctly."""
    test_json_file = test_data_dir / "x8F5xyUWy9e_0_gt_partial_dsg_1447.json"
    if not os.path.exists(test_json_file):
        warnings.warn(UserWarning("test data file missing. -- skip test"))
        return

    # read test hydra scene graph and extract room-object graph
    G = dsg.DynamicSceneGraph.load(str(test_json_file))
    G_ro = get_room_object_dsg(G, verbose=False)
    add_object_connectivity(G_ro, threshold_near=2.0, max_on=0.2, max_near=2.0)

    # setup 1: no semantic feature
    data_1 = G_ro.to_torch(
        use_heterogeneous=True,
        node_converter=dsg_node_converter(
            object_feature_converter=lambda i: np.empty(0),
            room_feature_converter=lambda i: np.empty(0),
        ),
    )

    data_2 = G_ro.to_torch(
        use_heterogeneous=False,
        node_converter=dsg_node_converter(
            object_feature_converter=lambda i: np.empty(0),
            room_feature_converter=lambda i: np.empty(0),
        ),
    )

    for i, node in enumerate(G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes):
        assert np.linalg.norm(node.attributes.position) > tol  # position is non-zero
        assert (
            np.linalg.norm(
                np.array(data_1["rooms"].x[i, 0:3]) - node.attributes.position
            )
            < tol
        )
        assert (
            np.linalg.norm(
                np.array(data_2.x[data_2.node_masks[4], :][i, 0:3])
                - node.attributes.position
            )
            < tol
        )

    for i, node in enumerate(G_ro.get_layer(dsg.DsgLayers.OBJECTS).nodes):
        size = node.attributes.bounding_box.max - node.attributes.bounding_box.min
        assert np.all(size > 0)  # size is positive
        assert np.linalg.norm(np.array(data_1["objects"].x[i, 3:6]) - size) < tol
        assert (
            np.linalg.norm(np.array(data_2.x[data_2.node_masks[2], :][i, 3:6]) - size)
            < tol
        )

    # setup 2: use word2vec object semantic feature
    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
        return
    else:
        colormap_data = pytest.colormap_data
        word2vec_model = pytest.word2vec_model

    data_3 = G_ro.to_torch(
        use_heterogeneous=True,
        node_converter=dsg_node_converter(
            object_feature_converter=hydra_object_feature_converter(
                colormap_data, word2vec_model
            ),
            room_feature_converter=lambda i: np.empty(0),
        ),
    )

    data_4 = G_ro.to_torch(
        use_heterogeneous=False,
        node_converter=dsg_node_converter(
            object_feature_converter=hydra_object_feature_converter(
                colormap_data, word2vec_model
            ),
            room_feature_converter=lambda i: np.empty(300),
        ),
    )

    assert data_3["rooms"].x.shape[1] == 6
    assert data_3["objects"].x.shape[1] == 306
    assert data_4.x.shape[1] == 306

    for i, node in enumerate(G_ro.get_layer(dsg.DsgLayers.ROOMS).nodes):
        assert np.linalg.norm(node.attributes.position) > tol  # position is non-zero
        assert (
            np.linalg.norm(
                np.array(data_3["rooms"].x[i, 0:3]) - node.attributes.position
            )
            < tol
        )
        assert (
            np.linalg.norm(
                np.array(data_4.x[data_4.node_masks[4], :][i, 0:3])
                - node.attributes.position
            )
            < tol
        )
        assert np.linalg.norm(np.array(data_4.x[data_4.node_masks[4], :][i, 6:0])) < tol

    feature_converter = hydra_object_feature_converter(colormap_data, word2vec_model)
    for i, node in enumerate(G_ro.get_layer(dsg.DsgLayers.OBJECTS).nodes):
        size = node.attributes.bounding_box.max - node.attributes.bounding_box.min
        assert np.all(size > 0)  # size is positive
        assert np.linalg.norm(np.array(data_3["objects"].x[i, 3:6]) - size) < tol
        assert (
            np.linalg.norm(np.array(data_4.x[data_4.node_masks[2], :][i, 3:6]) - size)
            < tol
        )

        feature = feature_converter(node.attributes.semantic_label)
        assert np.linalg.norm(np.array(data_3["objects"].x[i, 6:]) - feature) < tol


def test_convert_label_to_y(test_data_dir):
    """Test that ground-truth labels get stored correctly."""
    test_json_file = test_data_dir / "x8F5xyUWy9e_0_gt_partial_dsg_1447.json"
    gt_house_file = test_data_dir / "x8F5xyUWy9e.house"
    if not (os.path.exists(test_json_file) and os.path.exists(gt_house_file)):
        warnings.warn(UserWarning("test data file missing. -- skip test"))
        return

    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
        return
    else:
        colormap_data = pytest.colormap_data
        word2vec_model = pytest.word2vec_model

    # read test hydra scene graph, add room label using gt segmentation,
    # extract room-object graph
    G = dsg.DynamicSceneGraph.load(str(test_json_file))
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
    gt_house_info = load_mp3d_info(gt_house_file)
    add_gt_room_label(G, gt_house_info, angle_deg=-90)
    G_ro = get_room_object_dsg(G, verbose=False)

    # expected results
    object_synonyms = []
    room_synonyms = [("a", "t"), ("z", "Z", "x", "p", "\x15")]
    object_label_dict_exp = dict(zip(OBJECT_LABELS, range(len(OBJECT_LABELS))))
    individually_mapped_room = [
        label
        for label in ROOM_LABELS
        if label not in ["a", "t", "z", "Z", "x", "p", "\x15"]
    ]
    room_label_dict_exp = dict(
        zip(
            individually_mapped_room + ["a", "t", "z", "Z", "x", "p", "\x15"],
            list(range(len(individually_mapped_room)))
            + [len(individually_mapped_room)] * 2
            + [len(individually_mapped_room) + 1] * 5,
        )
    )

    def _check_dict(dict1, dict_2):
        assert len(dict1) == len(dict_2)
        assert all((dict1.get(k) == v for k, v in dict_2.items()))

    room_y_exp = [12, 23, 3, 23, 23]
    object_y_exp = [
        0,
        3,
        22,
        13,
        0,
        1,
        9,
        27,
        9,
        10,
        6,
        0,
        0,
        20,
        9,
        27,
        6,
        20,
        3,
        16,
        10,
        14,
        15,
        19,
        26,
        27,
        0,
        0,
        13,
        22,
        1,
        9,
        22,
        6,
        10,
        27,
        10,
        10,
        14,
        19,
        0,
        0,
        0,
        25,
        16,
        3,
        6,
        1,
        0,
        27,
        25,
        26,
        15,
        25,
        3,
        1,
        22,
        22,
        1,
        10,
        22,
        10,
    ]

    def _check_list(list1, list2):
        assert len(list1) == len(list2)
        assert all((list1[i] == list2[i] for i in range(len(list1))))

    # setup 1: convert room-object graph to homogeneous torch data
    data_homogeneous = G_ro.to_torch(
        use_heterogeneous=False,
        node_converter=dsg_node_converter(
            object_feature_converter=hydra_object_feature_converter(
                colormap_data, word2vec_model
            ),
            room_feature_converter=lambda i: np.empty(300),
        ),
    )
    object_label_dict, room_label_dict = convert_label_to_y(
        data_homogeneous,
        object_labels=OBJECT_LABELS,
        room_labels=ROOM_LABELS,
        object_synonyms=object_synonyms,
        room_synonyms=room_synonyms,
    )
    _check_dict(object_label_dict, object_label_dict_exp)
    _check_dict(room_label_dict, room_label_dict_exp)

    object_y = data_homogeneous.y[data_homogeneous.node_masks[2]]
    room_y = data_homogeneous.y[data_homogeneous.node_masks[4]]
    _check_list(object_y.tolist(), object_y_exp)
    _check_list(room_y.tolist(), room_y_exp)

    # setup 2: convert room-object graph to heterogeneous torch data
    data_heterogeneous = G_ro.to_torch(
        use_heterogeneous=True,
        node_converter=dsg_node_converter(
            object_feature_converter=hydra_object_feature_converter(
                colormap_data, word2vec_model
            ),
            room_feature_converter=lambda i: np.empty(0),
        ),
    )
    object_label_dict, room_label_dict = convert_label_to_y(
        data_heterogeneous,
        object_labels=OBJECT_LABELS,
        room_labels=ROOM_LABELS,
        object_synonyms=object_synonyms,
        room_synonyms=room_synonyms,
    )
    _check_dict(object_label_dict, object_label_dict_exp)
    _check_dict(room_label_dict, room_label_dict_exp)
    _check_list(data_heterogeneous["rooms"].y.tolist(), room_y_exp)
    _check_list(data_heterogeneous["objects"].y.tolist(), object_y_exp)
