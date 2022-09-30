import spark_dsg as dsg
import shapely.geometry
import shapely.ops
import numpy as np
import seaborn as sns
import open3d as o3d


def _add_building_node(G, index=0, prefix="B", label=None):
    building_id = dsg.NodeSymbol(prefix, index)
    attrs = dsg.SemanticNodeAttributes()
    attrs.name = str(building_id)
    if label is not None:
        attrs.semantic_label = label

    pos = np.zeros((3, 1))
    N = G.get_layer(dsg.DsgLayers.ROOMS).num_nodes()
    for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
        pos += room.attributes.position.reshape((3, 1))

    if N > 0:
        pos /= N

    attrs.position = pos
    G.add_node(dsg.DsgLayers.BUILDINGS, building_id.value, attrs)
    return building_id


def _add_room_node(G, room, color, prefix="R"):
    room_id = dsg.NodeSymbol(prefix, room["id"])
    room_attrs = dsg.RoomNodeAttributes()
    room_attrs.position = room["pos"] + np.array([0, 0, room["height"] / 2.0])
    room_attrs.name = str(room_id)
    room_attrs.bounding_box = dsg.BoundingBox(
        room["bbox_min"],
        room["bbox_max"],
        room_attrs.position,
        dsg.Quaternionf(1.0, 0.0, 0.0, 0.0),
    )
    room_attrs.semantic_label = ord(room["label"])
    room_attrs.color = (255 * np.array(color[0:3])).astype(np.uint8)

    G.add_node(dsg.DsgLayers.ROOMS, room_id.value, room_attrs)


def _add_object_node(G, object_info, categories, prefix="O", color_dict=None):
    if object_info["category"] not in categories:
        print(
            f"Invalid object category {object_info['category']} @ {object_info['id']}"
        )
        return

    node_id = dsg.NodeSymbol(prefix, object_info["id"])
    attrs = dsg.ObjectNodeAttributes()
    attrs.position = object_info["pos"]
    attrs.name = str(node_id)
    yaw = np.arctan2(object_info["a0"][1], object_info["a0"][0])
    q = dsg.Quaternionf(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    bbox_min = attrs.position - (object_info["r"] / 2.0)
    bbox_max = attrs.position + (object_info["r"] / 2.0)
    attrs.bounding_box = dsg.BoundingBox(bbox_min, bbox_max, attrs.position, q)
    attrs.semantic_label = categories[object_info["category"]]["mpcat_idx"]
    if color_dict and attrs.semantic_label in color_dict:
        attrs.color = color_dict[attrs.semantic_label]

    attrs.registered = False

    G.add_node(dsg.DsgLayers.OBJECTS, node_id.value, attrs)


def construct_mp3d_dsg(info, building_label, mesh_path=None, compression_size=None):
    """Make a ground-truth DSG from mp3d."""
    G = dsg.DynamicSceneGraph()

    for _, obj in info["O"].items():
        _add_object_node(G, obj, info["C"])

    room_colors = sns.color_palette("Paired")
    for idx, r_index in enumerate(info["R"]):
        _add_room_node(G, info["R"][r_index], room_colors[idx % len(room_colors)])

    if len(info["R"]) > 0:
        building_id = _add_building_node(G, label=building_label)
        for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
            G.insert_edge(building_id.value, room.id.value)

    if mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if compression_size is not None:
            mesh = mesh.simplify_vertex_clustering(compression_size)
        points = np.asarray(mesh.vertices).T
        colors = np.asarray(mesh.vertex_colors).T
        G.set_mesh_vertices(np.vstack((points, colors)))
        G.set_mesh_faces(np.asarray(mesh.triangles).T)

    return G


def repartition_rooms(G_prev, mp3d_info):
    """Remove old nodes and make new ones."""
    G = G_prev.clone()
    cmap = sns.color_palette("husl", len(rooms))

    for index, room in enumerate(rooms):
        color = cmap[index]
        attrs = dsg.RoomNodeAttributes()
        attrs.color = np.array([int(255 * c) for c in color][:3])
        attrs.name = str(dsg.NodeSymbol("R", room.index))
        attrs.position = np.array(
            [room.pos[0], room.pos[1], room.pos[2] + room.height / 2.0]
        )
        attrs.last_update_time_ns = 0
        attrs.semantic_label = ord(room.label)

        G.add_node(dsg.DsgLayers.ROOMS, dsg.NodeSymbol("R", room.index), attrs)

    room_map = {}
    missing_nodes = []
    for place in G.get_layer(dsg.DsgLayers.PLACES).nodes:
        pos = G.get_position(place.id.value)
        xy_pos = shapely.geometry.Point(pos[0], pos[1])
        for room in rooms:
            if pos[2] <= room.floor_z or pos[2] >= room.floor_z + room.height:
                continue

            if room.polygon.contains(xy_pos):
                room_id = dsg.NodeSymbol("R", room.index)
                room_map[place] = room_id
                G.add_edge(place.id.valud, room_id.value)
                break
        else:
            missing_nodes.append(place)

    invalid_rooms = []
    for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
        if not room.has_children():
            invalid_rooms.append(room.id.value)

    for room_id in invalid_rooms:
        G.remove_node(room_id)

    return G
