"""Functions for parsing a house file."""
import spark_dsg as dsg
import seaborn as sns
import numpy as np
import shapely.geometry
import shapely.ops


def _filter_line(line):
    contents = line.strip().split(" ")[1:]
    return line[0], [x for x in contents if x != ""]


def _make_vec3(raw_list, start):
    return np.array([float(x) for x in raw_list[start : start + 3]])


def parse_region(line):
    """Get region info from a line."""
    return {
        "region_index": int(line[0]),
        "level_index": int(line[1]),
        "label": line[4],
        "pos": _make_vec3(line, 5),
        "bbox_min": _make_vec3(line, 8),
        "bbox_max": _make_vec3(line, 11),
        "height": float(line[14]),
    }


def parse_surface(line):
    """Get surface info from a line."""
    return {
        "surface_index": int(line[0]),
        "region_index": int(line[1]),
        "label": line[3],
        "pos": _make_vec3(line, 4),
        "normal": _make_vec3(line, 7),
        "bbox_min": _make_vec3(line, 10),
        "bbox_max": _make_vec3(line, 13),
    }


def parse_vertex(line):
    """Get vertex info from a line."""
    return {
        "vertex_index": int(line[0]),
        "surface_index": int(line[1]),
        "label": line[2],
        "pos": _make_vec3(line, 3),
        "normal": _make_vec3(line, 6),
    }


def parse_category(line):
    """Get category info from a line."""
    return {
        "category_index": int(line[0]),
        "category_mapping_index": int(line[1]),
        "category_mapping_name": line[2],
        "mpcat40_index": int(line[3]),
        "mpcat40_name": line[4],
    }


def parse_object(line):
    """Get object info from a line."""
    return {
        "object_index": int(line[0]),
        "region_index": int(line[1]),
        "category_index": int(line[2]),
        "pos": _make_vec3(line, 3),
        "a0": _make_vec3(line, 6),
        "a1": _make_vec3(line, 9),
        "r": _make_vec3(line, 12),
    }


def parse_segment(line):
    """Get segment info from a line."""
    return {
        "segment_index": int(line[0]),
        "object_index": int(line[1]),
        "id": int(line[2]),
    }


PARSERS = {
    "V": parse_vertex,
    "S": parse_surface,
    "R": parse_region,
    "C": parse_category,
    "O": parse_object,
    "E": parse_segment,
}


def load_mp3d_info(house_path):
    """Load room info from a GT house file."""
    info = {x: [] for x in PARSERS}
    with open(house_path, "r") as fin:
        for line in fin:
            line_type, line = _filter_line(line)
            if line_type not in PARSERS:
                continue

            new_element = PARSERS[line_type](line)
            info[line_type].append(new_element)

    return info


class Mp3dRoom:
    """Quick utility class."""

    def __init__(self, index, region, vertices, angle_deg=90.0):
        """Make a polygon for a labeled room."""
        self._index = index
        self._label = region["label"]
        self._pos = np.array(
            [
                region["pos"][0],
                region["pos"][1],
                region["pos"][2] + region["height"] / 2.0,
            ]
        )

        self._min_z = np.mean(np.array(vertices)[:, 2])
        self._max_z = self._min_z + region["height"]

        # house files are rotated 90 degreees from Hydra convention
        xy_polygon = shapely.geometry.Polygon([x[:2].tolist() for x in vertices])

        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        self._pos[:2] = R @ self._pos[:2]
        rotated_vertices = [
            (R @ np.array(x)).tolist() for x in xy_polygon.exterior.coords
        ]
        self._polygon_xy = shapely.geometry.Polygon(rotated_vertices)

    def pos_inside_room(self, pos):
        """Check if a 3d position falls within the bounds of the room."""
        if pos[2] <= self._min_z or pos[2] >= self._max_z:
            return False

        xy_pos = shapely.geometry.Point(pos[0], pos[1])
        return self._polygon_xy.contains(xy_pos)

    def get_id(self):
        return dsg.NodeSymbol("R", self._index)

    def get_attrs(self, color):
        """Get DSG room attributes."""
        attrs = dsg.RoomNodeAttributes()
        attrs.color = color
        attrs.name = str(dsg.NodeSymbol("R", self._index))
        attrs.position = self._pos
        attrs.last_update_time_ns = 0
        attrs.semantic_label = ord(self._label)
        return attrs


def repartition_rooms(G_prev, mp3d_info):
    """Create a copy of the DSG with ground-truth room nodes."""
    G = G_prev.clone()

    # remove existing rooms
    existing_room_ids = [room.id.value for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes]
    for i in existing_room_ids:
        G.remove_node(i)

    rooms = []
    for region in mp3d_info["R"]:
        r_index = region["region_index"]

        valid_surfaces = []
        for surface in mp3d_info["S"]:
            if surface["region_index"] == r_index:
                valid_surfaces.append(surface["surface_index"])

        vertices = []
        for vertex in mp3d_info["V"]:
            if vertex["surface_index"] in valid_surfaces:
                vertices.append(vertex["pos"])

        rooms.append(Mp3dRoom(r_index, region, vertices))

    cmap = sns.color_palette("husl", len(rooms))

    for index, room in enumerate(rooms):
        color = np.array([int(255 * c) for c in cmap[index]][:3])
        attrs = room.get_attrs(color)

        G.add_node(dsg.DsgLayers.ROOMS, room.get_id().value, attrs)

    room_map = {}
    missing_nodes = []
    for place in G.get_layer(dsg.DsgLayers.PLACES).nodes:
        pos = G.get_position(place.id.value)
        for room in rooms:
            if not room.pos_inside_room(pos):
                continue

            room_id = room.get_id()
            room_map[place] = room_id
            G.insert_edge(place.id.value, room_id.value)
            break
        else:
            missing_nodes.append(place)

    for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
        if not room.has_children():
            G.remove_node(room.id.value)

    return G
