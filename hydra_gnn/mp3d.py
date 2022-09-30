import spark_dsg as dsg
import shapely.geometry
import shapely.ops
import numpy as np


def _filter_line(line):
    contents = line.strip().split(" ")[1:]
    return line[0], [x for x in contents if x != ""]


def _make_vec3(raw_list, start):
    return np.array([float(x) for x in raw_list[start : start + 3]])


def parse_region(line):
    """Get region info from a line."""
    return {
        "id": int(line[0]),
        "floor": int(line[1]),
        "label": line[4],
        "pos": _make_vec3(line, 5),
        "bbox_min": _make_vec3(line, 8),
        "bbox_max": _make_vec3(line, 11),
        "height": float(line[14]),
    }


def parse_surface(line):
    """Get surface info from a line."""
    return {
        "id": int(line[0]),
        "region": int(line[1]),
        "label": line[3],
        "pos": _make_vec3(line, 4),
        "bbox_min": _make_vec3(line, 7),
        "bbox_max": _make_vec3(line, 10),
        "height": float(line[13]),
    }


def parse_vertex(line):
    """Get vertex info from a line."""
    return {
        "id": int(line[0]),
        "surface": int(line[1]),
        "label": line[3],
        "pos": _make_vec3(line, 4),
        "normal": _make_vec3(line, 7),
    }


def parse_category(line):
    """Get category info from a line."""
    return {
        "id": int(line[0]),
        "map_idx": int(line[1]),
        "name": line[2],
        "mpcat_idx": int(line[3]),
        "mpcat_name": line[4],
    }


def parse_object(line):
    """Get object info from a line."""
    return {
        "id": int(line[0]),
        "region": int(line[1]),
        "category": int(line[2]),
        "pos": _make_vec3(line, 3),
        "a0": _make_vec3(line, 6),
        "a1": _make_vec3(line, 9),
        "r": _make_vec3(line, 12),
    }


def parse_segment(line):
    """Get segment info from a line."""
    return {
        "id": int(line[0]),
        "object": int(line[1]),
        "face_id": line[3],
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
    info = {x: {} for x in PARSERS}
    with house_path.open("r") as fin:
        for line in fin:
            line_type, line = _filter_line(line)
            if line_type not in PARSERS:
                continue

            new_element = PARSERS[line_type](line)
            info[line_type][new_element["id"]] = new_element

    return info


def construct_mp3d_rooms(info):
    """Make rooms given a loaded house file."""
    rooms = []
    for r_index, region in info["R"].items():
        vertices = []

        valid_surfaces = []
        for s_index, surface in info["S"].items():
            if surface["region"] == s_index:
                valid_surfaces.append(s_index)

        for v_index, vertex in info["V"].items():
            if vertex["surface"] in valid_surfaces:
                vertices.append(vertex["pos"])

        room = Room(
            index=r_index,
            label=region["label"],
            pos=region["pos"],
            height=region["height"],
            level=region["floor"],
        )
        room.set_vertices(vertices)
        rooms.append(room)

    for room in rooms:
        room.rotate(-90)

    return rooms
