#!/usr/bin/env python3
"""Build a PufferDrive self-play map from CARLA Town07 OpenDRIVE."""

import argparse
import hashlib
import json
import math
import struct
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import LineString
from shapely.ops import unary_union


DEFAULT_TOWN07_URL = (
    "https://raw.githubusercontent.com/carla-simulator/opendrive-test-files/master/OpenDrive/Town07.xodr"
)
DEFAULT_ROAD_JSON = Path("data_utils/carla/carla_py123d/Town07.json")
TRAJECTORY_LENGTH = 91
MIN_MAP_ELEMENT_LENGTH = 0.25
MIN_AGENT_LANE_LENGTH = 8.0
ROAD_LINE_SUPPLEMENT_DISTANCE = 1.5
LANE_SURFACE_BUFFER = 1.9
LANE_SURFACE_SIMPLIFY = 0.5
ROAD_EDGE_RING_MIN_LENGTH = 5.0


def poly(coeffs, s):
    return (
        coeffs.get("a", 0.0)
        + coeffs.get("b", 0.0) * s
        + coeffs.get("c", 0.0) * s * s
        + coeffs.get("d", 0.0) * s * s * s
    )


def attrs_as_float(elem, *names):
    return {name: float(elem.attrib.get(name, 0.0)) for name in names}


def select_record(records, s, key):
    best = records[0] if records else {"s": 0.0, "sOffset": 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    for record in records:
        if record[key] <= s:
            best = record
        else:
            break
    return best


def eval_lane_offset(records, s):
    record = select_record(records, s, "s")
    return poly(record, s - record["s"])


def eval_lane_width(lane, s_offset):
    records = lane["widths"]
    if not records:
        return 0.0
    record = select_record(records, s_offset, "sOffset")
    return max(0.0, poly(record, s_offset - record["sOffset"]))


class RoadGeometry:
    def __init__(self, road):
        self.length = float(road.attrib["length"])
        self.geometries = []
        for elem in road.findall("./planView/geometry"):
            item = attrs_as_float(elem, "s", "x", "y", "hdg", "length")
            child = next(iter(elem), None)
            item["kind"] = child.tag if child is not None else "line"
            item["curvature"] = float(child.attrib.get("curvature", 0.0)) if child is not None else 0.0
            self.geometries.append(item)
        self.geometries.sort(key=lambda g: g["s"])

    def pose_at(self, road_s):
        road_s = min(max(road_s, 0.0), self.length)
        geom = self.geometries[-1]
        for candidate in self.geometries:
            if candidate["s"] <= road_s:
                geom = candidate
            else:
                break

        ds = min(max(road_s - geom["s"], 0.0), geom["length"])
        x0, y0, hdg = geom["x"], geom["y"], geom["hdg"]
        if geom["kind"] == "arc" and abs(geom["curvature"]) > 1e-9:
            k = geom["curvature"]
            theta = hdg + ds * k
            x = x0 + (math.sin(theta) - math.sin(hdg)) / k
            y = y0 - (math.cos(theta) - math.cos(hdg)) / k
            return x, y, theta
        return x0 + ds * math.cos(hdg), y0 + ds * math.sin(hdg), hdg


def lane_travel_dir(lane_elem):
    vector_lane = lane_elem.find("./userData/vectorLane")
    if vector_lane is None:
        return "forward"
    return vector_lane.attrib.get("travelDir", "forward")


def parse_lane(lane_elem):
    return {
        "id": int(lane_elem.attrib["id"]),
        "type": lane_elem.attrib.get("type", ""),
        "travel_dir": lane_travel_dir(lane_elem),
        "widths": [
            attrs_as_float(width, "sOffset", "a", "b", "c", "d") for width in lane_elem.findall("width")
        ],
    }


def driving_lane_polylines(xodr_path, sample_spacing):
    return [
        elem
        for elem in opendrive_map_elements(xodr_path, sample_spacing)
        if elem["type"] == "lane" and polyline_length(element_points(elem)) >= MIN_AGENT_LANE_LENGTH
    ]


def sample_offset_polyline(road_geom, lane_offsets, start_s, end_s, sample_spacing, offset_at_s, reverse=False):
    count = max(2, int(math.ceil((end_s - start_s) / sample_spacing)) + 1)
    points = []
    for i in range(count):
        t = i / (count - 1)
        road_s = start_s + t * (end_s - start_s)
        offset = eval_lane_offset(lane_offsets, road_s) + offset_at_s(road_s - start_s)
        x, y, hdg = road_geom.pose_at(road_s)
        points.append(
            {
                "x": x - math.sin(hdg) * offset,
                "y": y + math.cos(hdg) * offset,
                "z": 0.0,
            }
        )
    if reverse:
        points.reverse()
    return dedupe_points(points)


def opendrive_map_elements(xodr_path, sample_spacing):
    root = ET.parse(xodr_path).getroot()
    elements = []
    element_id = 0

    for road in root.findall("road"):
        road_geom = RoadGeometry(road)
        lane_offsets = [
            attrs_as_float(offset, "s", "a", "b", "c", "d") for offset in road.findall("./lanes/laneOffset")
        ]
        lane_offsets.sort(key=lambda r: r["s"])

        lane_sections = road.findall("./lanes/laneSection")
        for section_idx, section in enumerate(lane_sections):
            section_s = float(section.attrib.get("s", 0.0))
            next_s = (
                float(lane_sections[section_idx + 1].attrib.get("s", road_geom.length))
                if section_idx + 1 < len(lane_sections)
                else road_geom.length
            )
            if next_s - section_s < MIN_MAP_ELEMENT_LENGTH:
                continue

            left = [parse_lane(lane) for lane in section.findall("./left/lane")]
            right = [parse_lane(lane) for lane in section.findall("./right/lane")]
            side_lanes = [
                ("left", sorted(left, key=lambda lane: lane["id"])),
                ("right", sorted(right, key=lambda lane: abs(lane["id"]))),
            ]

            has_driving = any(lane["type"] == "driving" for _, lanes in side_lanes for lane in lanes)
            if has_driving:
                points = sample_offset_polyline(
                    road_geom,
                    lane_offsets,
                    section_s,
                    next_s,
                    sample_spacing,
                    offset_at_s=lambda _local_s: 0.0,
                )
                if len(points) >= 2:
                    elements.append(
                        {
                            "id": element_id,
                            "type": "road_line",
                            "road_id": road.attrib.get("id", "0"),
                            "lane_id": 0,
                            "points": points,
                        }
                    )
                    element_id += 1

            for side, section_lanes in side_lanes:
                sign = 1.0 if side == "left" else -1.0
                inner_lanes = []
                driving_boundary_fns = []
                for lane_idx, lane in enumerate(section_lanes):
                    if lane["type"] != "driving" or eval_lane_width(lane, 0.0) <= 0.0:
                        inner_lanes.append(lane)
                        continue

                    preceding_lanes = tuple(inner_lanes)

                    def center_offset(local_s, lane=lane, preceding_lanes=preceding_lanes, sign=sign):
                        inner_width = sum(eval_lane_width(inner, local_s) for inner in preceding_lanes)
                        return sign * (inner_width + eval_lane_width(lane, local_s) / 2.0)

                    points = sample_offset_polyline(
                        road_geom,
                        lane_offsets,
                        section_s,
                        next_s,
                        sample_spacing,
                        center_offset,
                        reverse=lane["travel_dir"] == "backward",
                    )

                    if len(points) >= 2:
                        elements.append(
                            {
                                "id": element_id,
                                "type": "lane",
                                "road_id": road.attrib.get("id", "0"),
                                "lane_id": lane["id"],
                                "points": dedupe_points(points),
                            }
                        )
                        element_id += 1

                    boundary_lanes = tuple(inner_lanes + [lane])

                    def boundary_offset(local_s, boundary_lanes=boundary_lanes, sign=sign):
                        return sign * sum(eval_lane_width(inner, local_s) for inner in boundary_lanes)

                    driving_boundary_fns.append(boundary_offset)
                    inner_lanes.append(lane)

                for boundary_fn in driving_boundary_fns[:-1]:
                    points = sample_offset_polyline(
                        road_geom,
                        lane_offsets,
                        section_s,
                        next_s,
                        sample_spacing,
                        boundary_fn,
                    )
                    if len(points) >= 2:
                        elements.append(
                            {
                                "id": element_id,
                                "type": "road_line",
                                "road_id": road.attrib.get("id", "0"),
                                "lane_id": 0,
                                "points": points,
                            }
                        )
                        element_id += 1

                if driving_boundary_fns:
                    points = sample_offset_polyline(
                        road_geom,
                        lane_offsets,
                        section_s,
                        next_s,
                        sample_spacing,
                        driving_boundary_fns[-1],
                    )
                    if len(points) >= 2:
                        elements.append(
                            {
                                "id": element_id,
                                "type": "road_edge",
                                "road_id": road.attrib.get("id", "0"),
                                "lane_id": 0,
                                "points": points,
                            }
                        )
                        element_id += 1

    return [elem for elem in elements if polyline_length(elem["points"]) >= MIN_MAP_ELEMENT_LENGTH]


def load_road_json_elements(road_json_path):
    with open(road_json_path) as file:
        data = json.load(file)

    elements = []
    for idx, road in enumerate(data.get("roads", [])):
        road_type = road.get("type")
        if road_type not in {"lane", "road_line"}:
            continue

        points = [
            {"x": float(point.get("x", 0.0)), "y": float(point.get("y", 0.0)), "z": float(point.get("z", 0.0))}
            for point in road.get("geometry", [])
        ]
        points = dedupe_points(points)
        if len(points) < 2 or polyline_length(points) < MIN_MAP_ELEMENT_LENGTH:
            continue

        elements.append(
            {
                "id": int(road.get("id", idx)),
                "type": road_type,
                "road_id": road.get("map_element_id", road.get("id", idx)),
                "lane_id": road.get("id", idx),
                "points": points,
            }
        )

    return elements


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("CLIPGT parquet input requires pyarrow. Install pyarrow to use --clipgt-dir.") from exc
    return pq


def clipgt_payload_id(row):
    key = row.get("key") or {}
    map_id = key.get("map_id")
    if map_id is None:
        return None
    try:
        return int(map_id)
    except (TypeError, ValueError):
        digest = hashlib.sha1(str(map_id).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def clipgt_points(points):
    out = []
    for point in points or []:
        out.append(
            {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", 0.0)),
            }
        )
    return dedupe_points(out)


def centerline_from_rails(left_rail, right_rail, sample_spacing):
    left = clipgt_points(left_rail)
    right = clipgt_points(right_rail)
    if len(left) < 2 or len(right) < 2:
        return []

    left_length = polyline_length(left)
    right_length = polyline_length(right)
    length = max(left_length, right_length)
    count = max(2, int(math.ceil(length / sample_spacing)) + 1)
    center = []
    for idx in range(count):
        distance_along = length * idx / (count - 1)
        lpt = point_at_distance(left, min(distance_along, left_length))
        rpt = point_at_distance(right, min(distance_along, right_length))
        center.append(
            {
                "x": 0.5 * (lpt["x"] + rpt["x"]),
                "y": 0.5 * (lpt["y"] + rpt["y"]),
                "z": 0.5 * (lpt.get("z", 0.0) + rpt.get("z", 0.0)),
            }
        )
    return dedupe_points(center)


def load_clipgt_elements(clipgt_dir, sample_spacing=2.0):
    pq = require_pyarrow()
    clipgt_dir = Path(clipgt_dir)
    if not clipgt_dir.is_dir():
        raise FileNotFoundError(f"CLIPGT directory not found: {clipgt_dir}")

    elements = []
    next_id = 0

    lane_path = clipgt_dir / "lane.parquet"
    if lane_path.exists():
        for row in pq.read_table(lane_path).to_pylist():
            lane = row.get("lane") or {}
            points = centerline_from_rails(lane.get("left_rail"), lane.get("right_rail"), sample_spacing)
            if len(points) < 2 or polyline_length(points) < MIN_MAP_ELEMENT_LENGTH:
                continue
            elem_id = clipgt_payload_id(row)
            if elem_id is None:
                elem_id = next_id
            elements.append(
                {
                    "id": elem_id,
                    "type": "lane",
                    "road_id": elem_id,
                    "lane_id": elem_id,
                    "points": points,
                }
            )
            next_id += 1

    lane_line_path = clipgt_dir / "lane_line.parquet"
    if lane_line_path.exists():
        for row in pq.read_table(lane_line_path).to_pylist():
            points = clipgt_points((row.get("lane_line") or {}).get("line_rail"))
            if len(points) < 2 or polyline_length(points) < MIN_MAP_ELEMENT_LENGTH:
                continue
            elem_id = clipgt_payload_id(row)
            if elem_id is None:
                elem_id = next_id
            elements.append(
                {
                    "id": elem_id,
                    "type": "road_line",
                    "road_id": elem_id,
                    "lane_id": 0,
                    "points": points,
                }
            )
            next_id += 1

    road_boundary_path = clipgt_dir / "road_boundary.parquet"
    if road_boundary_path.exists():
        for row in pq.read_table(road_boundary_path).to_pylist():
            points = clipgt_points((row.get("road_boundary") or {}).get("location"))
            if len(points) < 2 or polyline_length(points) < MIN_MAP_ELEMENT_LENGTH:
                continue
            elem_id = clipgt_payload_id(row)
            if elem_id is None:
                elem_id = next_id
            elements.append(
                {
                    "id": elem_id,
                    "type": "road_edge",
                    "road_id": elem_id,
                    "lane_id": 0,
                    "points": points,
                }
            )
            next_id += 1

    return elements


def build_lane_surface(map_elements, buffer_distance=LANE_SURFACE_BUFFER):
    lane_polygons = []
    for elem in map_elements:
        if elem["type"] != "lane":
            continue
        points = element_points(elem)
        if len(points) < 2:
            continue
        line = LineString([(point["x"], point["y"]) for point in points])
        if line.length <= 0.0:
            continue
        lane_polygons.append(line.buffer(buffer_distance, cap_style=2, join_style=2))

    if not lane_polygons:
        return None
    return unary_union(lane_polygons).simplify(LANE_SURFACE_SIMPLIFY, preserve_topology=True)


def road_edges_from_lane_surface(lane_surface):
    if lane_surface is None or lane_surface.is_empty:
        return []

    road_edges = []
    next_id = 0

    def append_ring(coords):
        nonlocal next_id
        points = [{"x": float(x), "y": float(y), "z": 0.0} for x, y in coords]
        points = dedupe_points(points)
        if len(points) < 2 or polyline_length(points) < ROAD_EDGE_RING_MIN_LENGTH:
            return
        road_edges.append(
            {
                "id": next_id,
                "type": "road_edge",
                "road_id": "junction_bounds",
                "lane_id": 0,
                "points": points,
            }
        )
        next_id += 1

    def visit(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            append_ring(list(geom.exterior.coords))
            for hole in geom.interiors:
                append_ring(list(hole.coords))
        elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
            for child in geom.geoms:
                visit(child)

    visit(lane_surface)
    return road_edges


def point_segment_distance(point, start, end):
    vx = end["x"] - start["x"]
    vy = end["y"] - start["y"]
    wx = point["x"] - start["x"]
    wy = point["y"] - start["y"]
    denom = vx * vx + vy * vy
    t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    x = start["x"] + t * vx
    y = start["y"] + t * vy
    return math.hypot(point["x"] - x, point["y"] - y)


def build_segment_grid(polylines, cell_size):
    grid = {}
    for points in polylines:
        for segment in zip(points, points[1:]):
            add_segment_to_grid(grid, segment[0], segment[1], cell_size)
    return grid


def add_segment_to_grid(grid, start, end, cell_size):
    min_x = min(start["x"], end["x"])
    max_x = max(start["x"], end["x"])
    min_y = min(start["y"], end["y"])
    max_y = max(start["y"], end["y"])
    ix0 = math.floor(min_x / cell_size)
    ix1 = math.floor(max_x / cell_size)
    iy0 = math.floor(min_y / cell_size)
    iy1 = math.floor(max_y / cell_size)
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            grid.setdefault((ix, iy), []).append((start, end))


def nearest_segment_distance(point, segment_grid, cell_size, search_cells=1):
    ix = math.floor(point["x"] / cell_size)
    iy = math.floor(point["y"] / cell_size)
    best = float("inf")
    for dx in range(-search_cells, search_cells + 1):
        for dy in range(-search_cells, search_cells + 1):
            for start, end in segment_grid.get((ix + dx, iy + dy), ()):
                best = min(best, point_segment_distance(point, start, end))
    return best


def supplement_xodr_elements(map_elements, xodr_elements, element_type, min_distance):
    existing_elements = [elem["points"] for elem in map_elements if elem["type"] == element_type]
    if not existing_elements:
        return 0

    cell_size = max(1.0, min_distance * 2.0)
    segment_grid = build_segment_grid(existing_elements, cell_size)
    next_id = max((int(elem.get("id", 0)) for elem in map_elements), default=-1) + 1
    added = 0

    def append_run(run, source_elem):
        nonlocal added, next_id
        run = dedupe_points(run)
        if len(run) < 2 or polyline_length(run) < MIN_MAP_ELEMENT_LENGTH:
            return
        supplemented = {
            **source_elem,
            "id": next_id,
            "points": run,
            "source": f"xodr_{element_type}_supplement",
        }
        map_elements.append(supplemented)
        for start, end in zip(run, run[1:]):
            add_segment_to_grid(segment_grid, start, end, cell_size)
        next_id += 1
        added += 1

    for elem in xodr_elements:
        if elem["type"] != element_type:
            continue
        points = elem["points"]
        if len(points) < 2:
            continue

        run = []
        for start, end in zip(points, points[1:]):
            midpoint = {
                "x": (start["x"] + end["x"]) * 0.5,
                "y": (start["y"] + end["y"]) * 0.5,
                "z": (start.get("z", 0.0) + end.get("z", 0.0)) * 0.5,
            }
            missing = nearest_segment_distance(midpoint, segment_grid, cell_size, search_cells=1) > min_distance
            if missing:
                if not run:
                    run.append(start)
                run.append(end)
            elif run:
                append_run(run, elem)
                run = []
        if run:
            append_run(run, elem)

    return added


def supplement_xodr_road_lines(map_elements, xodr_elements, min_distance=ROAD_LINE_SUPPLEMENT_DISTANCE):
    return supplement_xodr_elements(map_elements, xodr_elements, "road_line", min_distance)


def element_points(elem):
    return elem.get("points", elem.get("geometry", []))


def dedupe_points(points):
    out = []
    for point in points:
        if not out or distance(out[-1], point) > 1e-4:
            out.append(point)
    return out


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def polyline_length(points):
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def point_at_distance(points, target):
    if target <= 0:
        return points[0]
    remaining = target
    for a, b in zip(points, points[1:]):
        seg = distance(a, b)
        if seg >= remaining:
            t = remaining / seg if seg > 0 else 0.0
            return {
                "x": a["x"] + (b["x"] - a["x"]) * t,
                "y": a["y"] + (b["y"] - a["y"]) * t,
                "z": a.get("z", 0.0) + (b.get("z", 0.0) - a.get("z", 0.0)) * t,
            }
        remaining -= seg
    return points[-1]


def heading_between(a, b):
    return math.atan2(b["y"] - a["y"], b["x"] - a["x"])


def trajectory_from_lane(points, speed, dt):
    start = min(1.0, max(0.0, polyline_length(points) * 0.05))
    positions = []
    headings = []
    for step in range(TRAJECTORY_LENGTH):
        pos = point_at_distance(points, start + speed * dt * step)
        nxt = point_at_distance(points, start + speed * dt * step + 0.5)
        positions.append({"x": pos["x"], "y": pos["y"], "z": 0.9})
        headings.append(heading_between(pos, nxt))
    return positions, headings


def build_map_data(map_elements, num_agents, speed, dt, scenario_id="Town07", metadata_source="CARLA Town07 OpenDRIVE"):
    roads = []
    lanes = [
        elem
        for elem in map_elements
        if elem["type"] == "lane" and polyline_length(element_points(elem)) >= MIN_AGENT_LANE_LENGTH
    ]
    for idx, elem in enumerate(map_elements):
        points = element_points(elem)
        roads.append(
            {
                "id": idx,
                "type": elem["type"],
                "geometry": points,
                "width": 3.5,
                "length": polyline_length(points),
                "height": 0.0,
            }
        )

    candidate_lanes = sorted(lanes, key=lambda lane: polyline_length(element_points(lane)), reverse=True)
    objects = []
    for agent_idx, lane in enumerate(candidate_lanes[:num_agents]):
        points = element_points(lane)
        positions, headings = trajectory_from_lane(points, speed=speed, dt=dt)
        goal = point_at_distance(points, min(polyline_length(points), 35.0))
        velocities = []
        for heading in headings:
            velocities.append({"x": speed * math.cos(heading), "y": speed * math.sin(heading), "z": 0.0})
        objects.append(
            {
                "id": agent_idx,
                "type": "vehicle",
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": [1] * TRAJECTORY_LENGTH,
                "width": 2.0,
                "length": 4.7,
                "height": 1.8,
                "goalPosition": {"x": goal["x"], "y": goal["y"], "z": 0.9},
                "mark_as_expert": 0,
            }
        )

    return {
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": [],
        "metadata": {
            "sdc_track_index": -1,
            "tracks_to_predict": [{"track_index": idx} for idx in range(len(objects))],
            "source": metadata_source,
        },
    }


def infer_actions(obj, dt):
    positions = obj["position"]
    headings = obj["heading"]
    zeros = [0.0] * TRAJECTORY_LENGTH
    dx = [0.0] * TRAJECTORY_LENGTH
    dy = [0.0] * TRAJECTORY_LENGTH
    dyaw = [0.0] * TRAJECTORY_LENGTH
    for i in range(TRAJECTORY_LENGTH - 1):
        x0, y0 = positions[i]["x"], positions[i]["y"]
        x1, y1 = positions[i + 1]["x"], positions[i + 1]["y"]
        heading = headings[i]
        wx, wy = x1 - x0, y1 - y0
        dx[i] = wx * math.cos(heading) + wy * math.sin(heading)
        dy[i] = -wx * math.sin(heading) + wy * math.cos(heading)
        dyaw[i] = wrap_angle(headings[i + 1] - heading)
    return zeros, zeros, dx, dy, dyaw


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def write_fixed_string(file, value, length):
    encoded = value.encode("utf-8")[:length]
    file.write(encoded + b"\0" * (length - len(encoded)))


def save_map_binary(map_data, output_file, unique_map_id, dt):
    with open(output_file, "wb") as file:
        metadata = map_data.get("metadata", {})
        write_fixed_string(file, map_data.get("scenario_id", f"map_{unique_map_id:03d}"), 16)
        file.write(struct.pack("<i", int(metadata.get("sdc_track_index", -1))))
        tracks_to_predict = metadata.get("tracks_to_predict", [])
        file.write(struct.pack("<i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            file.write(struct.pack("<i", int(track.get("track_index", -1))))

        file.write(struct.pack("<ii", len(map_data["objects"]), len(map_data["roads"])))

        for obj in map_data["objects"]:
            file.write(struct.pack("<iiii", unique_map_id, 1, int(obj["id"]), TRAJECTORY_LENGTH))
            for coord in ("x", "y", "z"):
                for pos in obj["position"]:
                    file.write(struct.pack("<f", float(pos.get(coord, 0.0))))
            for coord in ("x", "y", "z"):
                for vel in obj["velocity"]:
                    file.write(struct.pack("<f", float(vel.get(coord, 0.0))))
            file.write(struct.pack(f"<{TRAJECTORY_LENGTH}f", *[float(v) for v in obj["heading"]]))
            file.write(struct.pack(f"<{TRAJECTORY_LENGTH}i", *[int(v) for v in obj["valid"]]))
            for arr in infer_actions(obj, dt):
                file.write(struct.pack(f"<{TRAJECTORY_LENGTH}f", *arr))
            file.write(
                struct.pack(
                    "<ffffffi",
                    float(obj["width"]),
                    float(obj["length"]),
                    float(obj["height"]),
                    float(obj["goalPosition"]["x"]),
                    float(obj["goalPosition"]["y"]),
                    float(obj["goalPosition"].get("z", 0.0)),
                    int(obj.get("mark_as_expert", 0)),
                )
            )

        road_type_map = {
            "lane": 4,
            "road_line": 5,
            "road_edge": 6,
        }
        for road in map_data["roads"]:
            geometry = road["geometry"]
            file.write(
                struct.pack("<iiii", unique_map_id, road_type_map.get(road.get("type"), 4), int(road["id"]), len(geometry))
            )
            for coord in ("x", "y", "z"):
                for point in geometry:
                    file.write(struct.pack("<f", float(point.get(coord, 0.0))))
            file.write(
                struct.pack(
                    "<ffffffi",
                    float(road.get("width", 0.0)),
                    float(road.get("length", 0.0)),
                    float(road.get("height", 0.0)),
                    0.0,
                    0.0,
                    0.0,
                    int(road.get("mark_as_expert", 0)),
                )
            )


def save_map_json(map_data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(map_data, file, indent=2)
        file.write("\n")


def ensure_xodr(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urllib.request.urlretrieve(url, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", type=Path, default=Path("data_utils/carla/opendrive/Town07.xodr"))
    parser.add_argument("--xodr-url", default=DEFAULT_TOWN07_URL)
    parser.add_argument("--road-json", type=Path, default=DEFAULT_ROAD_JSON)
    parser.add_argument("--clipgt-dir", type=Path)
    parser.add_argument("--scenario-id", default="Town07")
    parser.add_argument("--no-road-json", action="store_true")
    parser.add_argument("--no-xodr-road-line-supplement", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("pufferlib/resources/drive/binaries/town07_selfplay"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument("--sample-spacing", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    if args.clipgt_dir:
        map_elements = load_clipgt_elements(args.clipgt_dir, sample_spacing=args.sample_spacing)
        geometry_source = str(args.clipgt_dir)
        supplemented_road_lines = 0
        preserve_input_road_edges = True
    elif not args.no_road_json and args.road_json.exists():
        ensure_xodr(args.xodr, args.xodr_url)
        map_elements = load_road_json_elements(args.road_json)
        geometry_source = str(args.road_json)
        supplemented_road_lines = 0
        preserve_input_road_edges = False
        if not args.no_xodr_road_line_supplement:
            xodr_elements = opendrive_map_elements(args.xodr, args.sample_spacing)
            supplemented_road_lines = supplement_xodr_road_lines(map_elements, xodr_elements)
    else:
        ensure_xodr(args.xodr, args.xodr_url)
        map_elements = opendrive_map_elements(args.xodr, args.sample_spacing)
        geometry_source = str(args.xodr)
        supplemented_road_lines = 0
        preserve_input_road_edges = False

    lane_surface = build_lane_surface(map_elements)
    derived_road_edges = road_edges_from_lane_surface(lane_surface)
    if preserve_input_road_edges and any(elem["type"] == "road_edge" for elem in map_elements):
        derived_road_edges = []
    else:
        map_elements = [elem for elem in map_elements if elem["type"] != "road_edge"] + derived_road_edges
    lanes = [elem for elem in map_elements if elem["type"] == "lane"]
    if not lanes:
        raise RuntimeError(f"No driving lanes parsed from {geometry_source}")

    map_data = build_map_data(
        map_elements,
        num_agents=args.num_agents,
        speed=args.speed,
        dt=args.dt,
        scenario_id=args.scenario_id,
        metadata_source=geometry_source,
    )
    if not map_data["objects"]:
        raise RuntimeError("No training agents generated from Town07 lanes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "map_000.bin"
    save_map_binary(map_data, output_file, unique_map_id=0, dt=args.dt)
    if args.output_json:
        save_map_json(map_data, args.output_json)

    print(f"Wrote {output_file}")
    if args.output_json:
        print(f"Wrote {args.output_json}")
    print(f"Road geometry source: {geometry_source}")
    print(f"Parsed driving lanes: {len(lanes)}")
    print(f"Parsed road lines: {sum(1 for elem in map_elements if elem['type'] == 'road_line')}")
    print(f"Parsed road edges: {sum(1 for elem in map_elements if elem['type'] == 'road_edge')}")
    print(f"Supplemented road lines: {supplemented_road_lines}")
    print(f"Derived road edges: {len(derived_road_edges)}")
    print(f"Generated agents: {len(map_data['objects'])}")


if __name__ == "__main__":
    main()
