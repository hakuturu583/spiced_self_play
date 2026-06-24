#!/usr/bin/env python3
"""Validate a Drive JSON map and optionally compare it with Town07 OpenDRIVE geometry."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts.build_town07_drive_maps import (
    build_segment_grid,
    build_lane_surface,
    element_points,
    nearest_segment_distance,
    opendrive_map_elements,
    polyline_length,
    road_edges_from_lane_surface,
)


def finite_point(point):
    return all(math.isfinite(float(point.get(coord, 0.0))) for coord in ("x", "y", "z"))


def segment_midpoint(start, end):
    return {
        "x": (start["x"] + end["x"]) * 0.5,
        "y": (start["y"] + end["y"]) * 0.5,
        "z": (start.get("z", 0.0) + end.get("z", 0.0)) * 0.5,
    }


def build_owned_segment_grid(elements, element_type, cell_size):
    grid = {}
    for idx, elem in enumerate(elements):
        if elem.get("type") != element_type:
            continue
        points = element_points(elem)
        for start, end in zip(points, points[1:]):
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
                    grid.setdefault((ix, iy), []).append((idx, start, end))
    return grid


def nearest_other_segment_distance(point, segment_grid, cell_size, owner_idx, search_cells=1):
    ix = math.floor(point["x"] / cell_size)
    iy = math.floor(point["y"] / cell_size)
    best = float("inf")
    for dx in range(-search_cells, search_cells + 1):
        for dy in range(-search_cells, search_cells + 1):
            for idx, start, end in segment_grid.get((ix + dx, iy + dy), ()):
                if idx == owner_idx:
                    continue
                best = min(best, point_segment_distance(point, start, end))
    return best


def report_distance(distance):
    return None if math.isinf(distance) else distance


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


def schema_stats(map_data):
    errors = []
    for key in ("scenario_id", "objects", "roads", "tl_states", "metadata"):
        if key not in map_data:
            errors.append(f"missing top-level key: {key}")

    roads = map_data.get("roads", [])
    objects = map_data.get("objects", [])
    degenerate_roads = []
    nonfinite_roads = []
    for idx, road in enumerate(roads):
        geometry = road.get("geometry", [])
        if len(geometry) < 2 or polyline_length(geometry) <= 0:
            degenerate_roads.append(idx)
        if any(not finite_point(point) for point in geometry):
            nonfinite_roads.append(idx)

    bad_objects = []
    for idx, obj in enumerate(objects):
        if len(obj.get("position", [])) != len(obj.get("heading", [])) or len(obj.get("position", [])) != len(
            obj.get("valid", [])
        ):
            bad_objects.append(idx)

    return {
        "errors": errors,
        "num_objects": len(objects),
        "num_roads": len(roads),
        "road_type_counts": dict(sorted(Counter(road.get("type") for road in roads).items())),
        "degenerate_road_count": len(degenerate_roads),
        "degenerate_road_indices": degenerate_roads[:20],
        "nonfinite_road_count": len(nonfinite_roads),
        "nonfinite_road_indices": nonfinite_roads[:20],
        "bad_object_trajectory_count": len(bad_objects),
        "bad_object_indices": bad_objects[:20],
    }


def dangling_endpoint_stats(roads, element_types, threshold):
    cell_size = max(1.0, threshold * 2.0)
    out = {}
    for element_type in element_types:
        segment_grid = build_owned_segment_grid(roads, element_type, cell_size)
        dangling = []
        for idx, road in enumerate(roads):
            if road.get("type") != element_type:
                continue
            points = road.get("geometry", [])
            if len(points) < 2:
                continue
            for endpoint_name, point in (("start", points[0]), ("end", points[-1])):
                distance = nearest_other_segment_distance(point, segment_grid, cell_size, idx, search_cells=1)
                if distance > threshold:
                    dangling.append(
                        {
                            "road_index": idx,
                            "road_id": road.get("id"),
                            "endpoint": endpoint_name,
                            "nearest_distance": report_distance(distance),
                            "x": point["x"],
                            "y": point["y"],
                        }
                    )
        dangling.sort(
            key=lambda item: float("inf") if item["nearest_distance"] is None else item["nearest_distance"],
            reverse=True,
        )
        out[element_type] = {
            "dangling_endpoint_count": len(dangling),
            "top": dangling[:20],
        }
    return out


def xodr_coverage_stats(roads, xodr_path, element_types, sample_spacing, threshold):
    cell_size = max(1.0, threshold * 2.0)
    out = {}
    xodr_elements = opendrive_map_elements(xodr_path, sample_spacing)
    for element_type in element_types:
        map_lines = [road.get("geometry", []) for road in roads if road.get("type") == element_type]
        segment_grid = build_segment_grid(map_lines, cell_size)
        uncovered = []
        if element_type == "road_edge":
            reference_elements = road_edges_from_lane_surface(build_lane_surface(xodr_elements))
        else:
            reference_elements = [elem for elem in xodr_elements if elem.get("type") == element_type]

        for elem in reference_elements:
            points = elem.get("points", [])
            for seg_idx, (start, end) in enumerate(zip(points, points[1:])):
                mid = segment_midpoint(start, end)
                distance = nearest_segment_distance(mid, segment_grid, cell_size, search_cells=1)
                if distance > threshold:
                    uncovered.append(
                        {
                            "xodr_id": elem.get("id"),
                            "segment_index": seg_idx,
                            "nearest_distance": report_distance(distance),
                            "x": mid["x"],
                            "y": mid["y"],
                            "length": math.hypot(end["x"] - start["x"], end["y"] - start["y"]),
                        }
                    )
        uncovered.sort(
            key=lambda item: float("inf") if item["nearest_distance"] is None else item["nearest_distance"],
            reverse=True,
        )
        out[element_type] = {
            "uncovered_xodr_segment_count": len(uncovered),
            "top": uncovered[:20],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--xodr", type=Path)
    parser.add_argument("--sample-spacing", type=float, default=2.0)
    parser.add_argument("--distance-threshold", type=float, default=1.5)
    parser.add_argument("--types", nargs="+", default=["road_line", "road_edge"])
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as file:
        map_data = json.load(file)

    roads = map_data.get("roads", [])
    report = {
        "input_json": str(args.input_json),
        "schema": schema_stats(map_data),
        "dangling_endpoints": dangling_endpoint_stats(roads, args.types, args.distance_threshold),
    }
    if args.xodr:
        report["xodr_coverage"] = xodr_coverage_stats(
            roads, args.xodr, args.types, args.sample_spacing, args.distance_threshold
        )

    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n", encoding="utf-8")

    if report["schema"]["errors"] or report["schema"]["degenerate_road_count"] or report["schema"]["nonfinite_road_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
