from pathlib import Path

from scripts.build_town07_drive_maps import (
    build_map_data,
    build_lane_surface,
    driving_lane_polylines,
    load_clipgt_elements,
    load_road_json_elements,
    opendrive_map_elements,
    road_edges_from_lane_surface,
    save_map_binary,
)
from scripts.convert_drive_bin import parse_current


SIMPLE_XODR = """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <road name="Road 0" length="40" id="0" junction="-1">
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="40">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneOffset s="0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0">
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0" a="3.0" b="0" c="0" d="0"/>
            <userData><vectorLane travelDir="backward"/></userData>
          </lane>
        </left>
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.0" b="0" c="0" d="0"/>
            <userData><vectorLane travelDir="forward"/></userData>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def test_xodr_to_drive_binary(tmp_path: Path):
    xodr = tmp_path / "simple.xodr"
    xodr.write_text(SIMPLE_XODR)

    lanes = driving_lane_polylines(xodr, sample_spacing=5.0)
    assert len(lanes) == 2
    assert lanes[0]["points"][0]["x"] > lanes[0]["points"][-1]["x"]
    assert lanes[1]["points"][0]["x"] < lanes[1]["points"][-1]["x"]

    elements = opendrive_map_elements(xodr, sample_spacing=5.0)
    assert [elem["type"] for elem in elements].count("lane") == 2
    assert [elem["type"] for elem in elements].count("road_line") == 1
    assert [elem["type"] for elem in elements].count("road_edge") == 2

    map_data = build_map_data(elements, num_agents=2, speed=3.0, dt=0.1)
    output = tmp_path / "map_000.bin"
    save_map_binary(map_data, output, unique_map_id=0, dt=0.1)

    parsed = parse_current(output)
    assert parsed["bytes_remaining"] == 0
    assert parsed["scenario_id"] == "Town07"
    assert parsed["num_objects"] == 2
    assert parsed["num_roads"] == 5
    assert parsed["object_type_counts"] == {1: 2}
    assert parsed["road_type_counts"] == {4: 2, 5: 1, 6: 2}


def test_road_json_to_drive_binary(tmp_path: Path):
    road_json = tmp_path / "roads.json"
    road_json.write_text(
        """{
          "roads": [
            {"id": 10, "map_element_id": 1, "type": "lane", "geometry": [
              {"x": 0, "y": 0, "z": 0}, {"x": 20, "y": 0, "z": 0}
            ]},
            {"id": 11, "map_element_id": 1, "type": "road_line", "geometry": [
              {"x": 0, "y": 1, "z": 0}, {"x": 20, "y": 1, "z": 0}
            ]},
            {"id": 12, "map_element_id": 1, "type": "road_edge", "geometry": [
              {"x": 0, "y": -2, "z": 0}, {"x": 20, "y": -2, "z": 0}
            ]}
          ]
        }"""
    )

    elements = load_road_json_elements(road_json)
    assert [elem["type"] for elem in elements] == ["lane", "road_line"]

    lane_surface = build_lane_surface(elements)
    derived_edges = road_edges_from_lane_surface(lane_surface)
    assert derived_edges
    assert all(edge["type"] == "road_edge" for edge in derived_edges)

    elements = [elem for elem in elements if elem["type"] != "road_edge"] + derived_edges

    map_data = build_map_data(elements, num_agents=1, speed=3.0, dt=0.1)
    output = tmp_path / "map_000.bin"
    save_map_binary(map_data, output, unique_map_id=0, dt=0.1)

    parsed = parse_current(output)
    assert parsed["bytes_remaining"] == 0
    assert parsed["num_objects"] == 1
    assert parsed["num_roads"] == 3
    assert parsed["road_type_counts"] == {4: 1, 5: 1, 6: 1}


def test_clipgt_to_drive_binary(tmp_path: Path):
    pq = __import__("pytest").importorskip("pyarrow.parquet")
    pa = __import__("pytest").importorskip("pyarrow")

    clipgt_dir = tmp_path / "clipgt"
    clipgt_dir.mkdir()

    key = {
        "clip_id": "unit",
        "label_class_id": "lanelet2:autoware:v0",
        "map_id": "100",
        "map_id_version": "1",
    }
    lane_rows = [
        {
            "key": key,
            "lane": {
                "left_rail": [{"x": 0.0, "y": 1.0, "z": 0.0}, {"x": 20.0, "y": 1.0, "z": 0.0}],
                "right_rail": [{"x": 0.0, "y": -1.0, "z": 0.0}, {"x": 20.0, "y": -1.0, "z": 0.0}],
            },
            "version": 1,
        }
    ]
    lane_line_rows = [
        {
            "key": {**key, "map_id": "101"},
            "lane_line": {
                "line_rail": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 20.0, "y": 0.0, "z": 0.0}],
            },
            "version": 1,
        }
    ]
    road_boundary_rows = [
        {
            "key": {**key, "map_id": "102"},
            "road_boundary": {
                "location": [{"x": 0.0, "y": -2.0, "z": 0.0}, {"x": 20.0, "y": -2.0, "z": 0.0}],
            },
            "version": 1,
        }
    ]

    pq.write_table(pa.Table.from_pylist(lane_rows), clipgt_dir / "lane.parquet")
    pq.write_table(pa.Table.from_pylist(lane_line_rows), clipgt_dir / "lane_line.parquet")
    pq.write_table(pa.Table.from_pylist(road_boundary_rows), clipgt_dir / "road_boundary.parquet")

    elements = load_clipgt_elements(clipgt_dir, sample_spacing=5.0)
    assert [elem["type"] for elem in elements] == ["lane", "road_line", "road_edge"]
    assert elements[0]["points"][0] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert elements[0]["points"][-1] == {"x": 20.0, "y": 0.0, "z": 0.0}

    map_data = build_map_data(
        elements,
        num_agents=1,
        speed=3.0,
        dt=0.1,
        scenario_id="Shinjuku",
        metadata_source=str(clipgt_dir),
    )
    output = tmp_path / "map_000.bin"
    save_map_binary(map_data, output, unique_map_id=0, dt=0.1)

    parsed = parse_current(output)
    assert parsed["bytes_remaining"] == 0
    assert parsed["scenario_id"] == "Shinjuku"
    assert parsed["num_objects"] == 1
    assert parsed["num_roads"] == 3
    assert parsed["road_type_counts"] == {4: 1, 5: 1, 6: 1}
