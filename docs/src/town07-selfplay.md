# Town07 Self-Play

Build a PufferDrive map from CARLA Town07 OpenDRIVE and train policy-controlled agents on it:

```bash
./scripts/train_town07_selfplay.sh
```

The script downloads `Town07.xodr` from:

```text
https://raw.githubusercontent.com/carla-simulator/opendrive-test-files/master/OpenDrive/Town07.xodr
```

It converts driving lanes into `resources/drive/binaries/town07_selfplay/map_000.bin`, then starts:

```bash
python3 -m pufferlib.pufferl train puffer_drive_town07
```

When `data_utils/carla/carla_py123d/Town07.json` is present, the converter uses that CARLA/pyxodr-derived road geometry for better junction and roundabout fidelity. It still supplements missing short `road_line` connectors from the OpenDRIVE parser when they are not near existing map markings, but `road_edge` is no longer taken from lane centerlines directly. Instead, the converter unions the lane surfaces and derives `road_edge` from the resulting outer and inner bounds, which keeps junction boundaries consistent with the roads that form them. To force the lightweight OpenDRIVE fallback parser:

```bash
uv run --no-sync python scripts/build_town07_drive_maps.py --no-road-json
```

To keep the CARLA/pyxodr geometry but disable the extra OpenDRIVE road-line connectors:

```bash
uv run --no-sync python scripts/build_town07_drive_maps.py --no-xodr-road-line-supplement
```

To export the generated map in the original Drive JSON dataset format:

```bash
uv run --no-sync python scripts/build_town07_drive_maps.py \
  --output-json data/processed/validation/town07_selfplay/Town07_0.json
```

Validate the JSON schema and check whether OpenDRIVE road-line/road-edge segments are covered:

```bash
uv run --no-sync python scripts/validate_drive_json_map.py \
  data/processed/validation/town07_selfplay/Town07_0.json \
  --xodr data_utils/carla/opendrive/Town07.xodr \
  --report-json data/processed/validation/town07_selfplay/Town07_validation_report.json
```

Use uv for the project environment. On a CUDA 12.8 machine:

```bash
uv venv --python 3.10
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install 'numpy<2.0' Cython wheel
uv pip install --no-build-isolation -e .
```

For a short GPU smoke run, pass normal PufferLib overrides:

```bash
./scripts/train_town07_selfplay.sh \
  --train.total-timesteps=1024 \
  --train.batch-size=1024 \
  --train.minibatch-size=1024 \
  --train.max-minibatch-size=1024 \
  --vec.num-workers=1 \
  --vec.num-envs=1 \
  --env.num-agents=32
```

Generate a quick MP4 visualization of the current map with:

```bash
uv run --no-sync python -m pufferlib.pufferl eval puffer_drive_town07 \
  --train.device=cuda \
  --env.render-mode=1 \
  --env.num-agents=32 \
  --eval.wosac-realism-eval=False
```
