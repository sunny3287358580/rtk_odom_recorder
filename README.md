# RTK ENU-to-world trajectory registration

The offline tool accepts recorder CSV files directly. It fits a planar frame
transform (yaw, X/Y translation, Z offset), then estimates the timestamp mapping
from spatial correspondences. RTK heading is not used.

## Run

From `/home/sunny/ros2/Super-LIO`:

```bash
source install/setup.bash
ros2 run rtk_odom_recorder align_trajectories.py \
  --lio /home/sunny/lio_data/relocation/20261020_155540_821111/lio_odom.csv \
  --rtk /home/sunny/lio_data/relocation/20261020_155540_821111/rtk_fix.csv \
  --output /home/sunny/ros2/Super-LIO/rtk_alignment_results/louxia1_new/alignment.yaml
```

Alternatively, run the source script without ROS:

```bash
python3 /home/sunny/ros2/Super-LIO/src/rtk_odom_recorder/scripts/align_trajectories.py --help
```

Dependencies: Python 3, NumPy, SciPy, Matplotlib. They are declared in the package.
`--no-plots` skips Matplotlib and PNG generation. Each run must use a fresh output
prefix; existing outputs are never overwritten.

## Algorithm and conventions

1. Read `timestamp,x,y,z,qx,qy,qz,qw` from LIO and
   `timestamp,status,east,north,up` from RTK. Filter RTK by
   `status >= --rtk-min-status` (default 2), discard nonfinite rows, deduplicate
   timestamps, and reject backwards time jumps.
2. Resample both XY paths by traveled distance (default 600 samples). Initial
   rotation/translation comes from equal progress along each path, without
   comparing absolute timestamps.
3. Alternate ordered, banded dynamic time warping and a rigid rotation/translation
   fit. Position and local travel direction determine the matching cost.
   Monotonic correspondences prevent freely jumping between outward/return legs
   or repeated visits at intersections. Repeated DTW indices are downweighted in
   the rigid fit. Stop when correspondences converge or after 12 iterations.
4. Fit a robust timestamp mapping using geometric matches:

   ```text
   t_lio = lio_reference_sec + shift_sec
           + scale * (t_rtk - rtk_reference_sec)
   ```

   `--time-model affine` estimates offset and scale; `--time-model offset` fixes
   scale to 1. The full time difference at a point is `t_lio - t_rtk`, not merely
   `shift_sec`. Reported drift is `(scale - 1) * 1e6` ppm. Large epoch differences
   do not require a wide seconds-based search.
5. Apply the fixed spatial transform and the estimated time mapping to the
   original RTK samples. Interpolate LIO only within `--max-gap` (default 0.2 s).
   Report these synchronized errors separately from the geometry fitting errors.

The printed/saved 4x4 matrix uses column vectors, meters, and radians:

```text
[x_world, y_world, z_world, 1]^T
  = T_world_enu * [east, north, up, 1]^T

T_world_enu =
[ cos(yaw) -sin(yaw) 0 tx ]
[ sin(yaw)  cos(yaw) 0 ty ]
[    0         0    1 tz ]
[    0         0    0  1 ]
```

The tool uses the ENU values already in the CSV. It does not recenter the RTK
origin. Use the same recorder ENU origin and LIO world frame when applying the
result. This frame transform does not identify a physical antenna-to-body lever
arm, roll/pitch extrinsics, or a body heading offset.

## Outputs

For `--output /some/new/run/alignment.yaml`:

| File | Contents |
| --- | --- |
| `alignment.yaml` | Recorder-compatible spatial transform parameters |
| `alignment.matrix.txt` | 4x4 `T_world_enu` matrix |
| `alignment.report.json` | Transform, input paths, time mapping, quality metrics and warnings |
| `alignment.matches.csv` | Geometry correspondences, both timestamps, observed/model time differences, XYZ residuals |
| `alignment.errors.csv` | RTK mapped to LIO time, interpolated LIO positions, XYZ/XY/3D residuals |
| `alignment.lio.tum` | LIO position/orientation with original LIO timestamps |
| `alignment.rtk_world.tum` | Transformed RTK positions with timestamps mapped to LIO time |
| `alignment.overview.png` | World XY overlay, geometry errors, matched time differences, synchronized errors |

RTK TUM quaternions are identity placeholders because no RTK orientation convention
has been confirmed. Use these files for **position** comparison, not orientation
error. Geometry RMSE weights distance-sampled DTW matches; synchronized RMSE
weights the original accepted RTK samples. All these residuals are on the fitting
data; they are not an independent accuracy evaluation or a full pose RPE.

For real data in `20261020_155540_821111`, the default settings produced yaw
175.222320245 degrees, geometry XY RMSE 0.395666 m and synchronized XY RMSE
0.396022 m. The corresponding artifacts are in
`/home/sunny/ros2/Super-LIO/rtk_alignment_results/louxia1_20261020_155540/`.

To show the transformed RTK path on the next replay:

```bash
ros2 launch super_lio relocation.py \
  alignment_file:=/home/sunny/ros2/Super-LIO/rtk_alignment_results/louxia1_20261020_155540/alignment.yaml
```

The recorder loads the spatial transform and publishes `/rtk/path_world`.
Its current runtime does not apply the offline timestamp mapping; corrected
timestamps are in the exported TUM/error files.

## Limits and tuning

- Both input ranges must describe the same traversal in the same direction/order.
  For partial recordings, explicitly trim with `--rtk-start`, `--rtk-end`,
  `--lio-start`, `--lio-end` (seconds from each file's first valid timestamp).
  Reversal, missing whole route sections, and multiple bag loops require separate
  runs. Straight or nearly stationary paths are rejected for joint time/space
  registration.
- The default progress band is 20%. `--match-band` changes it. A wider band
  increases the possibility of matching a different repeated section.
  `--samples` changes spatial resolution (20 to 2000). `--direction-weight`
  changes the local travel-direction cost in meters.
- At least 20 valid rows, at least 5 m RTK displacement and geometry XY RMSE at
  most 1 m are required by default. Failed fitting/export leaves no new usable
  YAML. A synchronized RMSE above the limit produces a warning rather than
  changing the successful geometry fit.
- Time drift inferred from trajectory shape is affected by spatial distortion,
  sample spacing, stops, antenna/body offsets and matching errors. The report
  includes timing residuals and first/second-half drift estimates. Do not treat
  ppm as an independently measured oscillator error.
- `--method timestamp` retains the earlier direct timestamp-pairing method for
  clocks that already share a time base. It exports YAML, matrix and JSON only.

Tests:

```bash
python3 /home/sunny/ros2/Super-LIO/src/rtk_odom_recorder/test/test_align_trajectories.py
```
