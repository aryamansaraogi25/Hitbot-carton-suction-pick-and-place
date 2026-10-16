# HITBOT Pick and Place

Python tools for a HITBOT S922 cell with an Intel RealSense D415 and a 4-pad
suction gripper. The main pipeline detects cartons from top-down depth data,
transforms the grasp pose through hand-eye calibration, and commands guarded
Cartesian moves through the HITBOT TCP API.

## Hardware

- HITBOT S922 / Z-Arm robot controller reachable over TCP
- Intel RealSense D415 mounted eye-in-hand on the robot flange
- 4-pad suction gripper driven from DO0
- Fixed flat work surface: table, pallet, or conveyor
- ChArUco board for hand-eye calibration

Default robot connection values live in `robot_config.py`:

```python
ARM_IP = "192.168.58.2"
ARM_PORT = 8080
```

`handeye_calibrate.py` and `weld_teach.py` also define `ARM_IP` locally, so
update those files too if the controller address changes.

## Safety

These scripts can move real hardware. Before running any motion script:

- Put the robot in AUTO/REMOTE mode from the pendant.
- Keep the physical e-stop in hand.
- Confirm the workcell is clear.
- Run the read-only and dry-run commands first.
- Verify `handeye.json` belongs to the same camera, flange mount, gripper, and
  robot cell.

The code validates inverse kinematics before motion and limits waypoint jumps,
but those checks are not a substitute for safe robot operation.

## Quick Start

Clone the repo, create a virtual environment, and install the Python packages:

```bash
git clone <repo-url>
cd hitbotpickplace
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the Intel RealSense runtime/udev rules for your OS before using the
camera. The Python package `pyrealsense2` is listed in `requirements.txt`, but
the camera still needs the system driver stack and USB permission rules.

Install the HITBOT Python SDK so `hitbot_tcp_api.py` is importable. On the
original machine it lives at:

```text
/home/xp/z-arms922/hitbot_ws
```

On another machine, either place the SDK there or export its path:

```bash
export PYTHONPATH=/path/to/z-arms922/hitbot_ws:$PYTHONPATH
```

## Recommended First Run

1. Verify the RealSense feed:

   ```bash
   python test.py
   ```

2. Check carton detection with no robot motion:

   ```bash
   python carton_pose.py
   ```

   Controls: `q` quits, `v` toggles the Open3D cloud view, and `s` saves
   `carton_snapshot.png` plus `carton_poses.json`.

3. Run the headless diagnostic if tuning detection thresholds:

   ```bash
   python diag_carton.py
   ```

   This writes `diag_carton_color.png`.

4. Calibrate hand-eye if this is a new camera mount or robot cell:

   ```bash
   python handeye_calibrate.py --square-mm 25 --marker-mm 18 --squares 5 7
   ```

   The first run is a dry run and does not move the robot. If the plan is safe:

   ```bash
   python handeye_calibrate.py --square-mm 25 --marker-mm 18 --squares 5 7 --go
   ```

   Successful calibration writes `handeye.json`.

5. Confirm base-frame grasp poses without motion:

   ```bash
   python detect_to_base.py
   ```

6. Run a hover test before any pick:

   ```bash
   python hover_test.py 30
   ```

   The number is pad clearance above the carton top in millimetres.

7. Dry-run the pick plan:

   ```bash
   python pick_carton.py --dry
   ```

8. Run the full pick/place cycle only after the dry run and hover test look
   correct:

   ```bash
   python pick_carton.py
   ```

## Main Scripts

| File | Purpose |
| --- | --- |
| `robot_config.py` | Shared robot IP, gripper geometry, hand-eye loading, and transform helpers. |
| `carton_pose.py` | Live top-down carton detection and 6D pose estimation from RealSense depth. |
| `diag_carton.py` | One-frame headless carton diagnostic for threshold tuning. |
| `handeye_calibrate.py` | Eye-in-hand calibration using a fixed ChArUco board. |
| `handeye_solver.py` | Local AX=XB hand-eye solver used by the calibration script. |
| `detect_to_base.py` | Read-only conversion from camera-frame carton poses to robot base poses. |
| `hover_test.py` | Moves pads above the detected carton, holds briefly, then returns. |
| `pick_carton.py` | Full detect, hover, descend, vacuum, lift, turn, place, and return sequence. |
| `track_and_pick.py` | Advanced track-and-commit routine for cartons that may shift before pick. |
| `weld_teach.py` | Tkinter teach/replay pendant for motion-only weld paths. |
| `test.py` | Basic RealSense point-cloud/pose visualization test. |

## Calibration and Generated Files

- `handeye.json` is the active flange-to-camera calibration loaded by motion
  scripts. Regenerate it when the camera mount, flange, or robot cell changes.
- `handeye_pre_zfix.json` is an older calibration snapshot kept for reference.
- `carton_snapshot.png`, `carton_poses.json`, and `diag_carton_color.png` are
  debug outputs and are ignored by git.
- `*.weldpath.json` files are local taught weld paths. Commit one explicitly
  with `git add -f` only if it should be shared.

## Vacuum Wiring

`pick_carton.py` drives DO0 through `SetDO`. Current wiring is inverted:

- `vacuum_on()` sends `DO0 -> 0`
- `vacuum_off()` sends `DO0 -> 1`

If the valve wiring moves from normally-closed to normally-open relay contacts,
flip those two helper functions.

## Troubleshooting

- `arm connect failed`: check controller IP, network cable, and AUTO/REMOTE mode.
- `robot errcode`: the pendant mode, enable state, or controller state is not
  ready for remote commands.
- `handeye.json missing`: run `handeye_calibrate.py` or copy the correct
  calibration for this exact hardware setup.
- `no support plane`: aim the camera more directly at the flat work surface and
  make sure depth frames include the table/conveyor around the carton.
- `nothing above the surface`: place a carton in view or tune the height and
  clustering constants near the top of `carton_pose.py`.
- `GetInverseKin does not reproduce the current joints`: do not move; the API
  format, robot model, or Euler convention needs investigation.
