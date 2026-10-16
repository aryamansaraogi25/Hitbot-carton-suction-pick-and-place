"""
FULL PICK SEQUENCE — HITBOT S922 + D415 + 4-pad suction gripper.

    detect -> hover above carton -> slow descent to contact -> [VACUUM ON]
           -> dwell -> lift -> turn base +30 deg -> descend -> [VACUUM OFF]
           -> retract -> return to start

The vacuum is a STUB (see the VACUUM section below): wire in SetDO when the pump
is plumbed. Until then the arm performs the complete motion open-loop.

Run:    python3 pick_carton.py           # full pick on the best graspable carton
        python3 pick_carton.py --dry     # detect + print the plan, no motion

Every Cartesian move is executed as short IK-checked waypoints (straight tool
path, no configuration flips), the same scheme validated by hover_test.py.
"""

import argparse
import sys
import time
import importlib.util

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/xp/z-arms922/hitbot_ws")
import robot_config as rc
from hitbot_tcp_api import HitbotTCPInterface

spec = importlib.util.spec_from_file_location("cp", "carton_pose.py")
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

# ---------------- pick parameters ----------------
HOVER_MM = 30.0        # approach clearance above the carton top
COMPRESSION_MM = 13.0  # how far past nominal contact to press. Tuned on hardware:
                       # 3 mm left the pads barely kissing the lid; 13 mm gives
                       # proper bellows squish for a reliable seal.
LIFT_MM = 400.0        # lift height above the carton top after grasp
TURN_DEG = 90.0       # base-joint (J1) rotation while lifted; place spot is the
                       # pick spot swung this far around the base axis. Sign picks
                       # the direction — negative turns the other way.
PLACE_DOWN_MM = 200   # how far to descend at the place spot before releasing (mm).
                       # None = go all the way back to the grasp height, i.e. set
                       # the box down on a surface level with the pick surface.
                       # Set a number (e.g. 200.0) to descend only that far from
                       # the lifted height — use this when placing somewhere
                       # higher/lower than the pick surface.
VACUUM_DWELL_S = 0.8   # let vacuum build before lifting
MAX_JOINT_STEP_DEG = 90.0
SEG_TIME_FAST = 1.2    # s per waypoint segment, transit moves
SEG_TIME_SLOW = 0.8    # s per waypoint segment, contact-zone moves (short segs)


# =====================================================================
# VACUUM — DO0 drives the relay module (LOW-trigger, sinking output),
# relay contact switches the 6 V valve. Verified with the toggle tests.
# A failed SetDO must not strand the arm mid-sequence, so errors are
# reported loudly but motion continues (open loop, no vacuum feedback).
# =====================================================================
VACUUM_DO = 0


def _set_do(arm, state):
    try:
        resp = arm._send_cmd(204, f"SetDO({VACUUM_DO},{state},0,0)")
        parts = resp.split("III")
        ok = len(parts) >= 6 and parts[4].strip() == "1"
        if not ok:
            print(f"  !! SetDO({VACUUM_DO},{state}) unexpected reply: "
                  f"{parts[4] if len(parts) >= 6 else resp!r}")
        return ok
    except Exception as e:
        print(f"  !! SetDO({VACUUM_DO},{state}) FAILED: {e} — "
              f"vacuum state unknown, watch the gripper")
        return False


# NOTE: inverted per observed hardware behaviour — DO0 LOW opens the valve.
# (Likely the valve sits on the relay's NC contact.) Consequence: when the
# controller boots or DOs reset to 0, the vacuum is ON by default. Moving the
# valve wire from the NC to the NO contact would restore vacuum-off-by-default;
# flip these two calls back if that gets rewired.
def vacuum_on(arm):
    print("  [VACUUM ON]  DO0 -> 0")
    _set_do(arm, 0)


def vacuum_off(arm):
    print("  [VACUUM OFF] DO0 -> 1")
    _set_do(arm, 1)


# ---------------------------------------------------------------------
def query(arm, cmd):
    resp = arm._send_cmd(377, cmd).split("III")
    if len(resp) < 6 or "errcode" in resp[4]:
        raise RuntimeError(f"{cmd} -> {resp[4] if len(resp) >= 6 else resp!r}")
    return [float(v) for v in resp[4].split(",")]


IK_FMT = "GetInverseKin(0,{x},{y},{z},{rx},{ry},{rz},-1)"


def validate_ik(arm, j0, p0):
    jik = query(arm, IK_FMT.format(x=p0[0], y=p0[1], z=p0[2],
                                   rx=p0[3], ry=p0[4], rz=p0[5]))
    err = max(abs(a - b) for a, b in zip(jik, j0))
    if err > 0.5:
        raise SystemExit(f"IK self-check failed ({err:.2f} deg) — not moving")


def cartesian_path(arm, p_from, p_to, n_wp, j_seed):
    """Straight-line waypoints p_from -> p_to (orientation of p_to), IK-checked."""
    path = [list(j_seed)]
    for i in range(1, n_wp + 1):
        a = i / n_wp
        wp = [p_from[k] + a * (p_to[k] - p_from[k]) for k in range(3)] + list(p_to[3:])
        jt = query(arm, IK_FMT.format(x=wp[0], y=wp[1], z=wp[2],
                                      rx=wp[3], ry=wp[4], rz=wp[5]))
        step = max(abs(x - y) for x, y in zip(jt, path[-1]))
        if step > MAX_JOINT_STEP_DEG:
            raise SystemExit(f"IK jump {step:.1f} deg mid-path — aborting")
        path.append(jt)
    return path[1:]


def run_path(arm, path, seg_time):
    for jt in path:
        arm.move_to_joints_smoothly(jt, duration_sec=seg_time, dt=0.1)


def tcp(arm):
    return query(arm, "GetActualTCPPose()")


# ---------------------------------------------------------------------
def detect_carton(T_bf, X):
    """One frame -> best graspable carton in base frame, or None."""
    pipeline = None
    for _ in range(3):
        try:
            pipeline, align, intr, depth_scale, filters = cp.start_camera()
            for _ in range(30):
                pipeline.wait_for_frames()
            break
        except RuntimeError:
            if pipeline:
                try: pipeline.stop()
                except Exception: pass
                pipeline = None
            time.sleep(3)
    if pipeline is None:
        raise SystemExit("camera failed to start")
    frames = align.process(pipeline.wait_for_frames())
    df = frames.get_depth_frame()
    for f in filters:
        df = f.process(df)
    depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
    pipeline.stop()

    pts = cp.deproject(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy, cp.STRIDE)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(cp.VOXEL_SIZE)
    P = np.asarray(pcd.points)
    n, d, _ = cp.fit_support_plane(pcd)
    if n is None:
        return None
    u1, u2 = cp.plane_basis(n)
    h = P @ n + d
    m = (h > cp.MIN_BOX_HEIGHT) & (h < cp.MAX_BOX_HEIGHT)
    if m.sum() < cp.DBSCAN_MIN_PTS:
        return None
    Pa, Ha = P[m], h[m]
    cl = o3d.geometry.PointCloud(); cl.points = o3d.utility.Vector3dVector(Pa)
    lab = np.array(cl.cluster_dbscan(cp.DBSCAN_EPS, cp.DBSCAN_MIN_PTS))
    best = None
    for k in range(lab.max() + 1):
        mm = lab == k
        if mm.sum() < cp.MIN_CLUSTER_PTS:
            continue
        c = cp.carton_from_cluster(Pa[mm], Ha[mm], n, d, u1, u2)
        if c is None or not rc.graspable(c["size"])[0]:
            continue
        if best is None or c["top_centre"][2] < best["top_centre"][2]:
            best = c
    if best is None:
        return None
    g_base, R_base = rc.carton_to_base(T_bf, X, best["top_centre"], best["R"])
    return {"grasp_base_mm": g_base * 1000.0, "R_base": R_base,
            "size_mm": best["size"] * 1000.0}


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="detect + plan only")
    a = ap.parse_args()

    X, meta = rc.load_handeye()
    if X is None:
        raise SystemExit("handeye.json missing")
    conv = meta.get("euler_convention", "ZYX")

    arm = HitbotTCPInterface(ip=rc.ARM_IP)
    if not arm.connect():
        raise SystemExit("arm connect failed")
    j0 = query(arm, "GetActualJointPosDegree()")
    p0 = tcp(arm)
    validate_ik(arm, j0, p0)
    vacuum_off(arm)      # DO0 low = suction with this wiring; ensure released state
    T_bf = rc.arm_pose_to_T(p0, conv)
    print(f"start TCP: {[round(v, 1) for v in p0]}")

    c = detect_carton(T_bf, X)
    if c is None:
        raise SystemExit("no graspable carton in view")
    g = c["grasp_base_mm"]
    s = c["size_mm"]
    print(f"carton {s[0]:.0f}x{s[1]:.0f}x{s[2]:.0f} mm, top centre "
          f"({g[0]:+.1f}, {g[1]:+.1f}, {g[2]:+.1f}) mm")

    tool = rc.TOOL_LENGTH_M * 1000.0
    ori = p0[3:]                       # keep validated tool-down orientation
    p_hover = [g[0], g[1], g[2] + tool + HOVER_MM] + ori
    p_grasp = [g[0], g[1], g[2] + tool - COMPRESSION_MM] + ori
    p_lift = [g[0], g[1], g[2] + tool + LIFT_MM] + ori
    print(f"plan: hover z={p_hover[2]:.1f} -> grasp z={p_grasp[2]:.1f} "
          f"(descent {p_hover[2]-p_grasp[2]:.0f} mm) -> lift z={p_lift[2]:.1f}")
    if a.dry:
        print("(dry run — no motion)"); return

    # ---- 1. HOVER ----
    print("\n[1/6] hover ...")
    run_path(arm, cartesian_path(arm, p0, p_hover, 8, j0), SEG_TIME_FAST)
    j_h = query(arm, "GetActualJointPosDegree()")

    # ---- 2. DESCEND (the only descent at the pick spot) ----
    print("[2/6] descend to contact ...")
    run_path(arm, cartesian_path(arm, p_hover, p_grasp, 5, j_h), SEG_TIME_SLOW)
    p_now = tcp(arm)
    print(f"      at contact: z={p_now[2]:.1f} mm "
          f"(pad face ~{p_now[2]-tool-g[2]:+.1f} mm vs box top)")

    # ---- 3. GRASP ----
    print("[3/6] grasp ...")
    vacuum_on(arm)
    time.sleep(VACUUM_DWELL_S)
    j_g = query(arm, "GetActualJointPosDegree()")

    # ---- 4. LIFT ----
    print("[4/6] lift ...")
    run_path(arm, cartesian_path(arm, p_grasp, p_lift, 5, j_g), SEG_TIME_SLOW)

    # ---- 5. TURN (swing the carton around the base axis at lift height) ----
    print(f"[5/6] turn {TURN_DEG:+.0f} deg ...")
    j_l = query(arm, "GetActualJointPosDegree()")
    j_turn = list(j_l)
    j_turn[0] += TURN_DEG
    arm.move_to_joints_smoothly(j_turn, duration_sec=3.0, dt=0.1)

    # ---- 6. PLACE (straight down at the turned spot, release, retract) ----
    print("[6/6] place ...")
    p_turn = tcp(arm)
    j_t = query(arm, "GetActualJointPosDegree()")
    down = PLACE_DOWN_MM if PLACE_DOWN_MM is not None else p_turn[2] - p_grasp[2]
    print(f"      descending {down:.0f} mm to z={p_turn[2]-down:.1f}")
    p_place = [p_turn[0], p_turn[1], p_turn[2] - down] + p_turn[3:]
    run_path(arm, cartesian_path(arm, p_turn, p_place, 5, j_t), SEG_TIME_SLOW)
    vacuum_off(arm)
    time.sleep(0.4)
    j_p = query(arm, "GetActualJointPosDegree()")
    p_clear = [p_place[0], p_place[1],
               max(p_hover[2], p_place[2] + HOVER_MM)] + p_turn[3:]
    run_path(arm, cartesian_path(arm, p_place, p_clear, 4, j_p), SEG_TIME_SLOW)
    arm.move_to_joints_smoothly(j0, duration_sec=4.0, dt=0.1)
    print("done — placed at the turned spot, back at start.")


if __name__ == "__main__":
    main()
