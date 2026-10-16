"""
HOVER TEST: detect the carton, transform to base frame, and position the suction
pads HOVER_MM above the carton top — no contact, no vacuum. Visual check that the
whole chain (camera -> hand-eye -> arm) lands centred over the box.

Safety design
-------------
* The arm's own GetInverseKin() is validated READ-ONLY first: we feed it the
  CURRENT pose and require it to return the CURRENT joints. If the protocol or
  convention is off, we find out without anything moving.
* The hover keeps the arm's CURRENT wrist orientation (already tool-down) and only
  translates — no orientation surprises on the first motion.
* Joint deltas are bounded; anything > MAX_JOINT_STEP_DEG aborts.
* Motion is slow (4 s out, 4 s back), returns to start, e-stop stays your job.
"""

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

HOVER_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0   # pad clearance above carton top
MAX_JOINT_STEP_DEG = 45.0       # abort if IK asks any joint to move more than this
HOLD_S = 6.0                    # hover dwell so you can eyeball it


def query(arm, cmd):
    resp = arm._send_cmd(377, cmd).split("III")
    if len(resp) < 6 or "errcode" in resp[4]:
        raise RuntimeError(f"{cmd} -> {resp[4] if len(resp)>=6 else resp!r}")
    return [float(v) for v in resp[4].split(",")]


def main():
    X, meta = rc.load_handeye()
    if X is None:
        raise SystemExit("handeye.json missing")
    conv = meta.get("euler_convention", "ZYX")

    arm = HitbotTCPInterface(ip=rc.ARM_IP)
    if not arm.connect():
        raise SystemExit("arm connect failed")
    j0 = query(arm, "GetActualJointPosDegree()")
    p0 = query(arm, "GetActualTCPPose()")
    print(f"start joints: {[round(v,2) for v in j0]}")
    print(f"start TCP   : {[round(v,2) for v in p0]}")

    # ---- 1. validate GetInverseKin read-only against the CURRENT pose ----
    ik_ok = False
    ik_cmd_fmt = None
    for fmt in ("GetInverseKin(0,{x},{y},{z},{rx},{ry},{rz},-1)",
                "GetInverseKin(0,{x},{y},{z},{rx},{ry},{rz})"):
        try:
            cmd = fmt.format(x=p0[0], y=p0[1], z=p0[2], rx=p0[3], ry=p0[4], rz=p0[5])
            jik = query(arm, cmd)
            err = max(abs(a - b) for a, b in zip(jik, j0))
            print(f"IK self-check [{fmt.split('(')[0]}({fmt.count(',')+1} args)]: "
                  f"max joint err {err:.3f} deg")
            if err < 0.5:
                ik_ok, ik_cmd_fmt = True, fmt
                break
        except Exception as e:
            print(f"  IK format failed: {str(e)[:70]}")
    if not ik_ok:
        raise SystemExit("GetInverseKin does not reproduce the current joints — "
                         "NOT moving. (Protocol variant needs investigating.)")
    print("IK validated against current pose ✓")

    # ---- 2. detect the carton in base frame ----
    T_bf = rc.arm_pose_to_T(p0, conv)
    pipeline = None
    for attempt in range(3):
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
        raise SystemExit("camera failed")
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
        raise SystemExit("no support plane")
    u1, u2 = cp.plane_basis(n)
    h = P @ n + d
    m = (h > cp.MIN_BOX_HEIGHT) & (h < cp.MAX_BOX_HEIGHT)
    Pa, Ha = P[m], h[m]
    cl = o3d.geometry.PointCloud(); cl.points = o3d.utility.Vector3dVector(Pa)
    lab = np.array(cl.cluster_dbscan(cp.DBSCAN_EPS, cp.DBSCAN_MIN_PTS))
    best = None
    for k in range(lab.max() + 1):
        mm = lab == k
        if mm.sum() < cp.MIN_CLUSTER_PTS:
            continue
        c = cp.carton_from_cluster(Pa[mm], Ha[mm], n, d, u1, u2)
        if c is None:
            continue
        ok, _ = rc.graspable(c["size"])
        if ok and (best is None or c["top_centre"][2] < best["top_centre"][2]):
            best = c
    if best is None:
        raise SystemExit("no graspable carton detected")

    g_base, R_base = rc.carton_to_base(T_bf, X, best["top_centre"], best["R"])
    s = best["size"]
    print(f"\ncarton: {s[0]*1000:.0f}x{s[1]*1000:.0f}x{s[2]*1000:.0f} mm  "
          f"top centre BASE ({g_base[0]*1000:+.1f}, {g_base[1]*1000:+.1f}, "
          f"{g_base[2]*1000:+.1f}) mm")

    # ---- 3. hover pose: translate only, keep current wrist orientation ----
    hover = [g_base[0] * 1000.0,
             g_base[1] * 1000.0,
             g_base[2] * 1000.0 + rc.TOOL_LENGTH_M * 1000.0 + HOVER_MM,
             p0[3], p0[4], p0[5]]
    print(f"hover TCP target: {[round(v,1) for v in hover]}  "
          f"(pads {HOVER_MM:.0f} mm above the box)")

    # Break the path into short Cartesian segments and IK each one. A single big
    # joint-space move sweeps the tool along an arc; chained waypoints keep it on
    # a near-straight line, and per-segment joint deltas stay small and checkable.
    N_WP = 8
    waypoints = []
    for i in range(1, N_WP + 1):
        a = i / N_WP
        wp = [p0[k] + a * (hover[k] - p0[k]) for k in range(3)] + hover[3:]
        waypoints.append(wp)

    joint_path = [j0]
    for wp in waypoints:
        jt = query(arm, ik_cmd_fmt.format(x=wp[0], y=wp[1], z=wp[2],
                                          rx=wp[3], ry=wp[4], rz=wp[5]))
        step = max(abs(a - b) for a, b in zip(jt, joint_path[-1]))
        if step > MAX_JOINT_STEP_DEG:
            raise SystemExit(f"IK jump {step:.1f} deg between adjacent waypoints "
                             f"(configuration flip) — aborting before motion.")
        joint_path.append(jt)
    total = [round(abs(a - b), 1) for a, b in zip(joint_path[-1], j0)]
    per_seg = max(max(abs(a - b) for a, b in zip(joint_path[i], joint_path[i + 1]))
                  for i in range(len(joint_path) - 1))
    print(f"path: {N_WP} waypoints, total joint travel {total} deg, "
          f"max per-segment {per_seg:.1f} deg")

    # ---- 4. move, hold, verify, return ----
    print("\nmoving to hover along straight-line waypoints ...")
    for jt in joint_path[1:]:
        arm.move_to_joints_smoothly(jt, duration_sec=1.2, dt=0.1)
    time.sleep(0.6)
    p_now = query(arm, "GetActualTCPPose()")
    print(f"reached TCP : {[round(v,1) for v in p_now]}")
    err_xy = np.hypot(p_now[0] - hover[0], p_now[1] - hover[1])
    err_z = abs(p_now[2] - hover[2])
    print(f"controller says: XY off {err_xy:.1f} mm, Z off {err_z:.1f} mm")
    print(f"\n>>> LOOK NOW: are the pads centred over the box? "
          f"Holding {HOLD_S:.0f} s ...")
    time.sleep(HOLD_S)

    print("returning to start along the same path ...")
    for jt in reversed(joint_path[:-1]):
        arm.move_to_joints_smoothly(jt, duration_sec=1.2, dt=0.1)
    time.sleep(0.5)
    print("done.")


if __name__ == "__main__":
    main()
