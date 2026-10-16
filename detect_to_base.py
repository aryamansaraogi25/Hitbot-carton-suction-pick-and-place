"""End-to-end (read-only, no motion): detect cartons and print their grasp poses
in ROBOT BASE coordinates, using the live arm pose + handeye.json.

This is what a pick routine consumes:
    flange target = grasp pose backed off by TOOL_LENGTH_M along approach.
"""
import sys
import importlib.util

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/xp/z-arms922/hitbot_ws")
import robot_config as rc

spec = importlib.util.spec_from_file_location("cp", "carton_pose.py")
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

X, meta = rc.load_handeye()
if X is None:
    print("handeye.json missing — run handeye_calibrate.py first"); raise SystemExit
conv = meta.get("euler_convention", "ZYX")
print(f"hand-eye loaded (residual {meta['residual_mm']:.1f} mm, conv {conv})")

# --- arm pose (read-only) ---
from hitbot_tcp_api import HitbotTCPInterface
arm = HitbotTCPInterface(ip=rc.ARM_IP)
if not arm.connect():
    raise SystemExit("arm connect failed")
resp = arm._send_cmd(377, "GetActualTCPPose()").split("III")[4]
pose = [float(v) for v in resp.split(",")]
T_bf = rc.arm_pose_to_T(pose, conv)
print(f"flange pose: {[round(v,1) for v in pose]}")

# --- one frame -> cartons (camera frame) ---
# The D415 sometimes times out on the first frames right after a re-open
# (previous process still releasing the USB pipe). Retry instead of dying.
import time
pipeline = None
for attempt in range(3):
    try:
        pipeline, align, intr, depth_scale, filters = cp.start_camera()
        for _ in range(30):
            pipeline.wait_for_frames()
        break
    except RuntimeError as e:
        print(f"  camera not ready ({e}); retrying in 3 s ...")
        if pipeline is not None:
            try: pipeline.stop()
            except Exception: pass
            pipeline = None
        time.sleep(3)
else:
    raise SystemExit("camera failed to start after 3 attempts")
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
n, d, frac = cp.fit_support_plane(pcd)
if n is None:
    raise SystemExit("no support plane in view")
u1, u2 = cp.plane_basis(n)
h = P @ n + d
m = (h > cp.MIN_BOX_HEIGHT) & (h < cp.MAX_BOX_HEIGHT)
if m.sum() < cp.DBSCAN_MIN_PTS:
    raise SystemExit("nothing above the surface")
Pa, Ha = P[m], h[m]
cl = o3d.geometry.PointCloud(); cl.points = o3d.utility.Vector3dVector(Pa)
lab = np.array(cl.cluster_dbscan(cp.DBSCAN_EPS, cp.DBSCAN_MIN_PTS))

found = 0
for k in range(lab.max() + 1):
    mm = lab == k
    if mm.sum() < cp.MIN_CLUSTER_PTS:
        continue
    c = cp.carton_from_cluster(Pa[mm], Ha[mm], n, d, u1, u2)
    if c is None:
        continue
    found += 1
    g_cam, R_cam, s = c["top_centre"], c["R"], c["size"]
    g_base, R_base = rc.carton_to_base(T_bf, X, g_cam, R_cam)
    ok, why = rc.graspable(s)

    # flange target: back off TOOL_LENGTH along the grasp z-axis (approach)
    flange_target = g_base + R_base[:, 2] * rc.TOOL_LENGTH_M

    print(f"\ncarton {found}:  size {s[0]*1000:.0f}x{s[1]*1000:.0f}x{s[2]*1000:.0f} mm"
          f"   [{'GRASPABLE' if ok else 'NO-GRASP: ' + why}]")
    print(f"  camera frame: ({g_cam[0]:+.3f}, {g_cam[1]:+.3f}, {g_cam[2]:+.3f}) m")
    print(f"  BASE frame  : ({g_base[0]*1000:+.1f}, {g_base[1]*1000:+.1f}, "
          f"{g_base[2]*1000:+.1f}) mm")
    print(f"  flange target (pads on box): ({flange_target[0]*1000:+.1f}, "
          f"{flange_target[1]*1000:+.1f}, {flange_target[2]*1000:+.1f}) mm")

print(f"\n{found} carton(s).")
