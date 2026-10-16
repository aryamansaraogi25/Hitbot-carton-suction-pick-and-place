"""
TRACK-AND-COMMIT PICK — follows a carton that drifts or gets moved, then picks
the moment it settles.

    ┌────────── loop at tracking height (camera ~0.5 m, depth valid) ─────────┐
    │  detect @ ~3 Hz (stop-and-go: capture only while the arm is still,      │
    │  so the frame and the flange pose are always in sync)                   │
    │    box moved?          -> step the arm's XY to shadow it                │
    │    stable 4 frames     -> COMMIT: descend, suck, lift                   │
    │    box lost?           -> hold position, keep looking                   │
    └──────────────────────────────────────────────────────────────────────────┘

Physics note: the D415 goes blind below ~0.45 m, so the final descent is
necessarily open-loop — this design tracks UNTIL commit, then strikes. A box
that keeps moving during the ~3 s descent will be missed (that case needs a
fixed overhead camera, different cell layout).

Vacuum: DO0 (via pick_carton.vacuum_on/off) drives the relay -> 6 V valve.
Ctrl-C at tracking height is safe: the arm just stops following.
"""

import sys
import time
import collections

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/xp/z-arms922/hitbot_ws")
import robot_config as rc
import pick_carton as pk          # reuse: query, IK_FMT, cartesian_path,
                                  # run_path, vacuum_on/off, tcp, cp
cp = pk.cp

# The camera tracks from ~0.5 m above the work surface. Depth beyond 0.9 m can
# only be floor/background — and after voxel downsampling (which weights by
# PHYSICAL area, not pixels) a distant floor can out-vote the table in the
# plane search and hijack the height reference. Crop it out entirely.
cp.MAX_DIST_M = 0.90
from hitbot_tcp_api import HitbotTCPInterface

# ---------------- tracking parameters ----------------
ALIGN_TOL_MM = 8.0      # arm XY within this of the box -> "aligned"
STABLE_TOL_MM = 5.0     # box motion below this across the window -> "stable"
STABLE_FRAMES = 4       # consecutive stable+aligned frames to commit
STEP_CAP_MM = 60.0      # max follow step per iteration (keeps moves gentle)
SETTLE_S = 0.35         # wait after each move before trusting a frame
FLUSH_FRAMES = 4        # frames to discard after motion (temporal filter lag)
TIMEOUT_S = 180.0       # give up if nothing ever stabilises


class LiveCamera:
    """Persistent pipeline — open once, detect many times (reopening the D415
    repeatedly is what wedged it earlier)."""

    def __init__(self):
        self.pipeline, self.align, self.intr, self.depth_scale, self.filters = \
            cp.start_camera()
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def detect(self, T_bf, X):
        for _ in range(FLUSH_FRAMES):
            self.pipeline.wait_for_frames()
        frames = self.align.process(self.pipeline.wait_for_frames())
        df = frames.get_depth_frame()
        for f in self.filters:
            df = f.process(df)
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * self.depth_scale

        i = self.intr
        pts = cp.deproject(depth_m, i.fx, i.fy, i.ppx, i.ppy, cp.STRIDE)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
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
        cl = o3d.geometry.PointCloud()
        cl.points = o3d.utility.Vector3dVector(Pa)
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
        g, _R = rc.carton_to_base(T_bf, X, best["top_centre"], best["R"])
        return {"g_mm": g * 1000.0, "size_mm": best["size"] * 1000.0}

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def main():
    X, meta = rc.load_handeye()
    if X is None:
        raise SystemExit("handeye.json missing")
    conv = meta.get("euler_convention", "ZYX")

    arm = HitbotTCPInterface(ip=rc.ARM_IP)
    if not arm.connect():
        raise SystemExit("arm connect failed")
    j0 = pk.query(arm, "GetActualJointPosDegree()")
    p0 = pk.tcp(arm)
    pk.validate_ik(arm, j0, p0)
    pk.vacuum_off(arm)   # DO0 low = suction with this wiring; ensure released state
    track_z = p0[2]                      # current height = validated tracking height
    ori = p0[3:]
    print(f"tracking from z={track_z:.0f} mm — move the box around; "
          f"it picks when the box holds still.  Ctrl-C to stop.")

    cam = LiveCamera()
    hist = collections.deque(maxlen=STABLE_FRAMES)
    t0 = time.time()
    committed = None

    try:
        while time.time() - t0 < TIMEOUT_S:
            time.sleep(SETTLE_S)
            p_now = pk.tcp(arm)
            T_bf = rc.arm_pose_to_T(p_now, conv)
            det = cam.detect(T_bf, X)
            if det is None:
                print("  ... no graspable carton in view")
                hist.clear()
                continue

            g = det["g_mm"]
            hist.append(g[:2].copy())
            off = float(np.hypot(g[0] - p_now[0], g[1] - p_now[1]))
            spread = (max(np.linalg.norm(a - b) for a in hist for b in hist)
                      if len(hist) > 1 else 999.0)

            if off > ALIGN_TOL_MM:
                # follow: step toward the box, capped
                dx, dy = g[0] - p_now[0], g[1] - p_now[1]
                scale = min(1.0, STEP_CAP_MM / max(off, 1e-6))
                tgt = [p_now[0] + dx * scale, p_now[1] + dy * scale,
                       track_z] + list(ori)
                print(f"  FOLLOW  box=({g[0]:+.0f},{g[1]:+.0f})  off={off:.0f}mm"
                      f"  -> step ({dx*scale:+.0f},{dy*scale:+.0f})")
                j_now = pk.query(arm, "GetActualJointPosDegree()")
                pk.run_path(arm, pk.cartesian_path(arm, p_now, tgt, 2, j_now), 0.7)
                if spread > STABLE_TOL_MM:
                    hist.clear()
                continue

            stable = len(hist) == STABLE_FRAMES and spread < STABLE_TOL_MM
            print(f"  ALIGNED off={off:.0f}mm  stability {len(hist)}/"
                  f"{STABLE_FRAMES} (spread {spread if spread<999 else 0:.0f}mm)"
                  f"{'  -> COMMIT' if stable else ''}")
            if stable:
                committed = det
                break
    except KeyboardInterrupt:
        print("\nstopped by user at tracking height.")
    finally:
        cam.stop()

    if committed is None:
        print("no stable carton within timeout — arm left at tracking height.")
        return

    # ---------- COMMIT: blind descent, grasp, lift (box must not move now) ----
    g = committed["g_mm"]
    s = committed["size_mm"]
    tool = rc.TOOL_LENGTH_M * 1000.0
    print(f"\nCOMMIT on carton {s[0]:.0f}x{s[1]:.0f}x{s[2]:.0f} mm at "
          f"({g[0]:+.0f},{g[1]:+.0f},{g[2]:+.0f})")
    p_here = pk.tcp(arm)
    p_hover = [g[0], g[1], g[2] + tool + pk.HOVER_MM] + list(ori)
    p_grasp = [g[0], g[1], g[2] + tool - pk.COMPRESSION_MM] + list(ori)
    p_lift = [g[0], g[1], g[2] + tool + pk.LIFT_MM] + list(ori)

    j_now = pk.query(arm, "GetActualJointPosDegree()")
    pk.run_path(arm, pk.cartesian_path(arm, p_here, p_hover, 6, j_now), 0.9)
    j_h = pk.query(arm, "GetActualJointPosDegree()")
    pk.run_path(arm, pk.cartesian_path(arm, p_hover, p_grasp, 4, j_h), pk.SEG_TIME_SLOW)
    pk.vacuum_on(arm)
    time.sleep(pk.VACUUM_DWELL_S)
    j_g = pk.query(arm, "GetActualJointPosDegree()")
    pk.run_path(arm, pk.cartesian_path(arm, p_grasp, p_lift, 5, j_g), pk.SEG_TIME_SLOW)
    print(f"lifted — holding {pk.HOLD_LIFTED_S:.0f} s")
    time.sleep(pk.HOLD_LIFTED_S)

    # set back down (place stub), release, home
    j_l = pk.query(arm, "GetActualJointPosDegree()")
    pk.run_path(arm, pk.cartesian_path(arm, p_lift, p_grasp, 5, j_l), pk.SEG_TIME_SLOW)
    pk.vacuum_off(arm)
    time.sleep(0.4)
    j_r = pk.query(arm, "GetActualJointPosDegree()")
    pk.run_path(arm, pk.cartesian_path(arm, p_grasp, p_hover, 3, j_r), pk.SEG_TIME_SLOW)
    arm.move_to_joints_smoothly(j0, duration_sec=4.0, dt=0.1)
    print("done — back at start.")


if __name__ == "__main__":
    main()
