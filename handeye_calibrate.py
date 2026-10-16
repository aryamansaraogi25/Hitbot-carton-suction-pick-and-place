"""
Eye-in-hand calibration for the HITBOT S922 (6-DOF) + RealSense D415.

Solves  X = flange -> camera,  so carton poses from carton_pose.py can be moved
into robot base coordinates:

    T_base_object = T_base_flange  @  X  @  T_camera_object

Usage
-----
  # 1. Jog the arm so the camera can SEE the ChArUco board, then:
  python3 handeye_calibrate.py --square-mm 25 --marker-mm 18 --squares 5 7

  #    (dry run: no motion. Checks the board is detected and prints the plan.)

  # 2. When the plan looks safe, keep the e-stop in hand and add --go:
  python3 handeye_calibrate.py --square-mm 25 --marker-mm 18 --squares 5 7 --go

Notes
-----
* The arm must be in AUTO/REMOTE mode on the pendant (see HITBOT_SDK_NOTES.md).
* TCP offset was measured as zero, so GetActualTCPPose() == flange pose.
* The board stays FIXED. The camera moves with the arm.
* Rotation diversity is what makes the maths solvable — the poses deliberately
  tilt the wrist about two axes, not just spin it about one.
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, "/home/xp/z-arms922/hitbot_ws")
import handeye_solver as hes

ARM_IP = "192.168.58.2"

# Candidate Euler conventions for the arm's (rx, ry, rz). The wire format isn't
# documented, so we try each and keep whichever gives the lowest residual.
EULER_CANDIDATES = ["xyz", "XYZ", "zyx", "ZYX"]

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100, "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


# ---------------------------------------------------------------------------
# rotation conventions
# ---------------------------------------------------------------------------
def euler_to_R(rx, ry, rz, conv):
    """Degrees -> rotation matrix, for a named convention.
    Lower-case = extrinsic (fixed axes, i.e. classic RPY), upper = intrinsic."""
    from scipy.spatial.transform import Rotation
    order = conv.lower()
    ang = {"x": rx, "y": ry, "z": rz}
    seq = [ang[c] for c in order]
    return Rotation.from_euler(conv, seq, degrees=True).as_matrix()


def pose_to_T(pose_mm_deg, conv):
    """HITBOT [x,y,z (mm), rx,ry,rz (deg)] -> 4x4 in METRES."""
    x, y, z, rx, ry, rz = pose_mm_deg
    return hes.rt_to_T(euler_to_R(rx, ry, rz, conv), [x / 1000.0, y / 1000.0, z / 1000.0])


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------
class Arm:
    def __init__(self):
        from hitbot_tcp_api import HitbotTCPInterface
        self.r = HitbotTCPInterface(ip=ARM_IP)
        if not self.r.connect():
            raise RuntimeError("arm connect failed — is it in AUTO/REMOTE mode?")

    def _query(self, cmd):
        resp = self.r._send_cmd(377, cmd)
        parts = resp.split("III")
        if len(parts) < 6:
            raise RuntimeError(f"bad response to {cmd}: {resp!r}")
        if "errcode" in parts[4]:
            raise RuntimeError(f"{cmd} -> {parts[4]} (arm not in AUTO/REMOTE?)")
        return [float(v) for v in parts[4].split(",")]

    def tcp_pose(self):
        return self._query("GetActualTCPPose()")     # mm, deg

    def joints(self):
        return self._query("GetActualJointPosDegree()")

    def move_joints(self, target, duration=3.0):
        self.r.move_to_joints_smoothly(list(target), duration_sec=duration, dt=0.1)


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
class Cam:
    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        prof = self.pipe.start(cfg)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        self.dist = np.array(i.coeffs, dtype=np.float64)
        for _ in range(20):
            self.pipe.wait_for_frames()

    def grab(self):
        f = self.pipe.wait_for_frames().get_color_frame()
        return np.asanyarray(f.get_data())

    def stop(self):
        self.pipe.stop()


# ---------------------------------------------------------------------------
# board
# ---------------------------------------------------------------------------
def make_board(squares, square_m, marker_m, dict_name):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    board = cv2.aruco.CharucoBoard((squares[0], squares[1]), square_m, marker_m, d)
    return board, cv2.aruco.CharucoDetector(board)


def detect_board(det, board, img, K, dist, min_corners=8):
    """-> (T_cam_target, n_corners) or (None, n)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cc, ci, _, _ = det.detectBoard(gray)
    if cc is None or ci is None or len(cc) < min_corners:
        return None, (0 if cc is None else len(cc))
    obj = board.getChessboardCorners()[ci.flatten()]
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float32), cc.astype(np.float32),
                                  K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, len(cc)
    R, _ = cv2.Rodrigues(rvec)
    return hes.rt_to_T(R, tvec.ravel()), len(cc)


def autodetect_dict(img, squares, square_m, marker_m):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best, best_n = None, 0
    for name, did in ARUCO_DICTS.items():
        d = cv2.aruco.getPredefinedDictionary(did)
        b = cv2.aruco.CharucoBoard((squares[0], squares[1]), square_m, marker_m, d)
        cc, ci, mc, mi = cv2.aruco.CharucoDetector(b).detectBoard(gray)
        n = 0 if mi is None else len(mi)
        if n > best_n:
            best, best_n = name, n
    return best, best_n


# ---------------------------------------------------------------------------
# pose plan
# ---------------------------------------------------------------------------
def plan_poses(j0, n=16):
    """Joint targets around the current pose.

    Wrist joints (4,5,6) get the big swings — that is what supplies rotation about
    two non-parallel axes, which the AX=XB solve REQUIRES. Base joints move only a
    little, to shift viewpoint without walking the camera off the board.
    """
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        d = np.zeros(6)
        d[3] = rng.uniform(-12, 12)      # wrist tilt, axis A
        d[4] = rng.uniform(-12, 12)      # wrist tilt, axis B  (non-parallel)
        d[5] = rng.uniform(-20, 20)      # wrist roll
        d[0] = rng.uniform(-4, 4)
        d[1] = rng.uniform(-3, 3)
        d[2] = rng.uniform(-3, 3)
        out.append(np.asarray(j0) + d)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", type=int, nargs=2, default=[5, 7],
                    help="board squares: X Y")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--marker-mm", type=float, required=True)
    ap.add_argument("--dict", default=None, help="aruco dict (default: autodetect)")
    ap.add_argument("--poses", type=int, default=16)
    ap.add_argument("--expect-cam-offset-mm", type=float, default=None,
                    help="known camera offset from flange centre (mm). Used as an "
                         "INDEPENDENT physical check on the solved transform — a low "
                         "residual alone does not prove the answer is right.")
    ap.add_argument("--go", action="store_true", help="ACTUALLY MOVE THE ARM")
    ap.add_argument("--out", default="handeye.json")
    a = ap.parse_args()

    sq, mk = a.square_mm / 1000.0, a.marker_mm / 1000.0
    cam = Cam()
    img = cam.grab()

    dict_name = a.dict
    if dict_name is None:
        dict_name, nm = autodetect_dict(img, a.squares, sq, mk)
        if dict_name is None:
            print("NO ARUCO MARKERS FOUND. Aim the camera at the board, check "
                  "lighting/focus, and confirm --squares matches your printout.")
            cam.stop(); return
        print(f"autodetected dictionary: {dict_name} ({nm} markers)")

    board, det = make_board(a.squares, sq, mk, dict_name)
    T, n = detect_board(det, board, img, cam.K, cam.dist)
    if T is None:
        print(f"board NOT detected ({n} corners). Move the camera so the whole "
              f"board is visible and in focus.")
        cam.stop(); return
    print(f"board detected: {n} corners, at {np.linalg.norm(T[:3,3]):.3f} m")

    arm = Arm()
    j0 = arm.joints()
    p0 = arm.tcp_pose()
    print(f"current joints: {[round(v,2) for v in j0]}")
    print(f"current TCP   : {[round(v,2) for v in p0]}")

    poses = plan_poses(j0, a.poses)
    if not a.go:
        print(f"\n--- DRY RUN — nothing will move. Planned {len(poses)} poses "
              f"(deltas from current joints):")
        for i, p in enumerate(poses[:6]):
            d = " ".join(f"{float(v-j0[k]):+6.1f}" for k, v in enumerate(p))
            print(f"  {i:2d}: [{d} ] deg")
        print(f"  ... {len(poses)-6} more (max wrist swing +/-12 deg, roll +/-20)")
        print("\nBoard is visible and the arm reads fine. Re-run with --go "
              "(e-stop in hand) to capture.")
        cam.stop(); return

    # ---- capture ----
    BF_raw, CT = [], []
    print(f"\ncapturing {len(poses)} poses ...")
    try:
        for i, jt in enumerate(poses):
            arm.move_joints(jt, duration=2.5)
            time.sleep(0.8)                       # settle: never capture mid-motion
            img = cam.grab()
            T_ct, ncor = detect_board(det, board, img, cam.K, cam.dist)
            pose = arm.tcp_pose()                 # read AFTER settling
            if T_ct is None:
                print(f"  {i:2d}: board not seen ({ncor} corners) — skipped")
                continue
            BF_raw.append(pose)
            CT.append(T_ct)
            print(f"  {i:2d}: ok ({ncor} corners)")
    finally:
        print("returning to start pose ...")
        try:
            arm.move_joints(j0, duration=3.0)
        except Exception as e:
            print(f"  (return failed: {e})")
        cam.stop()

    if len(BF_raw) < 5:
        print(f"only {len(BF_raw)} good samples — need >=5. Aborting.")
        return

    # ---- solve, trying each Euler convention ----
    print(f"\nsolving with {len(BF_raw)} samples; testing Euler conventions:")
    best = None
    for conv in EULER_CANDIDATES:
        BF = [pose_to_T(p, conv) for p in BF_raw]
        try:
            X, info, method = hes.solve(BF, CT, verbose=False)
            re, te = hes.residuals(X, *hes.build_motions(BF, CT))
            score = float(np.median(te))
            print(f"  {conv:4s}: rot {np.median(re):6.3f} deg  trans "
                  f"{score*1000:7.2f} mm   [{method}]")
            if best is None or score < best[0]:
                best = (score, conv, X, method, float(np.median(re)))
        except Exception as e:
            print(f"  {conv:4s}: {str(e)[:60]}")

    if best is None:
        print("all conventions failed — check rotation diversity."); return

    score, conv, X, method, rot_err = best
    print(f"\n==> convention '{conv}' fits best ({method})")
    print(f"    residual: {rot_err:.3f} deg, {score*1000:.2f} mm")
    if score > 0.010:
        print("    !! >10 mm residual — suspect. Check board size (--square-mm),"
              " lighting, and that the camera is rigidly bolted to the flange.")
    print(f"\nX = flange -> camera:\n{np.array2string(X, precision=5)}")
    off = X[:3, 3] * 1000.0
    radial = float(np.hypot(off[0], off[1]))
    print(f"    camera offset from flange: {off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f} mm")
    print(f"    radial (XY) offset: {radial:.1f} mm   |  total: "
          f"{np.linalg.norm(off):.1f} mm")

    # Independent physical cross-check. A small AX=XB residual only proves the
    # samples are self-consistent — it does NOT catch a wrong board scale, which
    # biases every translation. Comparing against a tape-measured mounting offset
    # is the check that does.
    if a.expect_cam_offset_mm is not None:
        err = abs(radial - a.expect_cam_offset_mm)
        print(f"    expected ~{a.expect_cam_offset_mm:.0f} mm -> off by {err:.1f} mm")
        if err > 15.0:
            print("    !! MISMATCH vs the physical mount. Most likely causes, in "
                  "order: wrong --square-mm (scales every translation), camera not "
                  "rigidly fixed, or the offset was measured to a different point "
                  "than the lens centre.")
        else:
            print("    OK: solved transform agrees with the physical mount.")

    json.dump({"X_flange_camera": X.tolist(), "euler_convention": conv,
               "method": method, "residual_mm": score * 1000,
               "residual_deg": rot_err, "n_samples": len(BF_raw),
               "board": {"squares": a.squares, "square_mm": a.square_mm,
                         "marker_mm": a.marker_mm, "dict": dict_name}},
              open(a.out, "w"), indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
