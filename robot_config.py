"""
Physical constants of the HITBOT S922 + D415 + suction-gripper cell.
Single source of truth — import these, don't re-type the numbers.

Frames
------
flange : robot tool flange. GetActualTCPPose() returns THIS, because the
         controller's TCP offset is zero (verified: GetTCPOffset() -> all 0).
tool   : the suction pad face, TOOL_LENGTH_M along the flange +Z axis.
camera : the D415 colour optical frame, bolted to the flange.
         X_flange_camera comes from handeye_calibrate.py -> handeye.json.

Pick chain
----------
    T_base_object = T_base_flange @ X_flange_camera @ T_camera_object
and to place the PADS on the carton top rather than driving the flange into it:
    T_base_flange_target = T_base_object @ inv(T_flange_tool)
"""

import numpy as np

ARM_IP = "192.168.58.2"
ARM_PORT = 8080

# --- suction gripper -------------------------------------------------------
# 4 pads at the corners of a rectangle, gripper fixed at flange centre.
TOOL_LENGTH_M = 0.070          # flange face -> pad face, along flange +Z
PAD_SPAN_M = (0.065, 0.060)    # rectangle spanned by the 4 pad CENTRES

# A pad needs material under it plus a little margin, so the smallest pickable
# carton top is bigger than the pad span itself. Raise PAD_MARGIN_M if boxes are
# being dropped at the edges.
PAD_MARGIN_M = 0.010

# --- camera mount ----------------------------------------------------------
CAM_OFFSET_FROM_CENTRE_MM = 90.0   # measured; cross-checks the hand-eye solve

HANDEYE_FILE = "handeye.json"


def T_flange_tool():
    """4x4 flange -> suction pad face."""
    T = np.eye(4)
    T[2, 3] = TOOL_LENGTH_M
    return T


def min_graspable_top_m():
    """Smallest carton top (long, short) all 4 pads can sit on."""
    return (max(PAD_SPAN_M) + 2 * PAD_MARGIN_M,
            min(PAD_SPAN_M) + 2 * PAD_MARGIN_M)


def graspable(size_m):
    """size_m = (long, short, height) of a carton. -> (ok, reason)."""
    need_long, need_short = min_graspable_top_m()
    long_e, short_e = float(size_m[0]), float(size_m[1])
    if long_e < need_long or short_e < need_short:
        return False, (f"top {long_e*1000:.0f}x{short_e*1000:.0f}mm < "
                       f"{need_long*1000:.0f}x{need_short*1000:.0f}mm needed "
                       f"for 4 pads")
    return True, "4 pads fit"


def load_handeye(path=HANDEYE_FILE):
    """-> (X_flange_camera 4x4, meta) or (None, None) if not calibrated yet."""
    import json
    import os
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return np.array(d["X_flange_camera"], dtype=np.float64), d


def arm_pose_to_T(pose_mm_deg, conv="ZYX"):
    """HITBOT [x,y,z(mm), rx,ry,rz(deg)] -> 4x4 base->flange in METRES.
    Convention 'ZYX' was determined empirically by handeye_calibrate.py
    (lowest AX=XB residual by a factor of ~8)."""
    from scipy.spatial.transform import Rotation
    x, y, z, rx, ry, rz = pose_mm_deg
    ang = {"x": rx, "y": ry, "z": rz}
    R = Rotation.from_euler(conv, [ang[c] for c in conv.lower()],
                            degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]
    return T


def carton_to_base(T_base_flange, X_flange_camera, grasp_cam, R_cam):
    """Carton pose from the camera frame into robot base coordinates.

    grasp_cam : top-face centre (m, camera frame)   R_cam : 3x3 carton rotation.
    Returns (grasp_base_m, R_base). To PLACE THE PADS there, command the flange to
        T_base_flange_target = T_base_grasp @ inv(T_flange_tool)
    i.e. stop TOOL_LENGTH_M short along the approach axis.
    """
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_cam
    T_cam_obj[:3, 3] = grasp_cam
    T_base_obj = T_base_flange @ X_flange_camera @ T_cam_obj
    return T_base_obj[:3, 3], T_base_obj[:3, :3]
