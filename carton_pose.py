"""
Top-down carton detection + 6D pose estimation with an Intel RealSense D415.

Assumes the camera looks DOWN at cartons resting on a flat surface
(table / pallet / conveyor). No training, no GPU — pure depth geometry.

Method
------
1.  Grab depth aligned to color; deproject to a metric point cloud.
2.  RANSAC the dominant support plane (the surface the cartons sit on).
3.  Keep only points standing ABOVE that plane -> candidate carton points.
4.  DBSCAN-cluster those points into individual cartons.
5.  Per carton, take the TOP FACE points and fit a 2-D minimum-area rectangle
    *inside the plane's own coordinate frame*. Because a carton on a plane is
    fully constrained (top face parallel to the support surface), this yields an
    exact yaw and true width/length — far more stable than an unconstrained 3-D
    OBB, whose axes flip between frames.
6.  Emit: 2-D bounding box (pixels), and 6-D pose (x,y,z + roll,pitch,yaw) in the
    camera frame, plus a grasp point at the centre of the top face.

Pose convention (camera frame: +X right, +Y down, +Z forward/away)
    origin  : centre of the carton's TOP face  (the suction/grasp target)
    z-axis  : plane normal, pointing UP out of the carton top (toward camera)
    x-axis  : along the carton's LONGER top edge
    y-axis  : completes a right-handed frame

Controls:  q = quit   |   v = toggle 3-D view   |   s = save snapshot
"""

import json
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

import robot_config as rc

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
W, H, FPS = 640, 480, 30

MIN_DIST_M = 0.20          # ignore depth closer than this
MAX_DIST_M = 2.00          # ignore depth farther than this (set to just past the table)

STRIDE = 2                 # pixel subsampling for speed (2 = every other pixel)
VOXEL_SIZE = 0.005         # cloud downsample leaf (m)

PLANE_DIST_THRESH = 0.008  # RANSAC band for the support plane (m)
PLANE_MIN_POINTS = 150     # absolute inliers for a plane to count as a candidate.
                           # NOT a fraction: when cartons cover most of the view the
                           # table is a small share of the cloud but still valid.
PLANE_SEARCH_ROUNDS = 5    # how many planes to peel while hunting the support surface
MIN_POINTS_ABOVE = 150     # if fewer points than this stand above the best plane,
                           # it may be carton TOPS rather than the surface itself
MAX_PLANE_TILT_DEG = 35.0  # support plane must face a top-down camera within this

MIN_BOX_HEIGHT = 0.020     # a carton must stand >= 2 cm above the surface
MAX_BOX_HEIGHT = 0.500     # ignore anything taller than 50 cm
TOP_SLAB_M = 0.020         # thickness of the "top face" slab (m)

DBSCAN_EPS = 0.020         # cluster radius (m). raise if one carton splits
DBSCAN_MIN_PTS = 15        # min neighbours to seed a cluster
MIN_CLUSTER_PTS = 80       # reject tiny noise clusters

MIN_EDGE_M = 0.05          # carton top: longest edge limits (m)
MAX_EDGE_M = 0.60
MIN_SHORT_EDGE_M = 0.05    # shortest edge too — a carton is not a 3 cm strip.
                           # This is what rejects chair legs, cables, tool handles.
MAX_ASPECT = 4.0           # reject long thin slivers
MIN_TOP_FILL = 0.85        # top face must actually FILL its rectangle (occupancy).
                           # A real carton scores ~0.95-1.0; an L/C shape ~0.75.
                           # Lower this if cartons with dark tape (which loses
                           # depth) start getting missed.
MIN_RECTANGULARITY = 0.85  # convex-hull area / min-rect area.
                           # square/rect ~1.00, circle 0.785, irregular lower ->
                           # this is the gate that rejects round & odd-shaped things.
MAX_TOP_RMS = 0.006        # top face flatness, RMS about its plane (m)
OCC_CELL = 0.010           # occupancy grid cell for the fill test (m)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def start_camera():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # High-accuracy preset gives cleaner planes on cardboard, if supported.
    try:
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset,
                                    float(rs.rs400_visual_preset.high_accuracy))
    except Exception:
        pass

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # NOTE: no decimation_filter — it halves depth resolution and desyncs it from
    # the colour image / intrinsics. Subsampling is done via STRIDE instead.
    filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)]
    return pipeline, align, intr, depth_scale, filters


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def deproject(depth_m, fx, fy, cx, cy, stride):
    """Metric depth image -> (N,3) points. Uses true pixel coords, so the color
    intrinsics stay valid regardless of stride."""
    vs, us = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = depth_m[vs, us]
    m = (z > MIN_DIST_M) & (z < MAX_DIST_M)
    z = z[m]
    u = us[m].astype(np.float64)
    v = vs[m].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def fit_support_plane(pcd):
    """Find the surface the cartons rest on.

    Taking the single largest plane is NOT enough: a wall or the background can
    dominate, and if cartons fill the view their TOPS become the biggest plane
    (which would make every carton measure zero height and disappear).

    The support surface is the plane that is (a) large, (b) roughly perpendicular
    to a top-down camera, and (c) the FARTHEST such plane — cartons sit in front
    of it. So peel off planes one by one and pick the best candidate.

    Returns (normal, d, frac) with the normal flipped to point TOWARD the camera,
    so height = n·p + d is positive for points standing above the surface.
    """
    N = np.asarray(pcd.points)
    if len(N) < 100:
        return None, None, 0.0

    work = pcd
    total = len(N)
    cand = []                        # (n_inliers, n, d)
    for _ in range(PLANE_SEARCH_ROUNDS):
        if len(work.points) < PLANE_MIN_POINTS:
            break
        model, inl = work.segment_plane(PLANE_DIST_THRESH, 3, 300)
        if len(inl) < PLANE_MIN_POINTS:
            break
        n = np.array(model[:3], dtype=np.float64)
        d = float(model[3])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            break
        n, d = n / ln, d / ln
        if n[2] > 0:                 # camera looks +Z; "toward camera" is n_z < 0
            n, d = -n, -d

        tilt = np.degrees(np.arccos(min(abs(n[2]), 1.0)))
        if tilt <= MAX_PLANE_TILT_DEG:
            # With the normals near-parallel, d is simply how far the plane sits.
            cand.append((len(inl), n, d))

        work = work.select_by_index(inl, invert=True)   # peel and look deeper

    if not cand:
        return None, None, 0.0

    def n_above(c):
        """How many points stand at carton height above this plane."""
        h = N @ c[1] + c[2]
        return int(((h > MIN_BOX_HEIGHT) & (h < MAX_BOX_HEIGHT)).sum())

    # Among sizeable planes that actually have carton-height points standing on
    # them, the support surface is the NEAREST one to the camera. Raw size is the
    # wrong criterion: from a high viewpoint the floor out-votes the table, and
    # then the table+cartons sit far outside the height band and vanish. Carton
    # TOPS are nearer still, but nothing stands above a top, so n_above skips them.
    strong = [c for c in cand
              if c[0] >= max(PLANE_MIN_POINTS, 0.05 * total)
              and n_above(c) >= MIN_POINTS_ABOVE]
    if strong:
        base = min(strong, key=lambda c: c[2])          # smallest d = nearest
    else:
        base = max(cand, key=lambda c: c[0])

    # Only if nothing stands above it do we look deeper. That happens when cartons
    # blanket the view and their TOPS became the dominant plane — then the real
    # surface is farther, by at least a carton's height.
    #
    # This must NOT be a general preference for the farthest plane: if the camera
    # sees a table with floor visible around its edges, the floor is farther and
    # would hijack the reference, pushing the table+cartons out of range entirely.
    if n_above(base) < MIN_POINTS_ABOVE:
        deeper = [c for c in cand
                  if c[2] > base[2] + MIN_BOX_HEIGHT
                  and c[0] >= max(PLANE_MIN_POINTS, 0.05 * total)]
        if deeper:
            alt = max(deeper, key=n_above)
            if n_above(alt) > n_above(base):
                base = alt

    return base[1], base[2], base[0] / total


def plane_basis(n):
    """Two orthonormal in-plane axes. Derived from the camera X axis so the basis
    is stable frame-to-frame (no arbitrary spinning)."""
    ex = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ex, n)) > 0.9:            # degenerate; pick another seed
        ex = np.array([0.0, 1.0, 0.0])
    u1 = ex - np.dot(ex, n) * n
    u1 /= (np.linalg.norm(u1) + 1e-12)
    u2 = np.cross(n, u1)
    u2 /= (np.linalg.norm(u2) + 1e-12)
    return u1, u2


def canonical_yaw(deg):
    """Rectangle orientation is only defined mod 180°. Fold to [-90, 90) so the
    reported yaw — and the gripper wrist — doesn't jump between frames."""
    y = (deg + 90.0) % 180.0 - 90.0
    return y


def rot_to_rpy_deg(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def project(pts, fx, fy, cx, cy):
    z = np.clip(pts[:, 2], 1e-6, None)
    u = pts[:, 0] * fx / z + cx
    v = pts[:, 1] * fy / z + cy
    return np.stack((u, v), axis=-1)


def occupancy_fill(quad, centre, long_e, short_e, ang_long_deg):
    """Fraction of the fitted rectangle actually covered by top-face points.

    Rotate the points into the rectangle's own frame, bin them into a grid, and
    count occupied cells. Robust to point density (a dense blob in one corner
    can't fake a full rectangle) and catches hollow/ring/L shapes.
    """
    th = np.radians(ang_long_deg)
    e1 = np.array([np.cos(th), np.sin(th)])
    e2 = np.array([-np.sin(th), np.cos(th)])
    rel = quad.astype(np.float64) - np.asarray(centre, dtype=np.float64)
    p = rel @ e1
    q = rel @ e2

    nx = max(int(np.ceil(long_e / OCC_CELL)), 1)
    ny = max(int(np.ceil(short_e / OCC_CELL)), 1)
    ip = np.clip(((p + long_e / 2.0) / OCC_CELL).astype(int), 0, nx - 1)
    iq = np.clip(((q + short_e / 2.0) / OCC_CELL).astype(int), 0, ny - 1)
    occupied = len(np.unique(ip.astype(np.int64) * ny + iq.astype(np.int64)))
    return occupied / float(nx * ny)


# ---------------------------------------------------------------------------
# Core: cluster -> carton pose
# ---------------------------------------------------------------------------
def carton_from_cluster(pts, heights, n, d, u1, u2):
    """Fit a carton to one cluster. Returns a dict, or None if it isn't box-like."""
    h_top = float(np.percentile(heights, 97))
    h_base = float(np.percentile(heights, 3))
    if not (MIN_BOX_HEIGHT <= h_top <= MAX_BOX_HEIGHT):
        return None

    # --- isolate the top face ---
    top_m = heights >= (h_top - TOP_SLAB_M)
    top = pts[top_m]
    if len(top) < 30:
        return None

    # flatness of the top face
    rms = float(np.sqrt(np.mean((heights[top_m] - heights[top_m].mean()) ** 2)))
    if rms > MAX_TOP_RMS:
        return None

    # --- fit a 2-D min-area rectangle in the plane frame ---
    a = top @ u1
    b = top @ u2
    quad = np.stack((a, b), axis=-1).astype(np.float32)
    (rc_a, rc_b), (w, l), ang = cv2.minAreaRect(quad)
    if w < 1e-6 or l < 1e-6:
        return None

    # order so x = longer edge
    if w >= l:
        long_e, short_e, ang_long = w, l, ang
    else:
        long_e, short_e, ang_long = l, w, ang + 90.0

    # --- size gates ---
    if not (MIN_EDGE_M <= long_e <= MAX_EDGE_M):
        return None
    if short_e < MIN_SHORT_EDGE_M or long_e / max(short_e, 1e-6) > MAX_ASPECT:
        return None

    # --- shape gate 1: rectangularity ---
    # A carton top is a filled rectangle, so its convex hull fills its bounding
    # rect (~1.0). A cylinder/bottle top gives pi/4 = 0.785; irregular clutter less.
    rect_area = long_e * short_e
    hull = cv2.convexHull(quad)
    rectangularity = float(cv2.contourArea(hull)) / max(rect_area, 1e-9)
    if rectangularity < MIN_RECTANGULARITY:
        return None

    # --- shape gate 2: occupancy fill ---
    # Rasterise the top face into cells inside the fitted rect. Counting POINTS
    # would just measure sensor density; counting occupied CELLS measures whether
    # the surface is really there (and catches C/L/ring shapes the hull misses).
    fill = occupancy_fill(quad, (rc_a, rc_b), long_e, short_e, ang_long)
    if fill < MIN_TOP_FILL:
        return None

    # --- build the 6-D pose ---
    th = np.radians(ang_long)
    x_axis = np.cos(th) * u1 + np.sin(th) * u2
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    z_axis = n                                   # up, out of the carton top
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)            # re-orthonormalise
    R = np.stack((x_axis, y_axis, z_axis), axis=1)

    # Reconstruct the 3-D top-face centre. (u1, u2, n) is an orthonormal basis, so
    # any point p = (p·u1)u1 + (p·u2)u2 + (p·n)n, and p·n = height - d.
    top_centre = rc_a * u1 + rc_b * u2 + (h_top - d) * n

    # The carton rests ON the support plane, so its height IS its top-face height
    # above that plane. Top-down, the side walls are barely visible, so measuring
    # from the plane is far more robust than inferring a base from sparse points.
    height = h_top

    # Report yaw in the CAMERA frame (the angle the gripper wrist needs), not in
    # the plane's mirrored basis.
    yaw_cam = canonical_yaw(np.degrees(np.arctan2(x_axis[1], x_axis[0])))

    return {
        "top_centre": top_centre,
        "R": R,
        "size": np.array([long_e, short_e, height]),
        "h_top": h_top,
        "h_base": h_base,
        "yaw": yaw_cam,
        "rms": rms,
        "fill": fill,
        "npts": int(len(pts)),
    }


def corners_3d(c, n):
    """8 corners: the fitted top rect, and the same rect dropped onto the support
    plane (the carton sits on it)."""
    R, size = c["R"], c["size"]
    hx, hy = size[0] / 2.0, size[1] / 2.0
    x_axis, y_axis = R[:, 0], R[:, 1]
    top_c = c["top_centre"]
    top = [top_c + sx * hx * x_axis + sy * hy * y_axis
           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    drop = n * c["h_top"]          # from the top face down to the plane
    bot = [p - drop for p in top]
    return np.array(top + bot)


# ---------------------------------------------------------------------------
def main():
    pipeline, align, intr, depth_scale, filters = start_camera()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print("=== TOP-DOWN CARTON 6D POSE ===   q quit | v 3D | s snapshot")
    print(f"intr fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} scale={depth_scale}")

    show3d, vis = False, None
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df:
                continue
            for f in filters:
                df = f.process(df)

            color = np.asanyarray(cf.get_data())
            disp = color.copy()
            depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            pts = deproject(depth_m, fx, fy, cx, cy, STRIDE)
            cartons, note = [], ""

            if len(pts) > 500:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                P = np.asarray(pcd.points)

                n, d, frac = fit_support_plane(pcd)
                if n is None:
                    note = "no support plane — is the camera top-down over the surface?"
                else:
                    u1, u2 = plane_basis(n)
                    heights = P @ n + d          # >0 above the surface

                    above = (heights > MIN_BOX_HEIGHT) & (heights < MAX_BOX_HEIGHT)
                    Pa, Ha = P[above], heights[above]
                    if len(Pa) < DBSCAN_MIN_PTS:
                        note = "nothing above the surface"
                    else:
                        cl = o3d.geometry.PointCloud()
                        cl.points = o3d.utility.Vector3dVector(Pa)
                        lab = np.array(cl.cluster_dbscan(DBSCAN_EPS, DBSCAN_MIN_PTS,
                                                         print_progress=False))
                        for k in range(lab.max() + 1):
                            m = lab == k
                            if m.sum() < MIN_CLUSTER_PTS:
                                continue
                            c = carton_from_cluster(Pa[m], Ha[m], n, d, u1, u2)
                            if c:
                                c["corners"] = corners_3d(c, n)
                                cartons.append(c)

                    # nearest first (top-down => smallest z is topmost/closest)
                    cartons.sort(key=lambda c: c["top_centre"][2])

            # ---------------- draw ----------------
            for i, c in enumerate(cartons):
                cor = project(c["corners"], fx, fy, cx, cy).astype(int)
                # 2-D axis-aligned bounding box (what you asked for)
                x0, y0 = cor[:, 0].min(), cor[:, 1].min()
                x1, y1 = cor[:, 0].max(), cor[:, 1].max()
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, W - 1), min(y1, H - 1)
                cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 180, 0), 1)

                # oriented top face + vertical edges
                cv2.polylines(disp, [cor[:4]], True, (0, 255, 0), 2)
                cv2.polylines(disp, [cor[4:]], True, (0, 140, 0), 1)
                for j in range(4):
                    cv2.line(disp, tuple(cor[j]), tuple(cor[j + 4]), (0, 200, 0), 1)

                # pose axes at the grasp point
                g = c["top_centre"]
                Rm = c["R"]
                ax = project(np.stack([g, g + Rm[:, 0] * 0.05,
                                       g + Rm[:, 1] * 0.05,
                                       g + Rm[:, 2] * 0.05]), fx, fy, cx, cy).astype(int)
                cv2.line(disp, tuple(ax[0]), tuple(ax[1]), (0, 0, 255), 2)
                cv2.line(disp, tuple(ax[0]), tuple(ax[2]), (0, 255, 0), 2)
                cv2.line(disp, tuple(ax[0]), tuple(ax[3]), (255, 0, 0), 2)
                cv2.circle(disp, tuple(ax[0]), 5, (0, 255, 255), -1)

                s = c["size"]
                ok, why = rc.graspable(s)
                col = (0, 255, 255) if ok else (0, 165, 255)
                cv2.putText(disp, f"#{i} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm"
                                  f"{'' if ok else '  NO-GRASP'}",
                            (x0, max(y0 - 20, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            col, 1, cv2.LINE_AA)
                cv2.putText(disp, f"yaw {c['yaw']:+.1f}deg  z {g[2]:.3f}m",
                            (x0, max(y0 - 6, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            col, 1, cv2.LINE_AA)

                rpy = rot_to_rpy_deg(Rm)
                print(f"carton#{i} grasp=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f})m "
                      f"size=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})m "
                      f"rpy=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f})deg "
                      f"yaw={c['yaw']:+.1f} bbox=({x0},{y0},{x1},{y1})")

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            cv2.putText(disp, f"{len(cartons)} cartons   {fps:4.1f} FPS   {note}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Carton 6D Pose (top-down)", disp)

            if show3d:
                if vis is None:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window("cloud", width=720, height=540)
                vis.clear_geometries()
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts[::2])
                vis.add_geometry(pc)
                vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
                vis.poll_events(); vis.update_renderer()

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('v'):
                show3d = not show3d
                if not show3d and vis:
                    vis.destroy_window(); vis = None
            if k == ord('s'):
                cv2.imwrite("carton_snapshot.png", disp)
                with open("carton_poses.json", "w") as fh:
                    json.dump([{ "grasp_xyz_m": c["top_centre"].tolist(),
                                 "size_m": c["size"].tolist(),
                                 "rpy_deg": rot_to_rpy_deg(c["R"]).tolist(),
                                 "R": c["R"].tolist()} for c in cartons], fh, indent=2)
                print("saved carton_snapshot.png + carton_poses.json")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if vis:
            vis.destroy_window()


if __name__ == "__main__":
    main()
