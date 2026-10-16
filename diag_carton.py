"""Headless diagnostic for carton_pose.py — one frame, gate-by-gate report.
Aim the camera top-down at the cartons and run this to tune thresholds."""
import importlib.util
import cv2
import numpy as np
import open3d as o3d

spec = importlib.util.spec_from_file_location("cp", "carton_pose.py")
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

pipeline, align, intr, depth_scale, filters = cp.start_camera()
for _ in range(30):
    pipeline.wait_for_frames()
frames = align.process(pipeline.wait_for_frames())
df = frames.get_depth_frame()
for f in filters:
    df = f.process(df)
color = np.asanyarray(frames.get_color_frame().get_data())
depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
pipeline.stop()

pts = cp.deproject(depth_m, intr.fx, intr.fy, intr.ppx, intr.ppy, cp.STRIDE)
pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
pcd = pcd.voxel_down_sample(cp.VOXEL_SIZE)
P = np.asarray(pcd.points)
print(f"cloud {len(pts)} -> voxel {len(P)}")

n, d, frac = cp.fit_support_plane(pcd)
if n is None:
    print("NO SUPPORT PLANE — camera must look down at a flat surface"); raise SystemExit
tilt = np.degrees(np.arccos(min(abs(n[2]), 1.0)))
print(f"support plane: {frac*100:.0f}% of cloud, tilt={tilt:.1f}deg, distance={d:.3f} m")
if tilt > cp.MAX_PLANE_TILT_DEG:
    print("  !! too tilted — aim the camera straight down")

u1, u2 = cp.plane_basis(n)
heights = P @ n + d
above = (heights > cp.MIN_BOX_HEIGHT) & (heights < cp.MAX_BOX_HEIGHT)
print(f"points standing {cp.MIN_BOX_HEIGHT*100:.0f}-{cp.MAX_BOX_HEIGHT*100:.0f} cm "
      f"above surface: {above.sum()}")
if above.sum() < cp.DBSCAN_MIN_PTS:
    print("  !! nothing above the surface — is anything in view?"); raise SystemExit

Pa, Ha = P[above], heights[above]
cl = o3d.geometry.PointCloud(); cl.points = o3d.utility.Vector3dVector(Pa)
lab = np.array(cl.cluster_dbscan(cp.DBSCAN_EPS, cp.DBSCAN_MIN_PTS))
print(f"clusters: {lab.max()+1}\n")

disp = color.copy()
npass = 0
for k in range(lab.max() + 1):
    m = lab == k
    npts = int(m.sum())
    if npts < cp.MIN_CLUSTER_PTS:
        print(f"cl{k}: {npts:5d} pts  reject: too few (<{cp.MIN_CLUSTER_PTS})")
        continue
    pk, hk = Pa[m], Ha[m]
    h_top = float(np.percentile(hk, 97))
    tm = hk >= (h_top - cp.TOP_SLAB_M)
    if tm.sum() < 30:
        print(f"cl{k}: {npts:5d} pts  reject: top face too small ({tm.sum()} pts)")
        continue
    top = pk[tm]
    rms = float(np.sqrt(np.mean((hk[tm] - hk[tm].mean()) ** 2)))
    quad = np.stack((top @ u1, top @ u2), -1).astype(np.float32)
    (ca, cb), (w, l), ang = cv2.minAreaRect(quad)
    le, se, al = (w, l, ang) if w >= l else (l, w, ang + 90)
    rect = float(cv2.contourArea(cv2.convexHull(quad))) / max(le * se, 1e-9)
    fill = cp.occupancy_fill(quad, (ca, cb), le, se, al)
    asp = le / max(se, 1e-6)

    why = []
    if not (cp.MIN_EDGE_M <= le <= cp.MAX_EDGE_M): why.append(f"long edge {le*100:.1f}cm")
    if se < cp.MIN_SHORT_EDGE_M: why.append(f"short edge {se*100:.1f}cm")
    if asp > cp.MAX_ASPECT: why.append(f"aspect {asp:.1f}")
    if rms > cp.MAX_TOP_RMS: why.append(f"not flat {rms*1000:.1f}mm")
    if rect < cp.MIN_RECTANGULARITY: why.append(f"not rectangular {rect:.2f}")
    if fill < cp.MIN_TOP_FILL: why.append(f"not filled {fill:.2f}")

    stats = (f"{le*100:4.1f}x{se*100:4.1f}x{h_top*100:4.1f}cm rect={rect:.2f} "
             f"fill={fill:.2f} rms={rms*1000:.1f}mm")
    c = cp.carton_from_cluster(pk, hk, n, d, u1, u2)
    if c is not None:
        npass += 1
        g = c["top_centre"]
        print(f"cl{k}: {npts:5d} pts  CARTON ✓  {stats}")
        print(f"        grasp=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f})m yaw={c['yaw']:+.1f}deg")
        cor = cp.project(cp.corners_3d(c, n), intr.fx, intr.fy, intr.ppx, intr.ppy).astype(int)
        x0, y0 = cor[:, 0].min(), cor[:, 1].min(); x1, y1 = cor[:, 0].max(), cor[:, 1].max()
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 180, 0), 2)
        cv2.polylines(disp, [cor[:4]], True, (0, 255, 0), 2)
    else:
        print(f"cl{k}: {npts:5d} pts  reject: {', '.join(why) or 'gate'}  {stats}")

print(f"\n==> {npass} carton(s) detected")
cv2.imwrite("diag_carton_color.png", disp)
print("saved diag_carton_color.png (with overlays)")
