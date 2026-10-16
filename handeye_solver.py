"""
Hand-eye calibration solver:  A X = X B

OpenCV 5.0 removed cv2.calibrateHandEye (it existed in 4.x), so this implements
the classic solvers directly. No OpenCV dependency, just numpy.

Eye-in-hand setup (camera bolted to the robot flange, board fixed in the world):
    X = flange -> camera  (the constant we want)

Given N robot poses (base->flange) and N board poses (camera->target), we form
relative motions between pose pairs:
    A_ij = (base->flange_i)^-1 (base->flange_j)      # flange motion
    B_ij = (cam->target_i) (cam->target_j)^-1        # camera motion
and solve A X = X B.

Two independent methods are provided so they can cross-check each other:
  * tsai_lenz   — Tsai & Lenz 1989, angle-axis least squares
  * park_martin — Park & Martin 1994, matrix-log / Lie-algebra least squares

IMPORTANT: rotation about a single axis is degenerate. The relative motions must
rotate about at least two non-parallel axes or X is not fully determined — the
translation along the shared rotation axis becomes unobservable. `motion_rank`
below reports this so you can catch it instead of trusting a bogus result.
"""

import numpy as np


# ---------------------------------------------------------------------------
# small SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------
def rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def log_SO3(R):
    """Rotation matrix -> rotation vector (axis * angle), numerically guarded."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(np.pi - th) < 1e-6:
        # near pi: use the symmetric part for stability
        A = (R + np.eye(3)) / 2.0
        ax = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(ax))
        v = A[:, i] / (ax[i] + 1e-12)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v * th
    return th / (2.0 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def exp_SO3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def project_SO3(M):
    """Nearest rotation matrix (SVD), guaranteeing det=+1."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# motion pair construction
# ---------------------------------------------------------------------------
def build_motions(T_base_flange, T_cam_target, min_angle_deg=5.0):
    """Relative motion pairs (A, B) from absolute poses.

    Uses ALL i<j pairs, not just consecutive ones — far more constraints from the
    same data. Pairs with too little rotation are dropped: they carry almost no
    information about X and just amplify noise.
    """
    A_list, B_list = [], []
    n = len(T_base_flange)
    for i in range(n):
        for j in range(i + 1, n):
            A = T_inv(T_base_flange[i]) @ T_base_flange[j]
            B = T_cam_target[i] @ T_inv(T_cam_target[j])
            ang = np.degrees(np.linalg.norm(log_SO3(A[:3, :3])))
            if ang < min_angle_deg:
                continue
            A_list.append(A)
            B_list.append(B)
    return A_list, B_list


def motion_rank(A_list):
    """Rank of the rotation-axis set. Must be >= 2 or the problem is degenerate
    (all rotations about one axis => X's translation along it is unobservable)."""
    axes = []
    for A in A_list:
        w = log_SO3(A[:3, :3])
        nw = np.linalg.norm(w)
        if nw > 1e-6:
            axes.append(w / nw)
    if not axes:
        return 0, np.zeros(3)
    M = np.array(axes)
    s = np.linalg.svd(M, compute_uv=False)
    rank = int((s > 0.1 * s[0]).sum()) if s[0] > 1e-9 else 0
    return rank, s


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------
def tsai_lenz(A_list, B_list):
    """Tsai & Lenz 1989."""
    S, b = [], []
    for A, B in zip(A_list, B_list):
        ra, rb = log_SO3(A[:3, :3]), log_SO3(B[:3, :3])
        tha, thb = np.linalg.norm(ra), np.linalg.norm(rb)
        if tha < 1e-8 or thb < 1e-8:
            continue
        # modified Rodrigues vectors
        Pa = 2 * np.sin(tha / 2) * (ra / tha)
        Pb = 2 * np.sin(thb / 2) * (rb / thb)
        S.append(skew(Pa + Pb))
        b.append(Pb - Pa)
    if not S:
        raise ValueError("no usable motion pairs")
    S = np.vstack(S)
    b = np.hstack(b)
    x, *_ = np.linalg.lstsq(S, b, rcond=None)
    # recover the rotation from the scaled vector
    Pcg = 2 * x / np.sqrt(1 + np.dot(x, x))
    npc = np.dot(Pcg, Pcg)
    Rx = (1 - npc / 2) * np.eye(3) + 0.5 * (np.outer(Pcg, Pcg) +
                                            np.sqrt(max(4 - npc, 0.0)) * skew(Pcg))
    Rx = project_SO3(Rx)

    # translation: (Ra - I) tx = Rx tb - ta
    C, d = [], []
    for A, B in zip(A_list, B_list):
        C.append(A[:3, :3] - np.eye(3))
        d.append(Rx @ B[:3, 3] - A[:3, 3])
    tx, *_ = np.linalg.lstsq(np.vstack(C), np.hstack(d), rcond=None)
    return rt_to_T(Rx, tx)


def park_martin(A_list, B_list):
    """Park & Martin 1994 (Lie-algebra / matrix-log least squares)."""
    M = np.zeros((3, 3))
    alpha, beta = [], []
    for A, B in zip(A_list, B_list):
        a = log_SO3(A[:3, :3])
        b = log_SO3(B[:3, :3])
        if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
            continue
        alpha.append(a)
        beta.append(b)
        M += np.outer(b, a)
    if not alpha:
        raise ValueError("no usable motion pairs")
    # Rx = (M^T M)^(-1/2) M^T
    MtM = M.T @ M
    w, V = np.linalg.eigh(MtM)
    w = np.clip(w, 1e-12, None)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    Rx = project_SO3(inv_sqrt @ M.T)

    C, d = [], []
    for A, B in zip(A_list, B_list):
        C.append(np.eye(3) - A[:3, :3])
        d.append(A[:3, 3] - Rx @ B[:3, 3])
    tx, *_ = np.linalg.lstsq(np.vstack(C), np.hstack(d), rcond=None)
    return rt_to_T(Rx, tx)


def residuals(X, A_list, B_list):
    """Per-pair error of A X = X B. Returns (rot_err_deg, trans_err_m) arrays."""
    re, te = [], []
    for A, B in zip(A_list, B_list):
        L = A @ X
        R = X @ B
        re.append(np.degrees(np.linalg.norm(log_SO3(L[:3, :3].T @ R[:3, :3]))))
        te.append(np.linalg.norm(L[:3, 3] - R[:3, 3]))
    return np.array(re), np.array(te)


def solve(T_base_flange, T_cam_target, min_angle_deg=5.0, verbose=True):
    """Run both solvers, return the one with the lower translation residual."""
    A_list, B_list = build_motions(T_base_flange, T_cam_target, min_angle_deg)
    if len(A_list) < 3:
        raise ValueError(f"only {len(A_list)} usable motion pairs; need >=3 "
                         f"(move the arm to more varied ORIENTATIONS)")
    rank, sv = motion_rank(A_list)
    if verbose:
        print(f"motion pairs: {len(A_list)}   rotation-axis rank: {rank} "
              f"(need >=2)")
    if rank < 2:
        raise ValueError(
            "DEGENERATE: all rotations share one axis. X cannot be recovered. "
            "Tilt the wrist about a second axis and re-capture.")

    out = {}
    for name, fn in (("tsai_lenz", tsai_lenz), ("park_martin", park_martin)):
        try:
            X = fn(A_list, B_list)
            re, te = residuals(X, A_list, B_list)
            out[name] = (X, float(np.median(re)), float(np.median(te)))
            if verbose:
                print(f"  {name:11s} residual: rot {np.median(re):.3f} deg, "
                      f"trans {np.median(te)*1000:.2f} mm")
        except Exception as e:
            if verbose:
                print(f"  {name:11s} FAILED: {e}")
    if not out:
        raise RuntimeError("both solvers failed")
    best = min(out.items(), key=lambda kv: kv[1][2])
    if verbose:
        print(f"  -> using {best[0]}")
    return best[1][0], {k: (v[1], v[2]) for k, v in out.items()}, best[0]
