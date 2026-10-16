"""
WELD PATH TEACH-AND-REPLAY PENDANT — HITBOT Z-Arm S922.

Drag the end effector along a weld seam by hand (or jog it there with the
on-screen buttons), click RECORD POINT at each spot, save the path, replay it.

Motion only: this tool has no notion of a torch, and never touches a digital
output. Wire welding in later — see _set_do() in pick_carton.py for the pattern.

    .venv/bin/python weld_teach.py

REQUIREMENTS
    The arm must be in AUTO/REMOTE on the teach pendant with the e-stop
    released, or every command comes back "cmd:0:robot errcode:NN".
    There is no TCP servo-on; enabling is a pendant action.

SAFETY
    The STOP button halts the ServoJ stream within ~100 ms. That stops
    *commanding* — it is not a controller-level halt and it is NOT an e-stop.
    Keep the physical e-stop in hand whenever this program is running.
"""

import json
import math
import os
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, "/home/xp/z-arms922/hitbot_ws")
from hitbot_tcp_api import HitbotTCPInterface

# ---------------------------------------------------------------- constants
ARM_IP = "192.168.58.2"
ARM_PORT = 8080

DRAG_ID = 333          # DragTeachSwitch — probed 2026-07-28: payload "1" = ok
QUERY_ID = 377         # every Get* query
MODE_ID = 303
RESET_ID = 107

MAX_JOINT_STEP_DEG = 90.0   # IK configuration-flip guard (from pick_carton)
IK_SELFCHECK_DEG = 0.5
APPROACH_WARN_DEG = 30.0    # ask again if the first waypoint is further than this

JOG_STEPS_MM = (1.0, 5.0, 20.0)
JOG_STEPS_DEG = (1.0, 2.0, 5.0)
JOG_TIME_S = 0.6

SEG_TIME_DEFAULT = 1.2      # s per replay waypoint at 1.00x
SEG_TIME_MIN = 0.3
WP_PER_100MM = 5            # Cartesian interpolation density
WP_MIN, WP_MAX = 2, 60

DT = 0.1                    # ServoJ cycle
POLL_MS = 200               # live-readout period
PATH_EXT = ".weldpath.json"

# Abort flag. Set from the Tk thread by STOP, polled by servo_to() each tick.
ABORT = threading.Event()


# ------------------------------------------------------- robot helpers
# Copied from pick_carton.py:94-133 rather than imported: importing that module
# executes `import open3d` and an importlib exec of carton_pose.py at module
# scope, which drags in the whole camera stack and only works when cwd is the
# project dir. These few functions are stable; keeping them local makes this a
# genuinely single-file, camera-free tool.
#
# ONE DELIBERATE CHANGE: the originals raise SystemExit. Here they raise
# RuntimeError — a SystemExit from a worker thread would tear down the
# interpreter out from under the GUI.

def query(arm, cmd):
    resp = arm._send_cmd(QUERY_ID, cmd).split("III")
    if len(resp) < 6 or "errcode" in resp[4]:
        raise RuntimeError(f"{cmd} -> {resp[4] if len(resp) >= 6 else resp!r}")
    return [float(v) for v in resp[4].split(",")]


# hover_test.py:60-70 probes both variants; the 8-arg form is the one that
# validated on this arm. Keep the fallback in case a controller update changes it.
IK_FORMATS = ("GetInverseKin(0,{x},{y},{z},{rx},{ry},{rz},-1)",
              "GetInverseKin(0,{x},{y},{z},{rx},{ry},{rz})")


def tcp(arm):
    return query(arm, "GetActualTCPPose()")


def joints(arm):
    return query(arm, "GetActualJointPosDegree()")


def solve_ik(arm, pose, ik_fmt):
    return query(arm, ik_fmt.format(x=pose[0], y=pose[1], z=pose[2],
                                    rx=pose[3], ry=pose[4], rz=pose[5]))


def validate_ik(arm):
    """Feed IK the CURRENT pose and require the CURRENT joints back.

    Returns the working format string. Raises if neither variant agrees --
    that means we cannot trust IK, so we must not command Cartesian motion.
    """
    j0, p0 = joints(arm), tcp(arm)
    last = None
    for fmt in IK_FORMATS:
        try:
            jik = solve_ik(arm, p0, fmt)
            err = max(abs(a - b) for a, b in zip(jik, j0))
            if err < IK_SELFCHECK_DEG:
                return fmt
            last = f"{fmt.count(',') + 1}-arg form off by {err:.2f} deg"
        except Exception as e:
            last = f"{fmt.count(',') + 1}-arg form failed: {str(e)[:60]}"
    raise RuntimeError(f"IK self-check failed ({last}) — refusing Cartesian motion")


def cartesian_path(arm, p_from, p_to, n_wp, j_seed, ik_fmt):
    """Straight-line waypoints p_from -> p_to (orientation of p_to), IK-checked.

    Builds and validates the ENTIRE segment before any of it is commanded, so an
    IK failure or a configuration flip is caught while the arm is still parked.
    """
    path = [list(j_seed)]
    for i in range(1, n_wp + 1):
        a = i / n_wp
        wp = [p_from[k] + a * (p_to[k] - p_from[k]) for k in range(3)] + list(p_to[3:])
        jt = solve_ik(arm, wp, ik_fmt)
        step = max(abs(x - y) for x, y in zip(jt, path[-1]))
        if step > MAX_JOINT_STEP_DEG:
            raise RuntimeError(f"IK jump {step:.1f} deg mid-path — segment refused. "
                               f"Try Joint mode.")
        path.append(jt)
    return path[1:]


def servo_to(arm, target, duration_s, dt=DT):
    """Local copy of move_to_joints_smoothly with a per-tick abort check.

    The driver's version (hitbot_tcp_api.py:65) has no escape hatch, which would
    cap STOP latency at one whole segment. Returns True if completed, False if
    aborted. Aborting stops the ServoJ stream; the arm decelerates to the last
    commanded target. That is not a controller halt -- the e-stop is.
    """
    start = joints(arm)
    steps = max(1, int(duration_s / dt))
    for i in range(1, steps + 1):
        if ABORT.is_set():
            return False
        a = i / steps
        arm.servo_j([start[k] + a * (target[k] - start[k]) for k in range(6)],
                    speed=50.0, acc=50.0, t=dt)
        time.sleep(dt)
    return True


def run_path(arm, path, seg_time):
    for jt in path:
        if not servo_to(arm, jt, seg_time):
            return False
    return True


def drag_teach(arm, on):
    """Toggle drag-teach. -> (ok, detail). Never raises.

    Probed on the real arm 2026-07-28: both directions reply payload "1" in
    ~12 ms and the socket stays in sync. Whether the brakes actually release is
    a physical question the readout answers, not the reply.
    """
    try:
        resp = arm._send_cmd(DRAG_ID, f"DragTeachSwitch({1 if on else 0})")
    except socket.timeout:
        _drain(arm)
        return False, "no reply in 5 s (command unsupported?)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    parts = resp.split("III")
    payload = parts[4].strip() if len(parts) >= 6 else resp.strip()
    if "errcode" in payload:
        return False, f"robot rejected: {payload}"
    if payload == "1":
        return True, "ok"
    return True, f"unexpected payload {payload!r} (treating as accepted)"


def _drain(arm):
    """Flush a late reply after a timeout so it cannot corrupt the next read.

    The probe showed no desync, so this is defensive — but it lives right at the
    one command whose framing we cannot fully vouch for.
    """
    try:
        arm.sock.settimeout(0.3)
        while True:
            if "/b/f" in arm.sock.recv(4096).decode(errors="replace"):
                break
    except Exception:
        pass
    finally:
        try:
            arm.sock.settimeout(5.0)
        except Exception:
            pass


# ------------------------------------------------------------- waypoints
@dataclass
class Waypoint:
    joints_deg: list
    pose: list
    label: str = ""
    t_captured: float = field(default_factory=time.time)

    def summary(self, i):
        p = self.pose
        tag = f"  {self.label}" if self.label else ""
        return (f"{i:02d}  x{p[0]:8.1f} y{p[1]:8.1f} z{p[2]:8.1f}   "
                f"rx{p[3]:7.1f} ry{p[4]:7.1f} rz{p[5]:7.1f}{tag}")


PATH_FORMAT = "hitbot_weld_path"
PATH_VERSION = 1


def save_path(fn, wps, seg_time, mode):
    doc = {
        "format": PATH_FORMAT,
        "version": PATH_VERSION,
        "robot": "HITBOT Z-Arm S922",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "units": {"position": "mm",
                  "orientation": "deg (rx,ry,rz from GetActualTCPPose)",
                  "joints": "deg"},
        "notes": "Motion-only trajectory. No torch/DO control.",
        "replay_defaults": {"seg_time_s": seg_time, "wp_per_100mm": WP_PER_100MM,
                            "mode": mode},
        "waypoints": [{"idx": i, "label": w.label, "t_captured": w.t_captured,
                       "joints_deg": w.joints_deg, "pose": w.pose}
                      for i, w in enumerate(wps)],
    }
    with open(fn, "w") as f:
        json.dump(doc, f, indent=2)


def load_path(fn):
    with open(fn) as f:
        doc = json.load(f)
    if doc.get("format") != PATH_FORMAT:
        raise ValueError(f"not a weld path file (format={doc.get('format')!r})")
    if doc.get("version") != PATH_VERSION:
        raise ValueError(f"unsupported version {doc.get('version')!r}")
    wps = []
    for i, w in enumerate(doc.get("waypoints", [])):
        j, p = w.get("joints_deg"), w.get("pose")
        if not (isinstance(j, list) and len(j) == 6):
            raise ValueError(f"waypoint {i}: joints_deg must be 6 values")
        if not (isinstance(p, list) and len(p) == 6):
            raise ValueError(f"waypoint {i}: pose must be 6 values")
        wps.append(Waypoint([float(v) for v in j], [float(v) for v in p],
                            w.get("label", ""), w.get("t_captured", 0.0)))
    if not wps:
        raise ValueError("file contains no waypoints")
    return wps, doc.get("replay_defaults", {})


# ----------------------------------------------------------- robot worker
class RobotWorker(threading.Thread):
    """Every socket byte goes through this one thread.

    The driver's socket is shared and non-reentrant, and a replay streams a
    ServoJ every 100 ms. A pose poller on the Tk thread would interleave writes
    and steal replies -- so polling happens here, and only while idle.
    """

    def __init__(self, out_q):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.in_q = queue.Queue()
        self.arm = None
        self.ik_fmt = IK_FORMATS[0]
        self.drag_on = False
        self._stop = threading.Event()

    # -- called from the Tk thread --
    def submit(self, fn, *a, **kw):
        self.in_q.put((fn, a, kw))

    def shutdown(self):
        self._stop.set()

    # -- worker side --
    def emit(self, tag, payload=None):
        self.out_q.put((tag, payload))

    def run(self):
        while not self._stop.is_set():
            try:
                fn, a, kw = self.in_q.get(timeout=0.2)
            except queue.Empty:
                self._poll_state()
                continue
            try:
                fn(self, *a, **kw)
            except Exception as e:
                # A robot fault must never kill the worker.
                self._safe_drag_off()
                self.emit("error", f"{type(e).__name__}: {e}")
            finally:
                self.emit("done", None)

    def _poll_state(self):
        if self.arm is None:
            return
        try:
            self.emit("state", (joints(self.arm), tcp(self.arm)))
        except Exception as e:
            self.arm = None
            self.emit("lost", str(e))

    def _safe_drag_off(self):
        if self.arm is not None and self.drag_on:
            try:
                drag_teach(self.arm, False)
            except Exception:
                pass
            self.drag_on = False
            self.emit("drag", False)

    def ensure_drag_off(self):
        """Drag mode likely parks the servo loop; a ServoJ issued while it is on
        is undefined. Always run this before commanding motion."""
        if self.arm is None:
            raise RuntimeError("not connected")
        drag_teach(self.arm, False)
        if self.drag_on:
            self.drag_on = False
            self.emit("drag", False)
        self.arm.switch_mode(0)
        self.arm.reset_errors()
        j = joints(self.arm)          # confirm we are readable before moving
        if len(j) != 6:
            raise RuntimeError("robot not responding with valid joint data")
        return j


# --------------------------------------------------------------- Tk app
class TeachPendantApp:
    def __init__(self, root):
        self.root = root
        root.title("Weld Path Teach — HITBOT S922")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.geometry("700x960")
        root.minsize(640, 860)

        self.out_q = queue.Queue()
        self.worker = RobotWorker(self.out_q)
        self.worker.start()

        self.wps = []
        self.dirty = False
        self.busy = False
        self.connected = False
        self.drag_supported = True
        self.drag_on = False
        self.busy_widgets = []
        self._pending_run = None

        self._build()
        self.root.after(POLL_MS, self._drain)

    # ------------------------------------------------------------ layout
    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        pad = dict(padx=6, pady=3)

        # --- connection ---
        top = ttk.Frame(r)
        top.grid(row=0, column=0, sticky="ew", **pad)
        self.btn_conn = ttk.Button(top, text="CONNECT", width=12,
                                   command=self.do_connect)
        self.btn_conn.pack(side="left")
        self.lbl_conn = ttk.Label(top, text="● Disconnected", foreground="#b00")
        self.lbl_conn.pack(side="left", padx=10)
        ttk.Label(top, text=f"{ARM_IP}:{ARM_PORT}",
                  foreground="#666").pack(side="right")

        # --- live readout ---
        live = ttk.LabelFrame(r, text="Live pose")
        live.grid(row=1, column=0, sticky="ew", **pad)
        mono = ("TkFixedFont", 10)
        self.lbl_tcp = ttk.Label(live, text="TCP   —", font=mono)
        self.lbl_tcp.pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_rot = ttk.Label(live, text="      —", font=mono)
        self.lbl_rot.pack(anchor="w", padx=8)
        self.lbl_jnt = ttk.Label(live, text="JNT   —", font=mono)
        self.lbl_jnt.pack(anchor="w", padx=8, pady=(0, 4))

        # --- teach ---
        teach = ttk.LabelFrame(r, text="Teach")
        teach.grid(row=2, column=0, sticky="ew", **pad)

        self.btn_drag = tk.Button(teach, text="DRAG TEACH: OFF",
                                  command=self.toggle_drag, height=2,
                                  bg="#ddd", activebackground="#cfc")
        self.btn_drag.pack(fill="x", padx=8, pady=(6, 2))
        self.busy_widgets.append(self.btn_drag)
        self.lbl_drag = ttk.Label(teach, text="", foreground="#b00",
                                  wraplength=520, justify="left")
        self.lbl_drag.pack(anchor="w", padx=8)

        jog = ttk.LabelFrame(teach, text="Jog")
        jog.pack(fill="x", padx=8, pady=6)

        steps = ttk.Frame(jog)
        steps.pack(anchor="w", pady=(4, 6))
        ttk.Label(steps, text="step").pack(side="left", padx=(4, 6))
        self.step_mm = tk.DoubleVar(value=5.0)
        for v in JOG_STEPS_MM:
            b = ttk.Radiobutton(steps, text=f"{v:g}", value=v, variable=self.step_mm)
            b.pack(side="left")
            self.busy_widgets.append(b)
        ttk.Label(steps, text="mm").pack(side="left", padx=(2, 14))
        self.step_deg = tk.DoubleVar(value=2.0)
        for v in JOG_STEPS_DEG:
            b = ttk.Radiobutton(steps, text=f"{v:g}", value=v, variable=self.step_deg)
            b.pack(side="left")
            self.busy_widgets.append(b)
        ttk.Label(steps, text="deg").pack(side="left", padx=2)

        grid = ttk.Frame(jog)
        grid.pack(anchor="w", padx=4, pady=(0, 6))
        for col, (lin, rot) in enumerate((("X", "RX"), ("Y", "RY"), ("Z", "RZ"))):
            self._jog_pair(grid, col, 0, lin, lin.lower(), False)
            self._jog_pair(grid, col, 1, rot, rot.lower(), True)
        # J1 escape hatch: at a Cartesian singularity IK jog refuses everything
        # and joint jog is the only way out.
        self._joint_pair(grid, 3)

        self.btn_rec = tk.Button(teach, text="RECORD POINT", height=2,
                                 bg="#2e7d32", fg="white",
                                 activebackground="#43a047",
                                 command=self.do_record,
                                 font=("TkDefaultFont", 10, "bold"))
        self.btn_rec.pack(fill="x", padx=8, pady=(2, 8))
        self.busy_widgets.append(self.btn_rec)

        # --- path ---
        path = ttk.LabelFrame(r, text="Path")
        path.grid(row=3, column=0, sticky="nsew", **pad)
        r.rowconfigure(3, weight=1)
        path.columnconfigure(0, weight=1)
        path.rowconfigure(0, weight=1)

        lb = ttk.Frame(path)
        lb.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=6)
        sb = ttk.Scrollbar(lb)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(lb, height=9, font=("TkFixedFont", 9),
                                  yscrollcommand=sb.set, activestyle="none",
                                  exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)

        side = ttk.Frame(path)
        side.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=6)
        for txt, cmd in (("Delete", self.do_delete), ("Clear", self.do_clear),
                         ("Save…", self.do_save), ("Load…", self.do_load)):
            b = ttk.Button(side, text=txt, width=10, command=cmd)
            b.pack(pady=2)
            self.busy_widgets.append(b)

        # --- replay ---
        rep = ttk.LabelFrame(r, text="Replay")
        rep.grid(row=4, column=0, sticky="ew", **pad)

        row1 = ttk.Frame(rep)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="mode").pack(side="left")
        self.mode = tk.StringVar(value="joint")
        for txt, val in (("Joint (safe)", "joint"), ("Cartesian", "cartesian")):
            b = ttk.Radiobutton(row1, text=txt, value=val, variable=self.mode)
            b.pack(side="left", padx=6)
            self.busy_widgets.append(b)
        ttk.Label(row1, text="passes").pack(side="left", padx=(18, 4))
        self.passes = tk.IntVar(value=1)
        sp = ttk.Spinbox(row1, from_=1, to=99, width=4, textvariable=self.passes)
        sp.pack(side="left")
        self.busy_widgets.append(sp)

        row2 = ttk.Frame(rep)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="speed").pack(side="left")
        self.speed = tk.DoubleVar(value=1.0)
        sc = ttk.Scale(row2, from_=0.25, to=2.0, variable=self.speed,
                       command=lambda _: self.lbl_speed.config(
                           text=f"{self.speed.get():.2f}x"))
        sc.pack(side="left", fill="x", expand=True, padx=8)
        self.busy_widgets.append(sc)
        self.lbl_speed = ttk.Label(row2, text="1.00x", width=6)
        self.lbl_speed.pack(side="left")

        row3 = ttk.Frame(rep)
        row3.pack(fill="x", padx=8, pady=(4, 8))
        self.btn_dry = ttk.Button(row3, text="DRY RUN", width=14,
                                  command=lambda: self.do_run(dry=True))
        self.btn_dry.pack(side="left")
        self.busy_widgets.append(self.btn_dry)
        self.btn_run = tk.Button(row3, text="RUN", bg="#1565c0", fg="white",
                                 activebackground="#1e88e5", height=2,
                                 font=("TkDefaultFont", 10, "bold"),
                                 command=lambda: self.do_run(dry=False))
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.busy_widgets.append(self.btn_run)

        # --- STOP: the one control that is never disabled ---
        self.btn_stop = tk.Button(r, text="S T O P", bg="#c62828", fg="white",
                                  activebackground="#e53935", height=2,
                                  font=("TkDefaultFont", 13, "bold"),
                                  command=self.do_stop)
        self.btn_stop.grid(row=5, column=0, sticky="ew", padx=6, pady=(2, 4))

        self.status = ttk.Label(r, text="not connected", relief="sunken",
                                anchor="w", wraplength=560)
        self.status.grid(row=6, column=0, sticky="ew", padx=6, pady=(0, 6))

        self._set_enabled(False)

    def _jog_pair(self, parent, col, sub, label, axis, is_rot):
        f = ttk.Frame(parent)
        f.grid(row=sub, column=col, padx=6, pady=2, sticky="w")
        for sign, txt in ((-1, f"{label}-"), (+1, f"{label}+")):
            b = ttk.Button(f, text=txt, width=5,
                           command=lambda a=axis, s=sign, rot=is_rot:
                           self.do_jog(a, s, rot))
            b.pack(side="left", padx=1)
            self.busy_widgets.append(b)

    def _joint_pair(self, parent, col):
        f = ttk.Frame(parent)
        f.grid(row=0, column=col, padx=(16, 6), pady=2, sticky="w")
        for sign, txt in ((-1, "J1-"), (+1, "J1+")):
            b = ttk.Button(f, text=txt, width=5,
                           command=lambda s=sign: self.do_jog_joint(0, s))
            b.pack(side="left", padx=1)
            self.busy_widgets.append(b)

    # ------------------------------------------------------------ helpers
    def _set_enabled(self, on):
        state = "normal" if on else "disabled"
        for w in self.busy_widgets:
            try:
                w.config(state=state)
            except tk.TclError:
                pass

    def _set_busy(self, busy):
        self.busy = busy
        self._set_enabled(self.connected and not busy)

    def say(self, msg):
        self.status.config(text=msg)

    def refresh_list(self):
        sel = self.listbox.curselection()
        self.listbox.delete(0, tk.END)
        for i, w in enumerate(self.wps):
            self.listbox.insert(tk.END, w.summary(i))
        if sel and sel[0] < len(self.wps):
            self.listbox.selection_set(sel[0])

    # ------------------------------------------------------- queue drain
    def _drain(self):
        try:
            while True:
                tag, payload = self.out_q.get_nowait()
                self._handle(tag, payload)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._drain)

    def _handle(self, tag, payload):
        if tag == "state":
            j, p = payload
            self.lbl_tcp.config(text=f"TCP   X {p[0]:9.1f}  Y {p[1]:9.1f}  "
                                     f"Z {p[2]:9.1f}   mm")
            self.lbl_rot.config(text=f"      RX{p[3]:9.1f}  RY{p[4]:9.1f}  "
                                     f"RZ{p[5]:9.1f}   deg")
            self.lbl_jnt.config(text="JNT  " + " ".join(f"{v:8.2f}" for v in j)
                                + "   deg")
        elif tag == "status":
            self.say(payload)
        elif tag == "connected":
            self.connected = True
            self.lbl_conn.config(text="● Connected", foreground="#080")
            self.btn_conn.config(text="CONNECT", state="disabled")
            self._set_busy(False)
        elif tag == "drag":
            self.drag_on = bool(payload)
            self.btn_drag.config(
                text=f"DRAG TEACH: {'ON' if self.drag_on else 'OFF'}",
                bg="#c8e6c9" if self.drag_on else "#ddd")
        elif tag == "drag_unsupported":
            self.drag_supported = False
            self.btn_drag.config(state="disabled", text="DRAG TEACH: unavailable")
            if self.btn_drag in self.busy_widgets:
                self.busy_widgets.remove(self.btn_drag)
            self.lbl_drag.config(text=f"Drag-teach unavailable ({payload}) — "
                                      f"use the jog buttons.")
        elif tag == "recorded":
            j, p = payload
            self.wps.append(Waypoint(j, p))
            self.dirty = True
            self.refresh_list()
            self.listbox.see(tk.END)
            self.say(f"recorded point {len(self.wps) - 1}")
        elif tag == "preflight":
            self._confirm_run(payload)
        elif tag == "error":
            self.say(f"ERROR: {payload}")
            messagebox.showerror("Robot error", payload)
        elif tag == "lost":
            self.connected = False
            self.lbl_conn.config(text="● Disconnected", foreground="#b00")
            self.btn_conn.config(text="RECONNECT", state="normal")
            self._set_enabled(False)
            self.say("Connection lost — the arm may still be executing its last "
                     "ServoJ. Use the e-stop if it is moving.")
        elif tag == "done":
            self._set_busy(False)

    # ------------------------------------------------------------ actions
    def do_connect(self):
        self.btn_conn.config(state="disabled")
        self.say("connecting…")

        def job(w):
            arm = HitbotTCPInterface(ip=ARM_IP, port=ARM_PORT)
            ok = arm.connect()
            if not ok:
                # connect() returns False for both "no route" and "connected but
                # not in auto mode" — the operator's fix differs, so distinguish.
                try:
                    query(arm, "GetActualJointPosDegree()")
                except Exception as e:
                    raise RuntimeError(
                        f"connect failed ({e}). Check the cable, and check the "
                        f"pendant is in AUTO/REMOTE with the e-stop released.")
            w.arm = arm
            j = joints(arm)
            if len(j) != 6:
                raise RuntimeError("connected but no valid joint data — "
                                   "pendant probably not in AUTO/REMOTE")
            drag_teach(arm, False)      # clear any state left by a prior session
            w.drag_on = False
            try:
                w.ik_fmt = validate_ik(arm)
                w.emit("status", "connected — IK validated ✓")
            except Exception as e:
                # Jog and Cartesian replay need IK; Joint replay does not.
                w.emit("status", f"connected — WARNING: {e}. Joint mode only.")
            w.emit("connected", None)

        self.worker.submit(job)

    def toggle_drag(self):
        if not self.connected:
            return
        want = not self.drag_on
        if want and not messagebox.askokcancel(
                "Enable drag-teach?",
                "The arm may drop under its own weight when drag-teach engages.\n\n"
                "Support the arm and keep the e-stop in hand.\n\nContinue?"):
            return
        self._set_busy(True)

        def job(w, on=want):
            ok, detail = drag_teach(w.arm, on)
            if not ok:
                if on:
                    w.emit("drag_unsupported", detail)
                    w.emit("status", f"drag-teach refused: {detail}")
                return
            w.drag_on = on
            w.emit("drag", on)
            w.emit("status",
                   "Drag ON — move the arm by hand. If it does not yield, click "
                   "DRAG OFF and use jog." if on else "Drag OFF.")

        self.worker.submit(job)

    def do_jog(self, axis, sign, is_rot):
        if not self.connected:
            return
        step = self.step_deg.get() if is_rot else self.step_mm.get()
        idx = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}[axis]
        self._set_busy(True)
        self.say(f"jog {axis.upper()} {sign * step:+g}…")

        def job(w):
            w.ensure_drag_off()
            j0 = joints(w.arm)
            p1 = list(tcp(w.arm))
            p1[idx] += sign * step
            jt = solve_ik(w.arm, p1, w.ik_fmt)
            jump = max(abs(a - b) for a, b in zip(jt, j0))
            if jump > MAX_JOINT_STEP_DEG:
                raise RuntimeError(f"IK jump {jump:.1f} deg — jog refused "
                                   f"(singularity or joint limit?). Try J1 jog.")
            ABORT.clear()
            done = servo_to(w.arm, jt, JOG_TIME_S)
            w.emit("status", "jog done" if done else "jog ABORTED")

        self.worker.submit(job)

    def do_jog_joint(self, jn, sign):
        if not self.connected:
            return
        step = self.step_deg.get()
        self._set_busy(True)
        self.say(f"jog J{jn + 1} {sign * step:+g}°…")

        def job(w):
            w.ensure_drag_off()
            jt = joints(w.arm)
            jt[jn] += sign * step
            ABORT.clear()
            done = servo_to(w.arm, jt, JOG_TIME_S)
            w.emit("status", "jog done" if done else "jog ABORTED")

        self.worker.submit(job)

    def do_record(self):
        if not self.connected:
            return
        self._set_busy(True)

        def job(w):
            # Read fresh from the robot — never build a waypoint from the
            # cached readout, which may be up to POLL_MS stale.
            w.emit("recorded", (joints(w.arm), tcp(w.arm)))

        self.worker.submit(job)

    def do_delete(self):
        sel = self.listbox.curselection()
        if not sel:
            self.say("select a point to delete")
            return
        self.wps.pop(sel[0])
        self.dirty = True
        self.refresh_list()
        self.say(f"deleted point {sel[0]}")

    def do_clear(self):
        if not self.wps:
            return
        if messagebox.askokcancel("Clear path", f"Discard all {len(self.wps)} "
                                                f"points?"):
            self.wps.clear()
            self.dirty = False
            self.refresh_list()
            self.say("path cleared")

    def do_save(self):
        if not self.wps:
            self.say("nothing to save")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=PATH_EXT, initialdir=os.getcwd(),
            filetypes=[("Weld path", f"*{PATH_EXT}"), ("All files", "*")])
        if not fn:
            return
        try:
            save_path(fn, self.wps, self._seg_time(), self.mode.get())
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.dirty = False
        self.say(f"saved {len(self.wps)} points -> {os.path.basename(fn)}")

    def do_load(self):
        if self.dirty and self.wps and not messagebox.askokcancel(
                "Discard unsaved path?",
                f"{len(self.wps)} unsaved points will be lost. Continue?"):
            return
        fn = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            filetypes=[("Weld path", f"*{PATH_EXT}"), ("JSON", "*.json"),
                       ("All files", "*")])
        if not fn:
            return
        try:
            wps, defaults = load_path(fn)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.wps = wps
        self.dirty = False
        if defaults.get("mode") in ("joint", "cartesian"):
            self.mode.set(defaults["mode"])
        self.refresh_list()
        self.say(f"loaded {len(wps)} points from {os.path.basename(fn)} — "
                 f"DRY RUN first")

    def _seg_time(self):
        return max(SEG_TIME_MIN, SEG_TIME_DEFAULT / max(0.01, self.speed.get()))

    def do_run(self, dry):
        if not self.connected or len(self.wps) < 2:
            self.say("need at least 2 points")
            return
        self._set_busy(True)
        self.say("pre-flight check…")
        wps = list(self.wps)

        def job(w):
            # Pre-flight is read-only: compute how far the arm must travel to
            # reach point 0, so the GUI can name the number in the dialog.
            w.ensure_drag_off()
            j_now = joints(w.arm)
            delta = max(abs(a - b) for a, b in zip(j_now, wps[0].joints_deg))
            w.emit("preflight", (wps, delta, dry))

        self.worker.submit(job)

    def _confirm_run(self, payload):
        wps, delta, dry = payload
        mode = self.mode.get()
        seg = self._seg_time()
        passes = max(1, self.passes.get())
        kind = "DRY RUN (no motion)" if dry else "RUN"
        if not messagebox.askokcancel(
                f"{kind}?",
                f"{len(wps)} waypoints, {passes} pass(es)\n"
                f"mode: {mode}\nsegment time: {seg:.2f} s\n"
                f"first waypoint is {delta:.1f}° from the current pose\n\n"
                f"{'No motion will be commanded.' if dry else 'The arm will move. E-stop in hand.'}"):
            self._set_busy(False)
            return
        if not dry and delta > APPROACH_WARN_DEG and not messagebox.askokcancel(
                "Large approach move",
                f"The first waypoint is {delta:.1f}° from the current pose.\n"
                f"The arm will make a large move to get there.\n\nContinue?"):
            self._set_busy(False)
            return
        self._set_busy(True)
        self.worker.submit(self._replay_job, wps, mode, seg, passes, dry, delta)

    @staticmethod
    def _replay_job(w, wps, mode, seg, passes, dry, delta):
        ABORT.clear()
        w.ensure_drag_off()
        approach = max(4.0, 0.05 * delta)

        def go(target, dur, what):
            if dry:
                w.emit("status", f"[dry] {what} -> "
                                 f"{[round(v, 2) for v in target]}")
                return True
            return servo_to(w.arm, target, dur)

        for p in range(passes):
            tag = f"pass {p + 1}/{passes}" if passes > 1 else ""
            # The documented jump hazard: a path's first waypoint is not the
            # arm's current pose, so approach it slowly and deliberately.
            w.emit("status", f"{tag} approaching point 0 ({approach:.1f}s)…")
            if not go(wps[0].joints_deg, approach, "approach"):
                w.emit("status", "ABORTED during approach")
                return

            for i in range(len(wps) - 1):
                a, b = wps[i], wps[i + 1]
                w.emit("status", f"{tag} segment {i}->{i + 1}")
                if mode == "joint":
                    if not go(b.joints_deg, seg, f"seg {i}->{i + 1}"):
                        w.emit("status", f"ABORTED at segment {i}->{i + 1}")
                        return
                else:
                    dist = math.dist(a.pose[:3], b.pose[:3])
                    n = max(WP_MIN, min(WP_MAX,
                                        math.ceil(dist / 100.0 * WP_PER_100MM)))
                    # Seed with the MEASURED joints, not the stored ones — they
                    # drift slightly after the preceding segment.
                    seed = wps[i].joints_deg if dry else joints(w.arm)
                    try:
                        path = cartesian_path(w.arm, a.pose, b.pose, n, seed,
                                              w.ik_fmt)
                    except RuntimeError as e:
                        # cartesian_path validates the whole segment before
                        # commanding any of it, so we are still parked at point
                        # i. Stop here rather than skipping ahead, which would
                        # cut a chord through unknown space.
                        w.emit("status", f"Segment {i}->{i + 1} failed: {e} "
                                         f"Stopped at waypoint {i}.")
                        return
                    if dry:
                        w.emit("status", f"[dry] seg {i}->{i + 1}: {n} waypoints, "
                                         f"{dist:.1f} mm, IK ok")
                    elif not run_path(w.arm, path, seg):
                        w.emit("status", f"ABORTED at segment {i}->{i + 1}")
                        return

            if p + 1 < passes:
                w.emit("status", f"{tag} returning to start…")
                if not go(wps[0].joints_deg, approach, "return"):
                    w.emit("status", "ABORTED during return")
                    return

        w.emit("status", "DRY RUN complete — no motion commanded" if dry
               else "replay complete")

    def do_stop(self):
        # Pure Tk-thread work: never enqueue a job here, the queue may be behind
        # a running move.
        ABORT.set()
        self.say("STOP — servo stream halted. This is not an e-stop; use the "
                 "physical e-stop if the arm is still moving.")

    def on_close(self):
        if self.busy and not messagebox.askokcancel(
                "Quit", "A job is running. Stop it and quit?"):
            return
        if self.dirty and self.wps and not messagebox.askokcancel(
                "Quit", f"{len(self.wps)} unsaved points will be lost. Quit?"):
            return
        ABORT.set()
        if self.worker.arm is not None:
            try:
                drag_teach(self.worker.arm, False)
                self.worker.arm.sock.close()
            except Exception:
                pass
        self.worker.shutdown()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    TeachPendantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
