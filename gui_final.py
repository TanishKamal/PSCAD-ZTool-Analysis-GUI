"""
Master GUI for:
1) Original waveform analyzer (kept completely independent)
2) Z-tool analysis driven by a selected .pswx workspace

Requirements:
- Python 3.11
- Original waveform analyzer:
    C:/Users/Asus/TwoLevelVSC_GUI/python/gui_improved.py
- Z-tool repository root:
    .../TwoLevelVSC_GUI/Z-tool-main/Z-tool-main/
- Z-tool workspace is selected by the user (e.g. sample.pswx)
- The selected workspace already contains the VSC project and AC Scan block.
"""

from __future__ import annotations

import os
import ctypes
import sys
import time
import traceback
import importlib
import threading
import subprocess
import io
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import linear_sum_assignment


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
GUI_ROOT = PROJECT_ROOT.parent.parent

# User's original, known-working waveform analyzer.
WAVEFORM_ANALYZER = GUI_ROOT / "python" / "gui_improved_311.py"

# Local MHI package used by the Z-tool installation.
MHI_ROOT = PROJECT_ROOT.parent / "mhi"

# Z-tool source package.
SOURCE_ROOT = PROJECT_ROOT / "Source"



# ============================================================
# PYTHON VERSION CHECK
# ============================================================

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "This combined GUI is configured for Python 3.11.\n\n"
        f"Current interpreter:\n{sys.executable}\n"
        f"Python version: {sys.version}\n\n"
        "Please activate the project's Python 3.11 .venv before running gui_ztool.py."
    )


# ============================================================
# PATH SETUP FOR Z-TOOL IMPORTS
# ============================================================

if not MHI_ROOT.exists():
    raise FileNotFoundError(
        "MHI package folder was not found:\n"
        f"{MHI_ROOT}\n\n"
        "Expected the local MHI package one level above the Z-tool repository."
    )

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(
        "Z-tool Source folder was not found:\n"
        f"{SOURCE_ROOT}"
    )

sys.path.insert(0, str(MHI_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import mhi.pscad  # noqa: E402


# ============================================================
# HELPERS
# ============================================================

def timestamp_folder_name(project_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{project_name}_{stamp}"


def safe_close_pscad(pscad_obj) -> None:
    if pscad_obj is None:
        return
    try:
        pscad_obj.quit()
    except Exception:
        pass


def get_project_names(pscad_obj):
    """Return loaded project/library names using current or legacy MHI APIs."""
    candidates = []

    for method_name in ("projects", "list_projects"):
        method = getattr(pscad_obj, method_name, None)
        if method is None:
            continue

        try:
            items = method()
        except Exception:
            continue

        if isinstance(items, dict):
            items = list(items.keys())
        elif isinstance(items, (tuple, list, set)):
            items = list(items)
        else:
            try:
                items = list(items)
            except Exception:
                items = []

        for item in items:
            if isinstance(item, str):
                candidates.append(item)
            else:
                name = getattr(item, "name", None)
                if name:
                    candidates.append(str(name))

        if candidates:
            break

    unique = []
    seen = set()
    for name in candidates:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def find_ac_scan_project(pscad_obj):
    """
    Find the first loaded project containing Z_tool:ACscan.

    The actual frequency_sweep implementation also checks for this
    component when no topology file is supplied.
    """
    names = get_project_names(pscad_obj)

    preferred = sorted(
        names,
        key=lambda n: (
            0 if ("vsc" in n.lower() or "2l" in n.lower()) else 1,
            n.lower(),
        ),
    )

    checked = []
    for name in preferred:
        try:
            project = pscad_obj.project(name)
        except Exception:
            continue

        try:
            scan = project.find_first("Z_tool:ACscan")
        except Exception:
            scan = None

        checked.append(name)
        if scan is not None:
            return name, project

    # Known example fallback.
    fallback = "Simple_2L_VSC_RLC"
    if fallback not in checked:
        try:
            project = pscad_obj.project(fallback)
            scan = project.find_first("Z_tool:ACscan")
            if scan is not None:
                return fallback, project
        except Exception:
            pass

    raise RuntimeError(
        "Could not find a project containing 'Z_tool:ACscan' in the selected workspace.\n\n"
        "Loaded projects found:\n"
        + (", ".join(names) if names else "(none detected)")
        + "\n\n"
        "The selected .pswx must already contain the VSC project with the AC Scan block."
    )


def track_eigenvalues(matrix_series: np.ndarray) -> np.ndarray:
    """Track two eigenvalue branches continuously across frequency."""
    n_freq = matrix_series.shape[0]
    tracked = np.empty((n_freq, 2), dtype=complex)

    vals = np.linalg.eigvals(matrix_series[0])
    tracked[0] = vals[:2]

    for k in range(1, n_freq):
        vals = np.linalg.eigvals(matrix_series[k])

        cost = np.empty((2, 2), dtype=float)
        for i in range(2):
            for j in range(2):
                cost[i, j] = abs(tracked[k - 1, i] - vals[j])

        rows, cols = linear_sum_assignment(cost)
        ordered = np.empty(2, dtype=complex)
        for r, c in zip(rows, cols):
            ordered[r] = vals[c]

        tracked[k] = ordered

    return tracked


def winding_number(curve: np.ndarray, critical=-1 + 0j) -> int:
    """Estimate net winding number around the critical point."""
    z = np.asarray(curve, dtype=complex) - critical
    if len(z) < 2:
        return 0

    z = np.concatenate([z, z[:1]])
    angles = np.unwrap(np.angle(z))
    total = angles[-1] - angles[0]
    return int(np.rint(total / (2.0 * np.pi)))



def detect_fortran_extension(workspace_dir: Path, project_name: str) -> str:
    """
    Detect the PSCAD compiler/output directory extension used by the project.

    The supplied Z-tool frequency_sweep.py defaults to '.gf46', but the
    user's working project has previously used a different compiler folder
    such as '.gf132'. The sweep builds out_dir as:

        workspace_dir + project_name + fortran_ext

    Therefore the GUI must pass the actual extension used by the project.
    """
    workspace_dir = Path(workspace_dir)

    prefix = f"{project_name}.gf"
    candidates = []

    try:
        for item in workspace_dir.iterdir():
            if item.is_dir() and item.name.lower().startswith(prefix.lower()):
                ext = item.name[len(project_name):]
                if ext.lower().startswith(".gf") and len(ext) > 3:
                    try:
                        mtime = item.stat().st_mtime
                    except OSError:
                        mtime = 0
                    candidates.append((mtime, ext))
    except OSError:
        candidates = []

    if candidates:
        # Prefer the most recently modified existing compiler directory.
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback to the version used by the supplied Z-tool code.
    return ".gf46"



# ============================================================
# MAIN APPLICATION
# ============================================================

class MasterPSCADGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PSCAD Analysis Suite")
        self.root.geometry("1500x940")
        self.root.minsize(1180, 760)
        self.root.configure(bg="#f5f7fa")

        # Waveform analyzer process state
        self.waveform_process = None
        self.waveform_started_at = 0.0

        # Z-tool state
        self.ztool_workspace = None
        self.ztool_project_name = None
        self.ztool_results_folder = None
        self.ztool_running = False
        self.last_ztool_data = None

        # Live simulation-status state
        self.sim_stage_var = tk.StringVar(value="Idle")
        self.sim_detail_var = tk.StringVar(value="Select a workspace and start a Z-tool scan.")
        self.sim_progress_var = tk.DoubleVar(value=0.0)

        # UI state
        self.active_page = "ztool"
        self.page_frames = {}
        self.nav_buttons = {}

        self._configure_style()
        self._build_gui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ========================================================
    # STYLE
    # ========================================================

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg = "#f5f7fa"
        panel = "#ffffff"
        panel2 = "#eef2f7"
        border = "#d5dbe5"
        text = "#17202d"
        muted = "#667085"
        accent = "#0078d4"
        accent_dark = "#0067b8"

        self.colors = {
            "bg": bg,
            "panel": panel,
            "panel2": panel2,
            "border": border,
            "text": text,
            "muted": muted,
            "accent": accent,
            "nav_dark": "#162033",
            "nav_hover": "#243554",
            "nav_active": "#0b6eac",
            "nav_text": "#ffffff",
            "accent_dark": accent_dark,
            "green": "#138a4b",
            "red": "#c62828",
            "amber": "#a86700",
            "white": "#ffffff",
            "terminal_bg": "#0f172a",
            "terminal_text": "#d7dee9",
            "terminal_muted": "#91a4bd",
        }

        style.configure("TFrame", background=bg)
        style.configure("Card.TLabelframe", background=panel, bordercolor=border,
                        relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=panel, foreground=accent,
                        font=("Segoe UI", 9, "bold"))
        style.configure("TLabel", background=bg, foreground=text, font=("Segoe UI", 9))
        style.configure("Card.TLabel", background=panel, foreground=text, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=bg, foreground=muted, font=("Segoe UI", 9))
        style.configure("CardMuted.TLabel", background=panel, foreground=muted, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=bg, foreground=text,
                        font=("Segoe UI Semibold", 21, "bold"))
        style.configure("PageTitle.TLabel", background=bg, foreground=text,
                        font=("Segoe UI Semibold", 17, "bold"))
        style.configure("Section.TLabel", background=panel, foreground=text,
                        font=("Segoe UI Semibold", 10, "bold"))
        style.configure("TButton", background=panel2, foreground=text,
                        bordercolor=border, focusthickness=0, padding=(12, 7),
                        font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", "#20324d")])
        style.configure("Accent.TButton", background=accent_dark, foreground="#ffffff",
                        bordercolor=accent_dark, padding=(18, 9),
                        font=("Segoe UI Semibold", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#0284c7")])
        style.configure("Danger.TButton", background="#7f1d1d", foreground="#ffe4e6",
                        bordercolor="#991b1b", padding=(12, 7), font=("Segoe UI", 9, "bold"))
        style.configure("TEntry", fieldbackground="#ffffff", foreground=text,
                        bordercolor=border, insertcolor=text, padding=7)
        style.configure("ReadOnly.TEntry", fieldbackground="#ffffff", foreground=text,
                        bordercolor=border, padding=7)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#16243a", foreground=muted,
                        padding=(16, 9), borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", accent_dark)],
                  foreground=[("selected", "#ffffff")])
        style.configure("Horizontal.TProgressbar", troughcolor="#dfe5ec",
                        background=accent, bordercolor=border, lightcolor=accent,
                        darkcolor=accent)

    # ========================================================
    # GUI
    # ========================================================

    def _build_gui(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors["bg"], height=78)
        header.pack(fill="x", padx=26, pady=(18, 6))
        header.pack_propagate(False)

        left_header = tk.Frame(header, bg=self.colors["bg"])
        left_header.pack(side="left", fill="y")
        tk.Label(left_header, text="PSCAD", bg=self.colors["bg"], fg=self.colors["accent"],
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left_header, text="Analysis Suite", style="Title.TLabel").pack(anchor="w")

        right_header = tk.Frame(header, bg=self.colors["bg"])
        right_header.pack(side="right", fill="y")
        tk.Label(right_header, text="PYTHON 3.11", bg=self.colors["panel2"],
                 fg=self.colors["muted"], font=("Segoe UI", 8, "bold"),
                 padx=10, pady=5).pack(side="right", pady=13)
        self.header_status = tk.Label(right_header, text="●  READY", bg=self.colors["bg"],
                                      fg=self.colors["green"], font=("Segoe UI", 9, "bold"))
        self.header_status.pack(side="right", padx=16)

        # Body: sidebar + content
        body = tk.Frame(self.root, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        sidebar = tk.Frame(body, bg=self.colors["panel"], width=210,
                           highlightbackground=self.colors["border"], highlightthickness=1)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="WORKSPACES", bg=self.colors["panel"], fg=self.colors["muted"],
                 font=("Segoe UI", 8, "bold"), padx=18).pack(anchor="w", pady=(22, 10))

        self._add_nav_button(sidebar, "waveform", "◈  Waveform Analyzer")
        self._add_nav_button(sidebar, "ztool", "◉  Z-Tool Analysis")

        tk.Frame(sidebar, bg=self.colors["panel"]).pack(expand=True, fill="both")

        self.content = tk.Frame(body, bg=self.colors["bg"])
        self.content.pack(side="left", fill="both", expand=True)

        # The waveform analyzer is an independent application. Selecting it
        # from the sidebar launches it directly instead of displaying a page
        # inside the master GUI.
        ztool_page = tk.Frame(self.content, bg=self.colors["bg"])
        self.page_frames["ztool"] = ztool_page

        self._build_ztool_tab(ztool_page)
        self._show_page("ztool")

    def _add_nav_button(self, parent, key, text):
        command = self.launch_waveform_analyzer if key == "waveform" else (lambda k=key: self._show_page(k))
        # Dark navigation buttons for stronger contrast on the light GUI.
        btn = tk.Button(
            parent, text=text, anchor="w", relief="flat", bd=0,
            bg=self.colors.get("nav_dark", "#162033"),
            fg=self.colors.get("nav_text", "#ffffff"),
            activebackground=self.colors.get("nav_hover", "#243554"),
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), padx=16, pady=11,
            cursor="hand2", command=command,
        )
        btn.pack(fill="x", padx=10, pady=3)
        self.nav_buttons[key] = btn

    def _show_page(self, key):
        self.active_page = key
        for name, frame in self.page_frames.items():
            frame.pack_forget()
            btn = self.nav_buttons.get(name)
            if btn:
                active = name == key
                btn.configure(
                    bg=self.colors["nav_active"] if active else self.colors["nav_dark"],
                    fg="#ffffff",
                    activebackground=self.colors["nav_active"] if active else self.colors["nav_hover"],
                )
        self.page_frames[key].pack(fill="both", expand=True)
        self.header_status.configure(text="●  Z-TOOL ANALYSIS")

    # ========================================================
    # WAVEFORM ANALYZER LAUNCH
    # ========================================================

    def _waveform_has_visible_window(self, pid):
        """Return True when the waveform child process owns a visible top-level window."""
        if os.name != "nt":
            return True

        user32 = ctypes.windll.user32
        found = False

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_proc(hwnd, _lparam):
            nonlocal found
            if not user32.IsWindowVisible(hwnd):
                return True
            window_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            if window_pid.value == pid:
                found = True
                return False
            return True

        user32.EnumWindows(enum_proc, 0)
        return found

    def launch_waveform_analyzer(self):
        """Launch the independent waveform analyzer and allow clean re-opening."""
        if self.waveform_process is not None:
            if self.waveform_process.poll() is None:
                # A closed Tk window can leave the Python child alive briefly
                # while PSCAD/MHI shuts down. Do not launch a duplicate while its
                # GUI is still visible. If no window remains after startup grace
                # period, treat the process as stale and clean it up.
                elapsed = time.monotonic() - self.waveform_started_at
                if self._waveform_has_visible_window(self.waveform_process.pid):
                    return
                if elapsed < 5.0:
                    return
                try:
                    self.waveform_process.terminate()
                    self.waveform_process.wait(timeout=2.0)
                except Exception:
                    try:
                        self.waveform_process.kill()
                    except Exception:
                        pass
            self.waveform_process = None
            self.waveform_started_at = 0.0

        if not WAVEFORM_ANALYZER.exists():
            messagebox.showerror(
                "Waveform Analyzer Not Found",
                f"Could not find:\n{WAVEFORM_ANALYZER}",
            )
            return

        try:
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            mhi_path = str(MHI_ROOT)
            env["PYTHONPATH"] = (
                mhi_path + os.pathsep + existing_pythonpath
                if existing_pythonpath else mhi_path
            )

            self.waveform_process = subprocess.Popen(
                [sys.executable, str(WAVEFORM_ANALYZER)],
                cwd=str(WAVEFORM_ANALYZER.parent),
                env=env,
            )
            self.waveform_started_at = time.monotonic()
            self.header_status.configure(text="●  WAVEFORM ANALYZER")
            threading.Thread(
                target=self._watch_waveform_process,
                args=(self.waveform_process,),
                daemon=True,
            ).start()

        except Exception as exc:
            self.waveform_process = None
            self.waveform_started_at = 0.0
            messagebox.showerror(
                "Waveform Analyzer Error",
                str(exc),
            )

    def _watch_waveform_process(self, process):
        """Clear the waveform subprocess handle after the child exits."""
        try:
            process.wait()
        except Exception:
            return

        def clear_if_same():
            if self.waveform_process is process:
                self.waveform_process = None
                self.waveform_started_at = 0.0
                if self.active_page == "ztool":
                    self.header_status.configure(text="●  Z-TOOL ANALYSIS", fg=self.colors["green"])

        try:
            self.root.after(0, clear_if_same)
        except Exception:
            pass

    # ========================================================
    # Z-TOOL TAB
    # ========================================================

    def _build_ztool_tab(self, parent):
        # Page heading
        heading = tk.Frame(parent, bg=self.colors["bg"])
        heading.pack(fill="x", pady=(4, 12))
        ttk.Label(heading, text="Z-Tool Analysis", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(heading, text="Frequency-domain stability analysis of the selected PSCAD workspace",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        # Workspace card
        workspace_frame = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        workspace_frame.pack(fill="x", pady=(0, 10))
        tk.Label(workspace_frame, text="WORKSPACE", bg=self.colors["panel"], fg=self.colors["accent"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=18, pady=(12, 3))
        row = tk.Frame(workspace_frame, bg=self.colors["panel"])
        row.pack(fill="x", padx=18, pady=(0, 10))
        self.ztool_workspace_var = tk.StringVar(value="Select a .pswx workspace...")
        entry = ttk.Entry(row, textvariable=self.ztool_workspace_var, state="readonly", style="ReadOnly.TEntry")
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(row, text="Browse .pswx", command=self.browse_ztool_workspace).pack(side="right")
        info = tk.Frame(workspace_frame, bg=self.colors["panel2"])
        info.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(info, text="AC SCAN PROJECT", bg=self.colors["panel2"], fg=self.colors["muted"],
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=12, pady=8)
        self.ztool_detected_project_var = tk.StringVar(value="Select workspace")
        self.ztool_project_badge = tk.Label(info, textvariable=self.ztool_detected_project_var,
                                            bg=self.colors["panel2"], fg=self.colors["amber"],
                                            font=("Segoe UI", 9, "bold"), anchor="e")
        self.ztool_project_badge.pack(side="right", padx=12)

        # Scan settings card
        settings = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        settings.pack(fill="x", pady=(0, 10))
        tk.Label(settings, text="SCAN SETTINGS", bg=self.colors["panel"], fg=self.colors["accent"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=18, pady=(12, 4))

        self.z_fmin = tk.StringVar(value="1.0")
        self.z_fmax = tk.StringVar(value="500.0")
        self.z_fpoints = tk.StringVar(value="400")
        self.z_fbase = tk.StringVar(value="0.5")
        self.z_tsnap = tk.StringVar(value="10")
        self.z_start_fft = tk.StringVar(value="1.0")
        self.z_fft_periods = tk.StringVar(value="1")
        self.z_dtinj = tk.StringVar(value="1")
        self.z_tstep = tk.StringVar(value="20.0")
        self.z_perturb = tk.StringVar(value="0.02")
        fields = [("f min (Hz)", self.z_fmin), ("f max (Hz)", self.z_fmax), ("points", self.z_fpoints),
                  ("f base (Hz)", self.z_fbase), ("snapshot (s)", self.z_tsnap), ("start FFT (s)", self.z_start_fft),
                  ("FFT periods", self.z_fft_periods), ("injection settle (s)", self.z_dtinj),
                  ("time step (us)", self.z_tstep), ("perturbation (pu)", self.z_perturb)]
        grid = tk.Frame(settings, bg=self.colors["panel"])
        grid.pack(fill="x", padx=14, pady=(0, 12))
        for i, (label, var) in enumerate(fields):
            r, c = divmod(i, 5)
            grid.columnconfigure(c, weight=1)
            cell = tk.Frame(grid, bg=self.colors["panel"])
            cell.grid(row=r, column=c, sticky="ew", padx=4, pady=4)
            tk.Label(cell, text=label, bg=self.colors["panel"], fg=self.colors["muted"],
                     font=("Segoe UI", 8)).pack(anchor="w")
            ttk.Entry(cell, textvariable=var).pack(fill="x", pady=(3, 0))

        # Actions/status
        action = tk.Frame(parent, bg=self.colors["bg"])
        action.pack(fill="x", pady=(0, 8))
        self.z_check_button = ttk.Button(action, text="✓  CHECK WORKSPACE", command=self.start_workspace_check)
        self.z_check_button.pack(side="left")
        self.z_run_button = ttk.Button(action, text="▶  RUN Z-TOOL", style="Accent.TButton", command=self.start_ztool_scan)
        self.z_run_button.pack(side="left", padx=8)
        self.z_open_results_button = ttk.Button(action, text="Open Results", state="disabled", command=self.open_ztool_results_folder)
        self.z_open_results_button.pack(side="left")
        self.z_status_var = tk.StringVar(value="Ready — select a .pswx workspace.")
        tk.Label(action, textvariable=self.z_status_var, bg=self.colors["bg"], fg=self.colors["muted"],
                 font=("Segoe UI", 9), anchor="e").pack(side="right", padx=4)

        # Live simulation status strip
        sim_status = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        sim_status.pack(fill="x", pady=(0, 10))
        top = tk.Frame(sim_status, bg=self.colors["panel"])
        top.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(top, text="SIMULATION STATUS", bg=self.colors["panel"], fg=self.colors["accent"],
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(top, textvariable=self.sim_stage_var, bg=self.colors["panel"], fg=self.colors["text"],
                 font=("Segoe UI", 9, "bold")).pack(side="right")
        tk.Label(sim_status, textvariable=self.sim_detail_var, bg=self.colors["panel"], fg=self.colors["muted"],
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(0, 6))
        self.sim_progress = ttk.Progressbar(sim_status, variable=self.sim_progress_var, maximum=100, mode="determinate")
        self.sim_progress.pack(fill="x", padx=14, pady=(0, 10))

        # Results area
        results = tk.PanedWindow(parent, orient="horizontal", sashwidth=6, bg=self.colors["bg"],
                                 bd=0, relief="flat")
        results.pack(fill="both", expand=True, pady=(0, 0))

        left = tk.Frame(results, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        right = tk.Frame(results, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        results.add(left, stretch="always", minsize=560)
        results.add(right, stretch="always", minsize=300)

        plot_header = tk.Frame(left, bg=self.colors["panel"])
        plot_header.pack(fill="x")
        tk.Label(plot_header, text="COMBINED NYQUIST PLOT", bg=self.colors["panel"], fg=self.colors["text"],
                 font=("Segoe UI Semibold", 10, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(plot_header, text="Loop gain eigenvalue branches", bg=self.colors["panel"], fg=self.colors["muted"],
                 font=("Segoe UI", 8)).pack(side="right", padx=16)

        self.z_figure, self.z_ax = plt.subplots(figsize=(8.5, 5.8), facecolor="#ffffff")
        self.z_ax.set_facecolor("#ffffff")
        self.z_ax.set_xlabel("Real", color="#344054")
        self.z_ax.set_ylabel("Imaginary", color="#344054")
        self.z_ax.set_title("Waiting for Z-tool results", color="#17202d", pad=12, fontsize=11, fontweight="bold")
        self.z_ax.tick_params(colors="#667085")
        for spine in self.z_ax.spines.values():
            spine.set_color("#d0d5dd")
        self.z_ax.grid(True, alpha=0.25, color="#98a2b3")
        self.z_canvas = FigureCanvasTkAgg(self.z_figure, master=left)
        self.z_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 4))
        toolbar = NavigationToolbar2Tk(self.z_canvas, left, pack_toolbar=False)
        toolbar.update()
        toolbar.configure(background=self.colors["panel"])
        toolbar.pack(fill="x", padx=8, pady=(0, 8))

        summary_header = tk.Frame(right, bg=self.colors["panel"])
        summary_header.pack(fill="x")
        tk.Label(summary_header, text="STABILITY SUMMARY", bg=self.colors["panel"], fg=self.colors["text"],
                 font=("Segoe UI Semibold", 10, "bold")).pack(side="left", padx=16, pady=12)
        self.z_result_badge = tk.Label(summary_header, text="NO RESULT", bg=self.colors["panel2"],
                                       fg=self.colors["muted"], font=("Segoe UI", 8, "bold"), padx=9, pady=5)
        self.z_result_badge.pack(side="right", padx=12)

        self.z_summary = tk.Text(right, font=("Consolas", 9), bg="#0b1322", fg="#cbd5e1",
                                 insertbackground="#ffffff", relief="flat", bd=0, padx=14, pady=12, wrap="word")
        self.z_summary.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.z_summary.insert("1.0", "No Z-tool result yet.\n\nSelect a .pswx workspace, check it, then run the frequency scan.")
        self.z_summary.config(state="disabled")

        # VS Code-style simulation terminal
        terminal_frame = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        terminal_frame.pack(fill="x", pady=(10, 0))
        terminal_header = tk.Frame(terminal_frame, bg=self.colors["panel"])
        terminal_header.pack(fill="x")
        tk.Label(terminal_header, text="SIMULATION TERMINAL", bg=self.colors["panel"], fg=self.colors["text"],
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=12, pady=8)
        tk.Label(terminal_header, text="Live Z-tool / PSCAD output", bg=self.colors["panel"], fg=self.colors["muted"],
                 font=("Segoe UI", 8)).pack(side="right", padx=12)
        terminal_body = tk.Frame(terminal_frame, bg=self.colors["terminal_bg"])
        terminal_body.pack(fill="x", padx=8, pady=(0, 8))
        self.z_log = tk.Text(terminal_body, height=9, font=("Cascadia Mono", 9), bg=self.colors["terminal_bg"],
                             fg=self.colors["terminal_text"], insertbackground="#ffffff", relief="flat", bd=0,
                             padx=12, pady=10, wrap="none")
        self.z_log.pack(side="left", fill="both", expand=True)
        terminal_scroll = ttk.Scrollbar(terminal_body, orient="vertical", command=self.z_log.yview)
        terminal_scroll.pack(side="right", fill="y")
        self.z_log.configure(yscrollcommand=terminal_scroll.set)

    # ========================================================
    # Z-TOOL INPUT / STATUS
    # ========================================================

    def browse_ztool_workspace(self):
        filename = filedialog.askopenfilename(
            title="Select Z-tool workspace",
            filetypes=[
                ("PSCAD Workspace (*.pswx)", "*.pswx"),
                ("All Files", "*.*"),
            ],
        )
        if not filename:
            return

        self.ztool_workspace = Path(filename).resolve()
        self.ztool_workspace_var.set(str(self.ztool_workspace))
        self._update_simulation_stage("WORKSPACE SELECTED", "Ready to check the selected .pswx workspace.", 0)
        self.ztool_detected_project_var.set("Detected project: not checked")
        self._set_ztool_status("Workspace selected. Click CHECK WORKSPACE.")
        self._zlog(f"Workspace: {self.ztool_workspace}")

    def _set_ztool_status(self, text):
        def update():
            self.z_status_var.set(text)
            if any(word in text for word in ("Running", "Opening", "Loading", "Checking", "Injection", "sweep")):
                self.header_status.configure(text="●  RUNNING", fg=self.colors["amber"])
        self.root.after(0, update)

    def _update_simulation_stage(self, stage, detail, progress=None):
        def update():
            self.sim_stage_var.set(stage)
            self.sim_detail_var.set(detail)
            if progress is not None:
                self.sim_progress_var.set(max(0.0, min(100.0, float(progress))))
        self.root.after(0, update)

    def _classify_simulation_line(self, text):
        lower = text.lower()
        if any(k in lower for k in ("d-axis", "d axis", "selector=1", "selector = 1", "_d.out")):
            self._update_simulation_stage("D-AXIS INJECTION", text.strip(), 35)
        elif any(k in lower for k in ("q-axis", "q axis", "selector=2", "selector = 2", "_q.out")):
            self._update_simulation_stage("Q-AXIS INJECTION", text.strip(), 65)
        elif "snapshot" in lower or "power flow" in lower or "powerflow" in lower:
            self._update_simulation_stage("SNAPSHOT / POWER FLOW", text.strip(), 20)
        elif "fft" in lower:
            self._update_simulation_stage("FFT PROCESSING", text.strip(), 78)
        elif "admittance" in lower or "compute y" in lower:
            self._update_simulation_stage("ADMITTANCE CALCULATION", text.strip(), 88)
        elif "nyquist" in lower or "stability" in lower:
            self._update_simulation_stage("STABILITY ANALYSIS", text.strip(), 95)

    def _zlog(self, text):
        self._classify_simulation_line(text)
        def update():
            self.z_log.insert("end", text + "\n")
            self.z_log.see("end")
        self.root.after(0, update)

    def _set_z_summary(self, text):
        def update():
            self.z_summary.config(state="normal")
            self.z_summary.delete("1.0", "end")
            self.z_summary.insert("1.0", text)
            self.z_summary.config(state="disabled")
        self.root.after(0, update)

    def _validate_scan_settings(self):
        f_min = float(self.z_fmin.get())
        f_max = float(self.z_fmax.get())
        f_points = int(self.z_fpoints.get())
        f_base = float(self.z_fbase.get())
        t_snap = float(self.z_tsnap.get())
        start_fft = float(self.z_start_fft.get())
        fft_periods = int(self.z_fft_periods.get())
        dtinj = float(self.z_dtinj.get())
        t_step = float(self.z_tstep.get())
        perturb = float(self.z_perturb.get())

        if f_min <= 0 or f_max <= f_min:
            raise ValueError("Require 0 < f min < f max.")
        if f_points < 2:
            raise ValueError("Frequency points must be >= 2.")
        if f_base <= 0:
            raise ValueError("f base must be positive.")
        if t_snap <= 0 or start_fft <= 0:
            raise ValueError("Snapshot and start FFT must be positive.")
        if fft_periods < 1:
            raise ValueError("FFT periods must be >= 1.")
        if dtinj < 0:
            raise ValueError("Injection settling time cannot be negative.")
        if t_step <= 0:
            raise ValueError("Time step must be positive.")
        if perturb <= 0:
            raise ValueError("Perturbation must be positive.")

        return (
            f_min, f_max, f_points, f_base,
            t_snap, start_fft, fft_periods, dtinj, t_step, perturb
        )

    # ========================================================
    # Z-TOOL WORKSPACE CHECK
    # ========================================================

    def start_workspace_check(self):
        if self.ztool_workspace is None:
            messagebox.showwarning(
                "No Workspace",
                "Select a .pswx workspace first.",
            )
            return

        self.z_check_button.config(state="disabled")
        self.z_run_button.config(state="disabled")

        threading.Thread(
            target=self._workspace_check_worker,
            daemon=True,
        ).start()

    def _workspace_check_worker(self):
        pscad = None
        try:
            self._update_simulation_stage("OPENING WORKSPACE", "Launching PSCAD and loading the selected .pswx workspace...", 5)
            self._set_ztool_status("Opening workspace in PSCAD...")
            self._zlog("=" * 72)
            self._zlog("WORKSPACE CHECK")
            self._zlog("=" * 72)

            pscad = mhi.pscad.launch(
                minimize=False,
                silence=False,
                x64=True,
            )
            time.sleep(4)
            pscad.load(str(self.ztool_workspace))

            self._zlog(f"Loaded workspace: {self.ztool_workspace}")

            project_name, _ = find_ac_scan_project(pscad)
            self.ztool_project_name = project_name

            def update_project(n=project_name):
                self.ztool_detected_project_var.set(f"{n}  ✓")
                self.ztool_project_badge.configure(fg=self.colors["green"])
            self.root.after(0, update_project)

            self._zlog(f"AC Scan project: {project_name}")
            self._set_ztool_status(f"Workspace valid — AC Scan project: {project_name}")

        except Exception as exc:
            self._set_ztool_status(f"Workspace check failed: {exc}")
            self.root.after(0, lambda: self.header_status.configure(text="●  ERROR", fg=self.colors["red"]))
            self._zlog("ERROR: " + repr(exc))
            self.root.after(
                0,
                lambda exc=exc: messagebox.showerror(
                    "Z-Tool Workspace Error",
                    str(exc),
                ),
            )
        finally:
            safe_close_pscad(pscad)
            self.root.after(
                0,
                lambda: (
                    self.z_check_button.config(state="normal"),
                    self.z_run_button.config(state="normal"),
                ),
            )

    # ========================================================
    # RUN Z-TOOL
    # ========================================================

    def start_ztool_scan(self):
        if self.ztool_workspace is None:
            messagebox.showwarning(
                "No Workspace",
                "Select the sample.pswx workspace first.",
            )
            return

        try:
            settings = self._validate_scan_settings()
        except Exception as exc:
            messagebox.showerror("Invalid Scan Settings", str(exc))
            return

        if self.ztool_running:
            return

        self.ztool_running = True
        self.z_run_button.config(state="disabled")
        self.z_check_button.config(state="disabled")
        self.z_open_results_button.config(state="disabled")

        threading.Thread(
            target=self._run_ztool_worker,
            args=(settings,),
            daemon=True,
        ).start()

    def _run_ztool_worker(self, settings):
        try:
            self._run_ztool(settings)
        except Exception as exc:
            self._zlog(traceback.format_exc())
            self._set_ztool_status(f"Z-tool ERROR: {exc}")
            self.root.after(0, lambda: self.header_status.configure(text="●  ERROR", fg=self.colors["red"]))
            self.root.after(
                0,
                lambda exc=exc: messagebox.showerror(
                    "Z-Tool Error",
                    str(exc),
                ),
            )
        finally:
            self.ztool_running = False
            self.root.after(
                0,
                lambda: self.z_run_button.config(state="normal"),
            )
            self.root.after(
                0,
                lambda: self.z_check_button.config(state="normal"),
            )

    def _run_ztool(self, settings):
        (
            f_min, f_max, f_points, f_base,
            t_snap, start_fft, fft_periods, dtinj, t_step, perturb
        ) = settings

        if self.ztool_workspace is None:
            raise RuntimeError("Z-tool workspace is not selected.")

        # ----------------------------------------------------
        # Verify workspace and identify the AC-scan project.
        # This session is separate from the waveform analyzer.
        # ----------------------------------------------------
        self._set_ztool_status("Checking the selected Z-tool workspace...")
        self._zlog("=" * 72)
        self._zlog("Z-TOOL FREQUENCY SCAN")
        self._zlog("=" * 72)

        pscad = None
        try:
            pscad = mhi.pscad.launch(
                minimize=False,
                silence=False,
                x64=True,
            )
            time.sleep(4)
            pscad.load(str(self.ztool_workspace))

            project_name, _ = find_ac_scan_project(pscad)
            self.ztool_project_name = project_name

            def update_project(n=project_name):
                self.ztool_detected_project_var.set(f"{n}  ✓")
                self.ztool_project_badge.configure(fg=self.colors["green"])
            self.root.after(0, update_project)

            self._zlog(f"Workspace: {self.ztool_workspace}")
            self._zlog(f"Project:   {project_name}")
            self._zlog("AC Scan:   FOUND")
        finally:
            safe_close_pscad(pscad)

        # ----------------------------------------------------
        # Import Z-tool modules only when needed.
        # ----------------------------------------------------
        self._set_ztool_status("Loading Z-tool modules...")

        sys.path.insert(0, str(SOURCE_ROOT))
        importlib.invalidate_caches()

        from Source.ztoolacdc import create_freq
        from Source.ztoolacdc import frequency_sweep
        from Source.ztoolacdc import read_admittance
        from Source.ztoolacdc import stability

        create_freq = importlib.reload(create_freq)
        frequency_sweep = importlib.reload(frequency_sweep)
        read_admittance = importlib.reload(read_admittance)
        stability = importlib.reload(stability)

        # ----------------------------------------------------
        # Per-run results directory.
        # ----------------------------------------------------
        run_name = timestamp_folder_name(self.ztool_project_name)
        results_root = self.ztool_workspace.parent / "ZTool_Results"
        results_root.mkdir(parents=True, exist_ok=True)

        results_folder = results_root / run_name
        results_folder.mkdir(parents=True, exist_ok=True)

        self.ztool_results_folder = results_folder

        output_files = "ZTool_scan"

        # IMPORTANT:
        # frequency_sweep.py constructs its raw-output directory from:
        #     working_dir + project_name + fortran_ext
        # Its default is .gf46, but this workspace/project may use another
        # compiler directory such as .gf132. Detect the real one before
        # starting the expensive sweep.
        fortran_ext = detect_fortran_extension(
            self.ztool_workspace.parent,
            self.ztool_project_name,
        )

        compiler_dir = (
            self.ztool_workspace.parent
            / f"{self.ztool_project_name}{fortran_ext}"
        )

        self._zlog(f"Results folder: {results_folder}")
        self._zlog(f"Detected PSCAD compiler/output folder: {compiler_dir}")
        self._zlog(f"Detected Fortran extension: {fortran_ext}")

        if not compiler_dir.exists():
            self._zlog(
                "WARNING: compiler folder does not exist before the sweep. "
                "The selected extension will still be passed to Z-tool."
            )

        freq = create_freq.loglist(
            f_min=f_min,
            f_max=f_max,
            f_points=f_points,
            f_base=f_base,
        )

        self._zlog(
            f"Scan: {f_min:g}–{f_max:g} Hz | "
            f"{len(freq)} frequency points"
        )

        # ----------------------------------------------------
        # Execute the existing Z-tool frequency sweep.
        # The sweep owns the actual PSCAD automation.
        # ----------------------------------------------------
        self._update_simulation_stage(
            "D-AXIS INJECTION",
            "Starting PSCAD frequency sweep. Z-tool performs d-axis injection first, followed by q-axis injection.",
            30,
        )
        self._set_ztool_status(
            "Running PSCAD Z-tool frequency sweep — d-axis and q-axis injections in progress..."
        )
        self._zlog("[PHASE 1/2] D-axis injection / frequency sweep starting...")

        capture = io.StringIO()

        class Tee(io.TextIOBase):
            def write(inner_self, s):
                if s:
                    for line in s.splitlines():
                        if line.strip():
                            self._zlog(line)
                capture.write(s)
                return len(s)

            def flush(inner_self):
                capture.flush()

        tee = Tee()

        with redirect_stdout(tee), redirect_stderr(tee):
            frequency_sweep.frequency_sweep(
                t_snap=t_snap,
                t_sim=start_fft + fft_periods / f_base,
                t_step=t_step,
                dt_injections=dtinj,
                f_base=f_base,
                freq=freq,
                start_fft=start_fft,
                fft_periods=fft_periods,
                v_perturb_mag=perturb,
                working_dir=str(self.ztool_workspace.parent) + os.sep,
                workspace_name=self.ztool_workspace.stem,
                project_name=self.ztool_project_name,
                fortran_ext=fortran_ext,
                results_folder=str(results_folder),
                output_files=output_files,
                show_powerflow=True,
                topology=None,
                compute_yz=True,
                save_td=False,
                run_sim=True,
                verbose=False,
            )

        self._zlog("[PHASE 2/2] Q-axis injection / frequency sweep completed.")
        self._zlog("D-axis + Q-axis frequency sweep completed.")
        self._update_simulation_stage("FREQUENCY SWEEP COMPLETE", "D-axis and q-axis injections completed. Reading admittance matrices...", 82)
        self._set_ztool_status("Sweep complete. Reading admittance matrices...")

        # ----------------------------------------------------
        # Read VSC and grid admittance results.
        # ----------------------------------------------------
        y_vsc = read_admittance.read_admittance(
            path=str(results_folder),
            involved_blocks="PCC-1",
            file_root=output_files,
        )

        y_grid = read_admittance.read_admittance(
            path=str(results_folder),
            involved_blocks="PCC-2",
            file_root=output_files,
        )

        if len(y_vsc.f) != len(y_grid.f):
            raise RuntimeError(
                "VSC and grid frequency vectors have different lengths: "
                f"{len(y_vsc.f)} and {len(y_grid.f)}"
            )

        # ----------------------------------------------------
        # Loop gain, using the same expression as the user's
        # successful Single_bus_analysis.py:
        # L = inv(Y_grid) @ Y_VSC
        # ----------------------------------------------------
        self._update_simulation_stage("ADMITTANCE CALCULATION", "Building VSC/grid loop gain matrix from PCC-1 and PCC-2 results...", 88)
        loop_gain = np.matmul(
            np.linalg.inv(y_grid.y),
            y_vsc.y,
        )

        eig_loop = track_eigenvalues(loop_gain)

        # ----------------------------------------------------
        # GNC / Nyquist encirclement count.
        # Positive winding = CCW, negative = CW.
        # ----------------------------------------------------
        winding_1 = winding_number(eig_loop[:, 0])
        winding_2 = winding_number(eig_loop[:, 1])
        net_winding = winding_1 + winding_2

        ccw = max(net_winding, 0)
        cw = max(-net_winding, 0)
        stable = (net_winding == 0)

        # Ask the Z-tool itself to generate its official GNC files.
        try:
            tool_stable = stability.nyquist(
                loop_gain,
                y_vsc.f,
                results_folder=str(results_folder),
                filename="PSCAD_case",
                verbose=False,
            )
        except Exception:
            tool_stable = stable

        # ----------------------------------------------------
        # Dominant closed-loop mode using the same closed-loop
        # matrix as the user's successful script.
        # ----------------------------------------------------
        closed_loop = np.linalg.inv(
            y_grid.y + y_vsc.y
        )

        dominant_frequency = None
        dominant_magnitude = None
        dominant_mode = None
        max_mag = -np.inf

        for k, g in enumerate(closed_loop):
            eig_g = np.linalg.eigvals(g)
            idx = int(np.argmax(np.abs(eig_g)))
            mag = float(abs(eig_g[idx]))

            if mag > max_mag:
                max_mag = mag
                dominant_frequency = float(y_vsc.f[k])
                dominant_magnitude = mag
                dominant_mode = idx + 1

        # ----------------------------------------------------
        # Grid inductance estimate at 50 Hz, matching the
        # user's compensation calculation.
        # ----------------------------------------------------
        z_grid = np.linalg.inv(y_grid.y)
        idx50 = int(np.argmin(np.abs(y_grid.f - 50.0)))
        xg = float(np.real(z_grid[idx50, 1, 0]))
        grid_inductance = xg / (2.0 * np.pi * 50.0)

        # ----------------------------------------------------
        # Combined Nyquist plot.
        # ----------------------------------------------------
        self.z_figure.clear()
        self.z_figure.patch.set_facecolor("#ffffff")
        ax = self.z_figure.add_subplot(111)
        ax.set_facecolor("#ffffff")

        ax.plot(
            np.real(eig_loop[:, 0]),
            np.imag(eig_loop[:, 0]),
            linewidth=1.6,
            label=r"$\lambda_1(L)$",
        )
        ax.plot(
            np.real(eig_loop[:, 1]),
            np.imag(eig_loop[:, 1]),
            linewidth=1.6,
            label=r"$\lambda_2(L)$",
        )
        ax.plot(
            -1.0,
            0.0,
            marker="x",
            markersize=10,
            mew=2,
            label="Critical point (-1, 0)",
        )

        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.0, linewidth=0.8)
        ax.set_xlabel("Real", color="#344054")
        ax.set_ylabel("Imaginary", color="#344054")
        ax.set_title("Combined Nyquist Plot — Z-tool Loop Gain", color="#17202d", pad=12, fontsize=11, fontweight="bold")
        ax.tick_params(colors="#667085")
        for spine in ax.spines.values():
            spine.set_color("#d0d5dd")
        ax.grid(True, alpha=0.25, color="#98a2b3")
        legend = ax.legend(loc="best", facecolor="#ffffff", edgecolor="#d0d5dd")
        for txt in legend.get_texts():
            txt.set_color("#344054")

        self.z_figure.tight_layout()

        nyquist_png = results_folder / "Combined_Nyquist.png"
        self.z_figure.savefig(
            nyquist_png,
            dpi=250,
            bbox_inches="tight",
        )

        self.root.after(0, self.z_canvas.draw_idle)

        self._update_simulation_stage("STABILITY ANALYSIS", "Computing Nyquist encirclements and dominant closed-loop mode...", 96)

        # ----------------------------------------------------
        # Summary.
        # ----------------------------------------------------
        overall_status = "STABLE" if stable else "UNSTABLE"
        official_status = (
            "STABLE" if bool(tool_stable) else "UNSTABLE"
        )

        summary = (
            "Z-TOOL STABILITY SUMMARY\n"
            "========================\n\n"
            f"Status:                 {overall_status}\n"
            f"Z-tool GNC status:      {official_status}\n\n"
            f"CCW encirclements:      {ccw}\n"
            f"CW encirclements:       {cw}\n"
            f"Net encirclements:      {net_winding:+d}\n"
            f"Critical point:         (-1, 0)\n\n"
            "DOMINANT MODE\n"
            "------------------------\n"
            f"Frequency:              {dominant_frequency:.6g} Hz\n"
            f"Eigenvalue mode:        lambda_{dominant_mode}\n"
            f"Eigenvalue magnitude:   {dominant_magnitude:.8g}\n\n"
            "GRID\n"
            "------------------------\n"
            f"Xg @ 50 Hz:             {xg:.8g} ohm\n"
            f"Grid inductance:        {grid_inductance:.8g} H\n\n"
            "SCAN\n"
            "------------------------\n"
            f"Frequency points:       {len(y_vsc.f)}\n"
            f"Frequency range:        {y_vsc.f[0]:.6g}–{y_vsc.f[-1]:.6g} Hz\n"
            f"Perturbation:           {perturb:g} pu\n\n"
            "PSCAD RAW OUTPUT\n"
            "------------------------\n"
            f"{compiler_dir}\n"
            f"Fortran extension:      {fortran_ext}\n\n"
            "RESULT FILES\n"
            "------------------------\n"
            f"{results_folder}\n\n"
            f"Combined Nyquist:\n{nyquist_png}\n"
        )

        self._set_z_summary(summary)
        self.root.after(0, lambda: self.z_result_badge.configure(
            text=overall_status, fg=self.colors["green"] if stable else self.colors["red"]))
        self.root.after(
            0,
            lambda: self.z_open_results_button.config(state="normal"),
        )

        self._set_ztool_status(f"Z-tool complete — results saved in {results_folder}")
        self._update_simulation_stage("COMPLETE", f"Z-tool analysis finished — {overall_status}. Results saved successfully.", 100)
        self.root.after(0, lambda: self.header_status.configure(text=f"●  {overall_status}", fg=self.colors["green"] if stable else self.colors["red"]))

        self._zlog("=" * 72)
        self._zlog("Z-TOOL ANALYSIS COMPLETE")
        self._zlog("=" * 72)
        self._zlog(f"GNC status: {overall_status}")
        self._zlog(f"CCW encirclements: {ccw}")
        self._zlog(f"CW encirclements: {cw}")
        self._zlog(f"Net encirclements: {net_winding:+d}")
        self._zlog(
            f"Dominant mode: {dominant_frequency:.6g} Hz, "
            f"lambda_{dominant_mode}"
        )
        self._zlog(f"Grid inductance: {grid_inductance:.8g} H")
        self._zlog(f"Results: {results_folder}")

        self.last_ztool_data = {
            "frequency": y_vsc.f,
            "Y_VSC": y_vsc.y,
            "Y_grid": y_grid.y,
            "L": loop_gain,
            "eigenvalues": eig_loop,
            "stable": stable,
            "cw": cw,
            "ccw": ccw,
            "net": net_winding,
            "dominant_frequency": dominant_frequency,
            "dominant_mode": dominant_mode,
            "dominant_magnitude": dominant_magnitude,
            "grid_inductance": grid_inductance,
            "results_folder": results_folder,
        }

    # ========================================================
    # OPEN RESULTS
    # ========================================================

    def open_ztool_results_folder(self):
        if self.ztool_results_folder is None:
            return
        try:
            os.startfile(str(self.ztool_results_folder))
        except Exception as exc:
            messagebox.showerror(
                "Open Results Folder Error",
                str(exc),
            )

    # ========================================================
    # CLOSE
    # ========================================================

    def on_close(self):
        safe_close_pscad(None)

        if self.waveform_process is not None:
            try:
                if self.waveform_process.poll() is None:
                    self.waveform_process.terminate()
            except Exception:
                pass
            self.waveform_process = None
            self.waveform_started_at = 0.0

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = MasterPSCADGUI(root)
    root.mainloop()
