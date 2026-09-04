import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import re
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import sys
from pathlib import Path as _Path

# Local MHI package used by this project.
# The original analyzer is kept otherwise unchanged.
_MHI_ROOT = _Path(r"C:\Users\Asus\TwoLevelVSC_GUI\Z-tool-main\mhi")
if _MHI_ROOT.exists():
    sys.path.insert(0, str(_MHI_ROOT))

import mhi.pscad


class PSCADWaveformAnalyzer:

    def __init__(self, root):

        self.root = root

        self.root.title("PSCAD Waveform Analyzer")
        self.root.geometry("1350x850")

        # =====================================================
        # PSCAD VARIABLES
        # =====================================================

        self.pscad = None
        self.project = None

        self.project_file = None
        self.project_name = None

        # =====================================================
        # OUTPUT FILE VARIABLES
        # =====================================================

        self.out_file = None
        self.inf_file = None

        self.data = None
        self.time_data = None

        # Dictionary:
        #
        # channel number -> variable name
        #
        # Example:
        #
        # 1 -> v_abc:1
        # 2 -> v_abc:2
        #
        self.channels = {}

        # Plot / variable-selection state
        self.filtered_channels = {}
        self.selected_channels = []

        # =====================================================
        # FILE INFORMATION BEFORE SIMULATION
        # =====================================================

        self.out_files_before = {}
        self.inf_files_before = {}

        # =====================================================
        # CREATE GUI
        # =====================================================

        self.create_gui()

    # =========================================================
    # GUI
    # =========================================================

    def create_gui(self):
        # -----------------------------------------------------
        # Window / theme
        # -----------------------------------------------------
        self.root.configure(bg="#f4f6f8")
        self.root.minsize(1150, 760)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#f4f6f8")
        style.configure("Card.TLabelframe", background="#ffffff")
        style.configure("Card.TLabelframe.Label", background="#ffffff",
                        font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#f4f6f8", font=("Segoe UI", 9))
        style.configure("Card.TLabel", background="#ffffff", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#f4f6f8",
                        font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background="#f4f6f8",
                        font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------
        header = ttk.Frame(self.root, padding=(20, 14, 20, 7))
        header.pack(fill="x")
        ttk.Label(header, text="PSCAD Simulation Analyzer",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(header,
                  text="Python–PSCAD simulation, output and waveform analysis",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        # -----------------------------------------------------
        # Project
        # -----------------------------------------------------
        project_frame = ttk.LabelFrame(
            self.root, text="  PROJECT  ", style="Card.TLabelframe", padding=10
        )
        project_frame.pack(fill="x", padx=20, pady=(4, 7))
        project_row = ttk.Frame(project_frame, style="Card.TLabelframe")
        project_row.pack(fill="x")
        self.project_path_var = tk.StringVar()
        self.project_entry = ttk.Entry(
            project_row, textvariable=self.project_path_var, state="readonly"
        )
        self.project_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(project_row, text="Browse .pscx",
                   command=self.browse_project).pack(side="right")
        self.project_info_label = ttk.Label(
            project_frame, text="No project selected.", style="Card.TLabel"
        )
        self.project_info_label.pack(anchor="w", pady=(6, 0))

        # -----------------------------------------------------
        # Simulation + output
        # -----------------------------------------------------
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=20, pady=(0, 7))

        simulation_frame = ttk.LabelFrame(
            top, text="  SIMULATION  ", style="Card.TLabelframe", padding=10
        )
        simulation_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        button_row = ttk.Frame(simulation_frame, style="Card.TLabelframe")
        button_row.pack(fill="x")
        self.run_button = ttk.Button(
            button_row, text="▶  RUN SIMULATION", style="Accent.TButton",
            command=self.start_simulation
        )
        self.run_button.pack(side="left")
        self.close_pscad_button = ttk.Button(
            button_row, text="Close PSCAD", command=self.close_pscad, state="disabled"
        )
        self.close_pscad_button.pack(side="left", padx=(10, 0))
        self.status_var = tk.StringVar(value="Ready — select a PSCAD project to begin.")
        ttk.Label(simulation_frame, textvariable=self.status_var,
                  style="Card.TLabel").pack(anchor="w", pady=(9, 4))
        self.progress = ttk.Progressbar(simulation_frame, mode="indeterminate")
        self.progress.pack(fill="x")

        output_frame = ttk.LabelFrame(
            top, text="  SIMULATION OUTPUT  ", style="Card.TLabelframe", padding=10
        )
        output_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.out_label = ttk.Label(output_frame, text="OUT file: Not loaded.",
                                   style="Card.TLabel")
        self.out_label.pack(anchor="w", pady=1)
        self.inf_label = ttk.Label(output_frame, text="INF file: Not loaded.",
                                   style="Card.TLabel")
        self.inf_label.pack(anchor="w", pady=1)
        self.channel_count_label = ttk.Label(output_frame, text="Output variables: 0",
                                             style="Card.TLabel")
        self.channel_count_label.pack(anchor="w", pady=(5, 0))

        # -----------------------------------------------------
        # Main analysis area
        # -----------------------------------------------------
        analysis = ttk.Frame(self.root)
        analysis.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        # Variable selection panel
        variable_frame = ttk.LabelFrame(
            analysis, text="  OUTPUT VARIABLES  ", style="Card.TLabelframe", padding=8
        )
        variable_frame.pack(side="left", fill="y", padx=(0, 6))

        search_row = ttk.Frame(variable_frame, style="Card.TLabelframe")
        search_row.pack(fill="x", pady=(0, 6))
        ttk.Label(search_row, text="Search:", style="Card.TLabel").pack(side="left")
        self.variable_search_var = tk.StringVar()
        self.variable_search_var.trace_add("write", self.filter_variables)
        ttk.Entry(search_row, textvariable=self.variable_search_var,
                  width=25).pack(side="left", padx=(6, 0))

        list_frame = ttk.Frame(variable_frame, style="Card.TLabelframe")
        list_frame.pack(fill="both", expand=True)
        self.variable_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False,
            width=38, height=12, font=("Segoe UI", 9),
            relief="solid", borderwidth=1
        )
        self.variable_listbox.pack(side="left", fill="both", expand=True)
        variable_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.variable_listbox.yview
        )
        variable_scroll.pack(side="right", fill="y")
        self.variable_listbox.config(yscrollcommand=variable_scroll.set)

        select_row = ttk.Frame(variable_frame, style="Card.TLabelframe")
        select_row.pack(fill="x", pady=(7, 0))
        ttk.Button(select_row, text="Select All",
                   command=self.select_all_variables).pack(side="left")
        ttk.Button(select_row, text="Clear Selection",
                   command=self.clear_variable_selection).pack(side="left", padx=5)
        ttk.Button(select_row, text="Plot Selected",
                   command=self.plot_selected).pack(side="left")

        # Plot panel
        plot_area = ttk.Frame(analysis, style="Card.TLabelframe")
        plot_area.pack(side="right", fill="both", expand=True)

        plot_frame = ttk.LabelFrame(
            plot_area, text="  WAVEFORM  ", style="Card.TLabelframe", padding=4
        )
        plot_frame.pack(fill="both", expand=True)

        self.figure, self.ax = plt.subplots(figsize=(10, 5))
        self.figure.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.toolbar = toolbar

        # -----------------------------------------------------
        # Plot controls
        # -----------------------------------------------------
        controls = ttk.LabelFrame(
            self.root, text="  PLOT CONTROLS  ", style="Card.TLabelframe", padding=7
        )
        controls.pack(fill="x", padx=20, pady=(0, 7))

        self.grid_var = tk.BooleanVar(value=True)
        self.legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Grid", variable=self.grid_var,
                        command=self.apply_plot_controls).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(controls, text="Legend", variable=self.legend_var,
                        command=self.apply_plot_controls).pack(side="left", padx=(0, 15))

        ttk.Label(controls, text="X min:").pack(side="left")
        self.xmin_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.xmin_var, width=10).pack(side="left", padx=4)
        ttk.Label(controls, text="X max:").pack(side="left")
        self.xmax_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.xmax_var, width=10).pack(side="left", padx=4)
        ttk.Label(controls, text="Y min:").pack(side="left", padx=(8, 0))
        self.ymin_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.ymin_var, width=10).pack(side="left", padx=4)
        ttk.Label(controls, text="Y max:").pack(side="left")
        self.ymax_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.ymax_var, width=10).pack(side="left", padx=4)

        ttk.Button(controls, text="Apply Limits",
                   command=self.apply_plot_limits).pack(side="left", padx=5)
        ttk.Button(controls, text="Auto Scale",
                   command=self.auto_scale_plot).pack(side="left", padx=3)
        ttk.Button(controls, text="Clear Plot",
                   command=self.clear_plot).pack(side="left", padx=3)
        ttk.Button(controls, text="Save Plot",
                   command=self.save_plot).pack(side="right")

        # -----------------------------------------------------
        # Footer
        # -----------------------------------------------------
        self.footer_var = tk.StringVar(value="Ready")
        footer = ttk.Frame(self.root, padding=(20, 1, 20, 6))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.footer_var).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =========================================================
    # UPDATE FILE LABELS
    # =========================================================

    def update_output_labels(self):
        """Display the actual output files selected by the analyzer."""
        if self.out_file is not None:
            self.out_label.config(text=f"OUT file: {self.out_file}")
        if self.inf_file is not None:
            self.inf_label.config(text=f"INF file: {self.inf_file}")

    # =========================================================
    # BROWSE PSCX
    # =========================================================

    def browse_project(self):

        filename = filedialog.askopenfilename(
            title="Select PSCAD Project",
            filetypes=[
                ("PSCAD Project (*.pscx)", "*.pscx"),
                ("All Files", "*.*")
            ]
        )

        if not filename:
            return

        self.project_file = Path(
            filename
        ).resolve()

        # -----------------------------------------------------
        # Automatically determine project name
        # -----------------------------------------------------

        self.project_name = (
            self.project_file.stem
        )

        self.project_path_var.set(
            str(self.project_file)
        )

        self.project_info_label.config(
            text=(
                f"Project: {self.project_name}    |    "
                f"Directory: {self.project_file.parent}"
            )
        )

        self.out_file = None
        self.inf_file = None
        self.data = None
        self.time_data = None
        self.channels = {}

        self.out_label.config(
            text="OUT file: Not loaded."
        )
        self.inf_label.config(
            text="INF file: Not loaded."
        )
        self.channel_count_label.config(
            text="Output variables: 0"
        )
        if hasattr(self, "variable_search_var"):
            self.variable_search_var.set("")
        if hasattr(self, "variable_listbox"):
            self.variable_listbox.delete(0, tk.END)
        self.filtered_channels = {}

        self.status_var.set(
            "Project selected. Ready to run simulation."
        )

        print("\nSelected PSCAD project:")
        print(self.project_file)

        print("\nAutomatically detected project name:")
        print(self.project_name)

    # =========================================================
    # START SIMULATION
    # =========================================================

    def start_simulation(self):

        if self.project_file is None:

            messagebox.showwarning(
                "No PSCAD Project",
                "Please select a .pscx project first."
            )

            return

        # -----------------------------------------------------
        # Disable controls during simulation
        # -----------------------------------------------------

        self.run_button.config(
            state="disabled"
        )

        self.progress.start(
            10
        )

        # -----------------------------------------------------
        # Start simulation in background thread
        # -----------------------------------------------------

        thread = threading.Thread(
            target=self.run_simulation,
            daemon=True
        )

        thread.start()

    # =========================================================
    # THREAD-SAFE STATUS
    # =========================================================

    def set_status(self, text):

        self.root.after(
            0,
            lambda text=text: (
                self.status_var.set(text),
                self.footer_var.set(text)
            )
        )

    # =========================================================
    # FIND ALL OUTPUT FILES
    # =========================================================

    def find_all_output_files(self):

        project_directory = (
            self.project_file.parent
        )

        output_files = []

        # -----------------------------------------------------
        # Search recursively.
        #
        # This allows different PSCAD compiler folders:
        #
        # .gf132
        # .gf
        # etc.
        # -----------------------------------------------------

        try:

            output_files = list(
                project_directory.rglob("*.out")
            )

        except Exception as e:

            print(
                "Error searching OUT files:",
                e
            )

        return output_files

    # =========================================================
    # RECORD OUTPUT FILES BEFORE SIMULATION
    # =========================================================

    def record_output_files_before_simulation(self):

        self.out_files_before = {}

        files = self.find_all_output_files()

        for file in files:

            try:

                self.out_files_before[
                    str(file.resolve())
                ] = (
                    file.stat().st_mtime,
                    file.stat().st_size
                )

            except Exception:
                pass

        print(
            "\nOUT files before simulation:"
        )

        for file in self.out_files_before:

            print(
                " ",
                file
            )

    # =========================================================
    # FIND UPDATED OUTPUT FILE
    # =========================================================

    def find_updated_output_file(self):

        current_files = (
            self.find_all_output_files()
        )

        if not current_files:

            return None

        candidates = []

        # -----------------------------------------------------
        # Compare files with their previous state
        # -----------------------------------------------------

        for file in current_files:

            try:

                path = str(
                    file.resolve()
                )

                stat = file.stat()

                current_mtime = (
                    stat.st_mtime
                )

                current_size = (
                    stat.st_size
                )

                previous = (
                    self.out_files_before.get(path)
                )

                # -------------------------------------------------
                # New file
                # -------------------------------------------------

                if previous is None:

                    candidates.append(
                        file
                    )

                    continue

                previous_mtime, previous_size = (
                    previous
                )

                # -------------------------------------------------
                # Existing file was rewritten
                # -------------------------------------------------

                if (
                    current_mtime > previous_mtime
                    or
                    current_size != previous_size
                ):

                    candidates.append(
                        file
                    )

            except Exception:

                pass

        # -----------------------------------------------------
        # If changed/new files were found,
        # choose newest.
        # -----------------------------------------------------

        if candidates:

            return max(
                candidates,
                key=lambda f: f.stat().st_mtime
            )

        # -----------------------------------------------------
        # Fallback:
        #
        # choose newest OUT file.
        #
        # This is useful if the file system timestamp
        # resolution is unusual.
        # -----------------------------------------------------

        return max(
            current_files,
            key=lambda f: f.stat().st_mtime
        )

    # =========================================================
    # WAIT FOR OUTPUT FILE
    # =========================================================

    def wait_for_output_file(
        self,
        timeout=300,
        check_interval=1
    ):

        start_time = time.time()

        while (
            time.time() - start_time
            < timeout
        ):

            out_file = (
                self.find_updated_output_file()
            )

            if out_file is not None:

                # -------------------------------------------------
                # Make sure file is no longer being written.
                #
                # Check size twice.
                # -------------------------------------------------

                try:

                    size_1 = (
                        out_file.stat().st_size
                    )

                    time.sleep(
                        1
                    )

                    size_2 = (
                        out_file.stat().st_size
                    )

                    if size_1 == size_2:

                        return out_file

                except Exception:
                    pass

            time.sleep(
                check_interval
            )

        return None

    # =========================================================
    # PSCAD OUTPUT CONFIGURATION
    # =========================================================

    def enable_out_file(self):
        # Python 3.11 / current MHI compatibility:
        # do not call project.parameters(PlotType="OUT", ...).
        #
        # The original analyzer already detects newly-created/updated
        # .out files recursively, so the selected PSCAD project keeps
        # its existing output configuration.
        print("\nUsing the selected project's existing .out output configuration.")
        self.set_status(
            "Using the selected project's existing PSCAD output configuration."
        )

    # =========================================================
    # RUN PSCAD
    # =========================================================

    def run_simulation(self):

        try:

            print("\n")
            print("=" * 70)
            print("GENERIC PSCAD AUTOMATION")
            print("=" * 70)

            # -------------------------------------------------
            # STEP 1
            # Record existing output files
            # -------------------------------------------------

            self.set_status(
                "Recording existing PSCAD output files..."
            )

            self.record_output_files_before_simulation()

            # -------------------------------------------------
            # STEP 2
            # Launch PSCAD
            # -------------------------------------------------

            self.set_status(
                "Launching PSCAD..."
            )

            print(
                "\nStarting PSCAD..."
            )

            self.pscad = mhi.pscad.launch(
                minimize=False,
                silence=False,
                x64=True
            )

            print(
                "PSCAD launched successfully."
            )

            self.root.after(
                0,
                lambda: self.close_pscad_button.config(
                    state="normal"
                )
            )

            # -------------------------------------------------
            # STEP 3
            # Wait for PSCAD
            # -------------------------------------------------

            time.sleep(
                5
            )

            # -------------------------------------------------
            # STEP 4
            # Load selected project
            # -------------------------------------------------

            self.set_status(
                "Loading selected PSCAD project..."
            )

            print(
                "\nLoading project:"
            )

            print(
                self.project_file
            )

            self.pscad.load(
                str(self.project_file)
            )

            print(
                "Project loaded successfully."
            )

            # -------------------------------------------------
            # STEP 5
            # Automatically get project name
            # -------------------------------------------------

            self.project_name = (
                self.project_file.stem
            )

            print(
                "\nProject name:"
            )

            print(
                self.project_name
            )

            # -------------------------------------------------
            # STEP 6
            # Get project object
            # -------------------------------------------------

            self.set_status(
                f"Finding project: {self.project_name}"
            )

            print(
                "\nFinding project..."
            )

            self.project = (
                self.pscad.project(
                    self.project_name
                )
            )

            print(
                "Project found successfully."
            )

            print(
                "Project object:",
                self.project
            )

            # -------------------------------------------------
            # STEP 7
            # OUTPUT CONFIGURATION
            # -------------------------------------------------

            self.enable_out_file()

            # -------------------------------------------------
            # STEP 8
            # Run simulation
            # -------------------------------------------------

            self.set_status(
                "Running PSCAD simulation..."
            )

            print(
                "\nStarting simulation..."
            )

            self.project.run()

            print(
                "Simulation command sent."
            )

            # -------------------------------------------------
            # STEP 9
            #
            # Wait for OUT file.
            #
            # Maximum wait = 5 minutes.
            # -------------------------------------------------

            self.set_status(
                "Simulation running — waiting for output..."
            )

            print(
                "\nWaiting for updated OUT file..."
            )

            out_file = (
                self.wait_for_output_file(
                    timeout=300
                )
            )

            if out_file is None:

                raise RuntimeError(
                    "Simulation output .out file "
                    "was not detected within 5 minutes."
                )

            self.out_file = (
                out_file.resolve()
            )

            # Show the actual OUT file in the GUI.
            self.root.after(
                0,
                lambda path=str(self.out_file): self.out_label.config(
                    text=f"OUT file: {path}"
                )
            )

            print(
                "\nOUT file detected:"
            )

            print(
                self.out_file
            )

            # -------------------------------------------------
            # STEP 10
            # Find corresponding INF
            # -------------------------------------------------

            self.set_status(
                "Simulation complete. Finding INF file..."
            )

            self.find_inf_file()

            # Show the actual INF file in the GUI.
            self.root.after(
                0,
                lambda path=str(self.inf_file): self.inf_label.config(
                    text=f"INF file: {path}"
                )
            )

            print(
                "\nINF file detected:"
            )

            print(
                self.inf_file
            )

            # -------------------------------------------------
            # STEP 11
            # Read channels
            # -------------------------------------------------

            self.set_status(
                "Reading output variables..."
            )

            self.read_inf_file()

            # -------------------------------------------------
            # STEP 12
            # Read numerical data
            # -------------------------------------------------

            self.set_status(
                "Reading waveform data..."
            )

            self.read_out_file()

            # -------------------------------------------------
            # Finished
            # -------------------------------------------------

            number_of_channels = (
                len(self.channels)
            )

            self.set_status(
                f"Complete — "
                f"{number_of_channels} output variables detected."
            )

            self.root.after(
                0,
                self.progress.stop
            )

            self.root.after(
                0,
                lambda: self.run_button.config(
                    state="normal"
                )
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "SIMULATION + DATA IMPORT COMPLETE"
            )

            print(
                "=" * 70
            )

        except Exception as e:

            print(
                "\n"
                + "=" * 70
            )

            print(
                "ERROR"
            )

            print(
                "=" * 70
            )

            print(
                repr(e)
            )

            self.set_status(
                f"ERROR: {e}"
            )

            self.root.after(
                0,
                self.progress.stop
            )

            self.root.after(
                0,
                lambda: self.run_button.config(
                    state="normal"
                )
            )

            error_text = str(e)

            self.root.after(
                0,
                lambda error_text=error_text: messagebox.showerror(
                    "PSCAD Simulation Error",
                    error_text
                )
            )

    # =========================================================
    # FIND INF FILE
    # =========================================================

    def find_inf_file(self):

        if self.out_file is None:

            raise RuntimeError(
                "OUT file is not available."
            )

        directory = (
            self.out_file.parent
        )

        out_stem = (
            self.out_file.stem
        )

        # -----------------------------------------------------
        # First try exact same name
        #
        # Model_01.out
        # Model_01.inf
        # -----------------------------------------------------

        exact = (
            directory /
            f"{out_stem}.inf"
        )

        if exact.exists():

            self.inf_file = (
                exact.resolve()
            )

            return

        # -----------------------------------------------------
        # Remove trailing output number
        #
        # Model_01
        # becomes
        # Model
        # -----------------------------------------------------

        base_stem = re.sub(
            r"_\d+$",
            "",
            out_stem
        )

        candidate = (
            directory /
            f"{base_stem}.inf"
        )

        if candidate.exists():

            self.inf_file = (
                candidate.resolve()
            )

            return

        # -----------------------------------------------------
        # Search all INF files in same directory
        # -----------------------------------------------------

        inf_files = list(
            directory.glob("*.inf")
        )

        if not inf_files:

            # Search recursively as fallback
            inf_files = list(
                self.project_file.parent.rglob("*.inf")
            )

        if not inf_files:

            raise FileNotFoundError(
                "No .inf file could be found "
                "for the generated .out file."
            )

        # -----------------------------------------------------
        # Prefer one with project name
        # -----------------------------------------------------

        project_lower = (
            self.project_name.lower()
        )

        for inf in inf_files:

            if (
                project_lower
                in inf.stem.lower()
            ):

                self.inf_file = (
                    inf.resolve()
                )

                return

        # -----------------------------------------------------
        # Final fallback
        # -----------------------------------------------------

        self.inf_file = (
            max(
                inf_files,
                key=lambda f: f.stat().st_mtime
            ).resolve()
        )

    # =========================================================
    # READ INF FILE
    # =========================================================

    def read_inf_file(self):

        self.channels = {}

        with open(
            self.inf_file,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:

            lines = file.readlines()

        # -----------------------------------------------------
        # Search for PGB entries
        # -----------------------------------------------------

        for line in lines:

            line = line.strip()

            if not line.startswith(
                "PGB("
            ):
                continue

            # -------------------------------------------------
            # Channel number
            # -------------------------------------------------

            number_match = re.search(
                r"PGB\((\d+)\)",
                line
            )

            if number_match is None:
                continue

            channel_number = int(
                number_match.group(1)
            )

            # -------------------------------------------------
            # Description
            # -------------------------------------------------

            desc_match = re.search(
                r'Desc="([^"]*)"',
                line
            )

            if desc_match is None:
                continue

            channel_name = (
                desc_match.group(1)
            )

            self.channels[
                channel_number
            ] = channel_name

        # -----------------------------------------------------
        # Sort by channel number
        # -----------------------------------------------------

        self.channels = dict(
            sorted(
                self.channels.items()
            )
        )

        if not self.channels:

            raise RuntimeError(
                "No PGB output channels were found "
                "in the INF file."
            )

        # -----------------------------------------------------
        # Update GUI
        # -----------------------------------------------------

        def update_variable_list():
            self.populate_variable_list()
            self.channel_count_label.config(
                text=f"Output variables: {len(self.channels)}"
            )

        self.root.after(0, update_variable_list)

        # -----------------------------------------------------
        # Print detected variables
        # -----------------------------------------------------

        print(
            "\nDetected output variables:"
        )

        for number, name in (
            self.channels.items()
        ):

            print(
                f"  PGB({number}) -> {name}"
            )

    # =========================================================
    # READ OUT FILE
    # =========================================================

    def read_out_file(self):

        if self.out_file is None:

            raise RuntimeError(
                "No OUT file selected."
            )

        print(
            "\nReading OUT file..."
        )

        # -----------------------------------------------------
        # Read numerical data
        # -----------------------------------------------------

        # PSCAD legacy .out files contain a text description
        # in the first row (for example: "Single").
        # The numerical waveform data starts from the next row.
        try:
            self.data = np.loadtxt(
                self.out_file,
                skiprows=1
            )
        except ValueError as e:
            raise RuntimeError(
                "The detected .out file could not be read as numerical "
                "waveform data. PSCAD's first text/header row is skipped, "
                "but the remaining file still contains non-numeric data. "
                f"File: {self.out_file}. Details: {e}"
            ) from e

        # -----------------------------------------------------
        # Ensure 2D
        # -----------------------------------------------------

        if self.data.ndim == 1:

            self.data = (
                self.data.reshape(
                    1,
                    -1
                )
            )

        # -----------------------------------------------------
        # First column = time
        # -----------------------------------------------------

        self.time_data = (
            self.data[:, 0]
        )

        # -----------------------------------------------------
        # Check channel count
        # -----------------------------------------------------

        expected_columns = (
            len(self.channels) + 1
        )

        actual_columns = (
            self.data.shape[1]
        )

        print(
            "Expected columns:",
            expected_columns
        )

        print(
            "Actual columns:",
            actual_columns
        )

        if actual_columns < expected_columns:

            raise RuntimeError(
                f"OUT file contains {actual_columns} columns, "
                f"but {expected_columns} were expected."
            )

        print(
            "Number of samples:",
            len(self.time_data)
        )

        print(
            "Simulation time:",
            self.time_data[0],
            "to",
            self.time_data[-1],
            "seconds"
        )

    # =========================================================
    # PLOT SELECTED VARIABLE
    # =========================================================

    def populate_variable_list(self):
        """Populate the multi-select variable list from the detected channels."""
        if not hasattr(self, "variable_listbox"):
            return
        self.variable_listbox.delete(0, tk.END)
        search = self.variable_search_var.get().strip().lower() if hasattr(self, "variable_search_var") else ""
        self.filtered_channels = {}
        for number, name in self.channels.items():
            display = f"PGB({number})  |  {name}"
            if search and search not in display.lower():
                continue
            self.filtered_channels[number] = name
            self.variable_listbox.insert(tk.END, display)

    def filter_variables(self, *args):
        """Filter the variable list without changing the underlying channels."""
        self.populate_variable_list()

    def select_all_variables(self):
        if hasattr(self, "variable_listbox"):
            self.variable_listbox.select_set(0, tk.END)

    def clear_variable_selection(self):
        if hasattr(self, "variable_listbox"):
            self.variable_listbox.selection_clear(0, tk.END)

    def _get_selected_channel_numbers(self):
        selected = self.variable_listbox.curselection()
        numbers = list(self.filtered_channels.keys())
        return [numbers[i] for i in selected if 0 <= i < len(numbers)]

    def plot_selected(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Run a PSCAD simulation first.")
            return

        selected_numbers = self._get_selected_channel_numbers()
        if not selected_numbers:
            messagebox.showwarning("No Variables", "Select one or more output variables.")
            return

        self.ax.clear()
        plotted_names = []

        for channel_number in selected_numbers:
            name = self.channels[channel_number]
            if channel_number >= self.data.shape[1]:
                continue
            y = self.data[:, channel_number]
            self.ax.plot(self.time_data, y, linewidth=1, label=name)
            plotted_names.append(name)

        if not plotted_names:
            messagebox.showerror("Plot Error", "No selected variables could be plotted.")
            return

        self.ax.set_title("PSCAD Output Waveforms")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(self.grid_var.get())
        if self.legend_var.get():
            self.ax.legend(loc="best")

        # Populate limits with the actual data range.
        self.xmin_var.set(f"{self.time_data[0]:.6g}")
        self.xmax_var.set(f"{self.time_data[-1]:.6g}")
        self.ymin_var.set("")
        self.ymax_var.set("")

        self.figure.tight_layout()
        self.canvas.draw()

        self.status_var.set(
            f"Plotting {len(plotted_names)} variable(s)  |  Samples: {len(self.time_data)}  |  "
            f"Time: {self.time_data[0]:.6f} - {self.time_data[-1]:.6f} s"
        )
        self.footer_var.set("Plot updated")

    def apply_plot_controls(self):
        """Apply grid and legend settings to the current plot."""
        if self.data is None:
            return
        self.ax.grid(self.grid_var.get())
        legend = self.ax.get_legend()
        if self.legend_var.get():
            if legend is None:
                handles, labels = self.ax.get_legend_handles_labels()
                if handles:
                    self.ax.legend(loc="best")
        elif legend is not None:
            legend.remove()
        self.canvas.draw_idle()

    def apply_plot_limits(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Run a PSCAD simulation and plot a waveform first.")
            return
        try:
            xmin = float(self.xmin_var.get()) if self.xmin_var.get().strip() else None
            xmax = float(self.xmax_var.get()) if self.xmax_var.get().strip() else None
            ymin = float(self.ymin_var.get()) if self.ymin_var.get().strip() else None
            ymax = float(self.ymax_var.get()) if self.ymax_var.get().strip() else None
            if xmin is not None and xmax is not None and xmin >= xmax:
                raise ValueError("X min must be smaller than X max.")
            if ymin is not None and ymax is not None and ymin >= ymax:
                raise ValueError("Y min must be smaller than Y max.")
            if xmin is not None or xmax is not None:
                self.ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.ax.set_ylim(bottom=ymin, top=ymax)
            self.canvas.draw_idle()
            self.footer_var.set("Plot limits applied")
        except ValueError as e:
            messagebox.showerror("Invalid Plot Limits", str(e))

    def auto_scale_plot(self):
        if self.data is None:
            return
        self.ax.relim()
        self.ax.autoscale_view()
        self.xmin_var.set(f"{self.time_data[0]:.6g}")
        self.xmax_var.set(f"{self.time_data[-1]:.6g}")
        self.ymin_var.set("")
        self.ymax_var.set("")
        self.canvas.draw_idle()
        self.footer_var.set("Plot automatically scaled")

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(self.grid_var.get())
        self.canvas.draw_idle()
        self.footer_var.set("Plot cleared")

    def save_plot(self):
        if self.data is None or not self.ax.lines:
            messagebox.showwarning("No Plot", "Plot one or more variables before saving.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save Waveform Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("SVG Image", "*.svg"),
                ("All Files", "*.*")
            ]
        )
        if not filename:
            return
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            self.footer_var.set(f"Plot saved: {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # =========================================================
    # CLOSE PSCAD
    # =========================================================

    def close_pscad(self):

        try:

            if self.pscad is not None:

                print(
                    "\nClosing PSCAD..."
                )

                self.pscad.quit()

                self.pscad = None
                self.project = None

                self.close_pscad_button.config(
                    state="disabled"
                )

                self.set_status(
                    "PSCAD closed."
                )

                print(
                    "PSCAD closed."
                )

        except Exception as e:

            print(
                "Error closing PSCAD:",
                e
            )

    # =========================================================
    # CLOSE APPLICATION
    # =========================================================

    def on_close(self):

        try:

            if self.pscad is not None:

                self.pscad.quit()

        except Exception:
            pass

        self.root.destroy()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PSCADWaveformAnalyzer(
        root
    )

    root.mainloop()
