
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import traceback
import math

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    _SYMPY = False

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from numerical_methods import (
    incremental_search, bisection, regula_falsi, newton_raphson, secant,
)
from matrix_ops import MatrixOps

plt.rcParams["toolbar"] = "toolbar2"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     LIGHT-TEAL COLOUR PALETTE                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BG        = "#f8fafc"        # main window background (light blue-gray)
PANEL     = "#ffffff"        # left panel / card background (white)
PANEL2    = "#f1f5f9"        # secondary panel / dividers (light blue-gray)
ACCENT    = "#0f766e"        # primary accent (teal - professional)
ACCENT_LT = "#14b8a6"        # lighter accent (light teal)
GREEN     = "#059669"        # success / root found
YELLOW    = "#d97706"        # bracket markers
RED       = "#dc2626"        # error / not found
TEXT      = "#1e293b"        # primary text (dark slate)
SUBTEXT   = "#64748b"        # secondary / label text (medium slate)
INPUT_BG  = "#ffffff"        # entry / textbox background (white)
BORDER    = "#cbd5e1"        # borders and separators (light slate)
ROW_EVEN  = "#f8fafc"        # treeview even rows
ROW_ODD   = "#ffffff"        # treeview odd rows
HDR_BG    = "#0f766e"        # header bar background (matching teal)

# Matplotlib light theme
M_BG      = "#ffffff"        # figure background
M_AX      = "#f8fafc"        # axes background
M_GRID    = "#e2e8f0"        # grid lines
M_AXIS    = "#0f766e"        # X/Y axis lines (teal, bold, visible)
M_LINE    = "#0f766e"        # function curve (teal)
M_ROOT    = "#059669"        # root marker (green)
M_FILL_P  = "#ccfbf1"        # positive fill (light teal)
M_FILL_N  = "#fecaca"        # negative fill (light red)

FM  = ("Consolas", 10)
FUI = ("Segoe UI", 10)
FB  = ("Segoe UI", 10, "bold")
FT  = ("Segoe UI", 13, "bold")
FS  = ("Segoe UI",  8)
FSB = ("Segoe UI",  8, "bold")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         WIDGET HELPERS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_label(parent, text, fg=TEXT, font=FUI, bg=PANEL, **kw):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, **kw)


def make_entry(parent, width=14, textvariable=None, **kw):
    return tk.Entry(
        parent, width=width, bg=INPUT_BG, fg=TEXT,
        insertbackground=TEXT, font=FM, relief="flat", bd=4,
        highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=ACCENT, textvariable=textvariable, **kw)


def make_button(parent, text, cmd, bg=ACCENT, fg="white", **kw):
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg=fg,
        activebackground=ACCENT_LT, activeforeground="white",
        font=FB, relief="flat", cursor="hand2",
        padx=10, pady=5, **kw)


def make_section(parent, title, bg=PANEL):
    return tk.LabelFrame(
        parent, text=f"  {title}  ", bg=bg, fg=ACCENT,
        font=FSB, relief="groove", bd=1)


def make_separator(parent, bg=BORDER):
    return tk.Frame(parent, bg=bg, height=1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         EQUATION PARSER                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_equation(expr_str: str):
    """
    Parse a string expression into (f, f_prime) callables.
    Uses sympy when available for exact symbolic differentiation,
    falls back to a central-difference numerical derivative.
    Raises ValueError on bad input.
    """
    expr_str = expr_str.strip()
    # Convert ^ (caret) to ** (exponentiation) for user convenience
    expr_str = expr_str.replace("^", "**")
    
    # Handle implicit multiplication: convert patterns like "2x", "3x" to "2*x", "3*x"
    import re
    expr_str = re.sub(r'(\d)([x(])', r'\1*\2', expr_str)  # e.g., 2x -> 2*x, 2(x+1) -> 2*(x+1)
    expr_str = re.sub(r'(\))(\d|x|\()', r'\1*\2', expr_str)  # e.g., (x)2 -> (x)*2, (x)(y) -> (x)*(y)
    
    if not expr_str:
        raise ValueError("Expression is empty.")

    if _SYMPY:
        xsym  = sp.Symbol("x")
        expr  = sp.sympify(expr_str, locals={"x": xsym})
        dexpr = sp.diff(expr, xsym)
        f_    = sp.lambdify(xsym, expr,  modules=["math"])
        fp_   = sp.lambdify(xsym, dexpr, modules=["math"])
    else:
        ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        def f_(xv, _e=expr_str, _ns=ns):
            return eval(_e, {"x": xv, **_ns})
        def fp_(xv, h=1e-7):
            return (f_(xv + h) - f_(xv - h)) / (2 * h)

    # Smoke-test at x=1 to catch obvious parse errors early
    f_(1.0)
    fp_(1.0)
    return f_, fp_


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   PART A — ROOT FINDING TAB                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class NumericalTab(tk.Frame):
    """
    Root-Finding tab.
    Layout: fixed left config panel | expanding right area (graph top, table bottom).

    Key design decisions:
      - NO "Apply Equation" button. Equation is parsed fresh inside _run()
        every time Calculate is clicked.
      - Graph auto-updates on KeyRelease in the equation entry (debounced).
      - Dynamic fields (XU, dx) shown/hidden via grid()/grid_remove() only —
        never pack_forget — so widget order is always preserved.
    """

    METHODS = [
        "Incremental Search",
        "Bisection",
        "Regula-Falsi",
        "Newton-Raphson",
        "Secant",
    ]
    DEFAULT_EQ = ""

    _PARAM_CONFIG = {
        "Incremental Search": {
            "label_a": "Starting Point  (XL)",
            "label_b": None,
            "show_dx": True,
        },
        "Bisection": {
            "label_a": "Lower Bound  (XL)",
            "label_b": "Upper Bound  (XU)",
            "show_dx": False,
        },
        "Regula-Falsi": {
            "label_a": "Lower Bound  (XL)",
            "label_b": "Upper Bound  (XU)",
            "show_dx": False,
        },
        "Newton-Raphson": {
            "label_a": "Initial Guess  (X0)",
            "label_b": None,
            "show_dx": False,
        },
        "Secant": {
            "label_a": "First Guess  (X0)",
            "label_b": "Second Guess  (X1)",
            "show_dx": False,
        },
    }

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._f         = None
        self._fp        = None
        self._root      = None
        self._iterations = []
        self._resize_id = None
        self._eq_debounce_id = None
        self._build_ui()
        self._refresh_param_fields()
        # Parse default equation silently for initial graph
        try:
            self._f, self._fp = parse_equation(self.DEFAULT_EQ)
        except Exception:
            pass
        # Defer plot drawing until UI is fully loaded
        self.after_idle(self._draw_plot)

    # ── UI layout ─────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=10, pady=8)
        outer.columnconfigure(0, weight=0, minsize=300)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────
        left_outer = tk.Frame(outer, bg=PANEL, width=300,
                               relief="flat", bd=0,
                               highlightthickness=1,
                               highlightbackground=BORDER)
        left_outer.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left_outer.grid_propagate(False)
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)

        lcanvas = tk.Canvas(left_outer, bg=PANEL, bd=0,
                             highlightthickness=0, width=298)
        lsb = ttk.Scrollbar(left_outer, orient="vertical",
                              command=lcanvas.yview)
        lcanvas.configure(yscrollcommand=lsb.set)
        lcanvas.grid(row=0, column=0, sticky="nsew")
        lsb.grid(row=0, column=1, sticky="ns")

        self._left_inner = tk.Frame(lcanvas, bg=PANEL)
        _lwin = lcanvas.create_window((0, 0), window=self._left_inner,
                                       anchor="nw")

        self._left_inner.bind(
            "<Configure>",
            lambda e: lcanvas.configure(
                scrollregion=lcanvas.bbox("all")))
        lcanvas.bind(
            "<Configure>",
            lambda e: lcanvas.itemconfig(_lwin, width=e.width))

        def _lwheel(e):
            delta = -1 if (getattr(e, "delta", 0) < 0 or e.num == 5) else 1
            lcanvas.yview_scroll(delta, "units")

        lcanvas.bind("<MouseWheel>", _lwheel)
        lcanvas.bind("<Button-4>",   _lwheel)
        lcanvas.bind("<Button-5>",   _lwheel)

        self._build_left_panel(self._left_inner)

        # ── Right panel ───────────────────────────────────────────────
        right = tk.Frame(outer, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)
        self._build_graph(right)
        self._build_table(right)

    # ── Left panel ────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        r = 0

        # Title
        make_label(parent, "Root-Finding Methods",
                   fg=ACCENT, font=("Segoe UI", 11, "bold"),
                   bg=PANEL).grid(row=r, column=0,
                                   pady=(14, 4), padx=16, sticky="ew")
        r += 1
        make_separator(parent).grid(row=r, column=0,
                                     sticky="ew", padx=12, pady=(0, 10))
        r += 1

        # ── Method selector ───────────────────────────────────────────
        make_label(parent, "Method", fg=SUBTEXT,
                   font=FS, bg=PANEL).grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 2))
        r += 1
        self.method_var = tk.StringVar(value=self.METHODS[0])
        self._cb_method = ttk.Combobox(
            parent, textvariable=self.method_var,
            values=self.METHODS, state="readonly", width=27, font=FUI)
        self._cb_method.grid(row=r, column=0, sticky="ew",
                              padx=16, pady=(0, 10))
        self._cb_method.bind("<<ComboboxSelected>>",
                              lambda _: self._on_method_change())
        r += 1

        make_separator(parent).grid(row=r, column=0,
                                     sticky="ew", padx=12, pady=(0, 10))
        r += 1

        # ── Equation f(x) — NO Apply button ──────────────────────────
        make_label(parent, "Equation  f(x)", fg=SUBTEXT,
                   font=FS, bg=PANEL).grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 2))
        r += 1
        self.eq_var = tk.StringVar(value=self.DEFAULT_EQ)
        self.eq_entry = make_entry(parent, width=30,
                                    textvariable=self.eq_var)
        self.eq_entry.grid(row=r, column=0, sticky="ew",
                            padx=16, pady=(0, 12), ipady=4)
        # Live graph preview on key release (debounced 600 ms)
        self.eq_entry.bind("<KeyRelease>", self._on_eq_keyrelease)
        r += 1

        make_separator(parent).grid(row=r, column=0,
                                     sticky="ew", padx=12, pady=(0, 10))
        r += 1

        # ── Parameter A ───────────────────────────────────────────────
        self.lbl_a = make_label(parent, "Lower Bound  (XL)",
                                 fg=SUBTEXT, font=FS, bg=PANEL)
        self.lbl_a.grid(row=r, column=0, sticky="w",
                         padx=16, pady=(0, 2))
        r += 1
        self.var_a = tk.StringVar()
        self.ent_a = make_entry(parent, width=30, textvariable=self.var_a)
        self.ent_a.grid(row=r, column=0, sticky="ew",
                         padx=16, pady=(0, 8), ipady=4)
        r += 1

        # ── Parameter B (XU / X1) — grid_remove when not needed ──────
        self.lbl_b = make_label(parent, "Upper Bound  (XU)",
                                 fg=SUBTEXT, font=FS, bg=PANEL)
        self.lbl_b.grid(row=r, column=0, sticky="w",
                         padx=16, pady=(0, 2))
        r += 1
        self.var_b = tk.StringVar()
        self.ent_b = make_entry(parent, width=30, textvariable=self.var_b)
        self.ent_b.grid(row=r, column=0, sticky="ew",
                         padx=16, pady=(0, 8), ipady=4)
        r += 1

        # ── Step Size dx (Incremental Search only) ────────────────────
        self.lbl_dx = make_label(parent, "Step Size  (dx)",
                                  fg=SUBTEXT, font=FS, bg=PANEL)
        self.lbl_dx.grid(row=r, column=0, sticky="w",
                          padx=16, pady=(0, 2))
        r += 1
        self.var_dx = tk.StringVar(value="0.1")
        self.ent_dx = make_entry(parent, width=30,
                                  textvariable=self.var_dx)
        self.ent_dx.grid(row=r, column=0, sticky="ew",
                          padx=16, pady=(0, 8), ipady=4)
        r += 1

        # ── Max Iterations ────────────────────────────────────────────
        make_label(parent, "Max Iterations", fg=SUBTEXT,
                   font=FS, bg=PANEL).grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 2))
        r += 1
        self.var_mi = tk.StringVar(value="100")
        self.ent_mi = make_entry(parent, width=30,
                                  textvariable=self.var_mi)
        self.ent_mi.grid(row=r, column=0, sticky="ew",
                          padx=16, pady=(0, 8), ipady=4)
        r += 1

        # ── Tolerance ─────────────────────────────────────────────────
        make_label(parent, "Tolerance", fg=SUBTEXT,
                   font=FS, bg=PANEL).grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 2))
        r += 1
        self.var_tol = tk.StringVar(value="0.0001")
        self.ent_tol = make_entry(parent, width=30,
                                   textvariable=self.var_tol)
        self.ent_tol.grid(row=r, column=0, sticky="ew",
                           padx=16, pady=(0, 12), ipady=4)
        r += 1

        make_separator(parent).grid(row=r, column=0,
                                     sticky="ew", padx=12, pady=(0, 12))
        r += 1

        # ── Action buttons ────────────────────────────────────────────
        btn_frame = tk.Frame(parent, bg=PANEL)
        btn_frame.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 4))
        btn_frame.columnconfigure(0, weight=1)
        r += 1

        tk.Button(btn_frame, text="▶  Calculate Root",
                  command=self._run,
                  bg=ACCENT, fg="white",
                  activebackground=ACCENT_LT, activeforeground="white",
                  font=FB, relief="flat", cursor="hand2",
                  padx=10, pady=8).grid(row=0, column=0,
                                         sticky="ew", pady=(0, 6))
        tk.Button(btn_frame, text="✕  Clear",
                  command=self._clear_inputs,
                  bg=PANEL2, fg=TEXT,
                  activebackground=BORDER, activeforeground=TEXT,
                  font=FB, relief="flat", cursor="hand2",
                  padx=10, pady=8).grid(row=1, column=0, sticky="ew")

        # ── Result display bar ────────────────────────────────────────
        tk.Frame(parent, bg=PANEL, height=8).grid(row=r, column=0)
        r += 1
        self.result_lbl = make_label(
            parent, "", fg=SUBTEXT,
            font=("Consolas", 9), bg=PANEL,
            wraplength=265, justify="left")
        self.result_lbl.grid(row=r, column=0, sticky="ew",
                              padx=16, pady=(4, 14))

    # ── Graph ─────────────────────────────────────────────────────────────

    def _build_graph(self, parent):
        graph_sec = make_section(parent, "Graph  —  scroll to zoom")
        graph_sec.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        graph_sec.rowconfigure(0, weight=1)
        graph_sec.columnconfigure(0, weight=1)

        self._graph_host = tk.Frame(graph_sec, bg=M_BG)
        self._graph_host.grid(row=0, column=0, sticky="nsew",
                               padx=4, pady=(4, 0))
        self._graph_host.rowconfigure(0, weight=1)
        self._graph_host.columnconfigure(0, weight=1)

        tb_row = tk.Frame(graph_sec, bg=BG, height=30)
        tb_row.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        tb_row.grid_propagate(False)

        self.fig, self.ax = plt.subplots(figsize=(5, 3.5), dpi=96)
        self.fig.patch.set_facecolor(M_BG)
        self.ax.set_facecolor(M_AX)
        self.fig.subplots_adjust(left=0.11, right=0.97,
                                  top=0.91, bottom=0.12)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self._graph_host)
        cw = self.canvas.get_tk_widget()
        cw.grid(row=0, column=0, sticky="nsew")

        self._tb = NavigationToolbar2Tk(self.canvas, tb_row)
        self._tb.update()
        self._style_toolbar()

        self._graph_host.bind("<Configure>", self._on_resize)
        cw.bind("<MouseWheel>", self._on_scroll)
        cw.bind("<Button-4>",   self._on_scroll)
        cw.bind("<Button-5>",   self._on_scroll)

        self._draw_plot()

    def _style_toolbar(self):
        try:
            self._tb.config(bg=BG)
            for child in self._tb.winfo_children():
                try:
                    child.config(bg=BG, fg=TEXT,
                                  activebackground=PANEL2,
                                  activeforeground=TEXT,
                                  highlightbackground=BG,
                                  relief="flat")
                except Exception:
                    pass
        except Exception:
            pass

    def _on_resize(self, event):
        if self._resize_id:
            self.after_cancel(self._resize_id)
        self._resize_id = self.after(
            60, self._do_resize, event.width, event.height)

    def _do_resize(self, w, h):
        self._resize_id = None
        if w < 20 or h < 20:
            return
        dpi = self.fig.dpi
        self.fig.set_size_inches(w / dpi, h / dpi, forward=False)
        self.fig.subplots_adjust(left=0.11, right=0.97,
                                  top=0.91, bottom=0.12)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if hasattr(event, "delta"):
            factor = 0.85 if event.delta < 0 else 1.0 / 0.85
        elif event.num == 5:
            factor = 0.85
        else:
            factor = 1.0 / 0.85
        ax = self.ax
        try:
            inv = ax.transData.inverted()
            xd  = inv.transform((event.x, event.y))[0]
            yd  = inv.transform((event.x, event.y))[1]
        except Exception:
            return
        ax.set_xlim([xd + (x - xd) * factor for x in ax.get_xlim()])
        ax.set_ylim([yd + (y - yd) * factor for y in ax.get_ylim()])
        self.canvas.draw_idle()

    # ── Iteration table ───────────────────────────────────────────────────

    def _build_table(self, parent):
        tbl_sec = make_section(parent, "Iteration Table")
        tbl_sec.grid(row=1, column=0, sticky="nsew")
        tbl_sec.rowconfigure(0, weight=1)
        tbl_sec.columnconfigure(0, weight=1)

        inner = tk.Frame(tbl_sec, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=4, pady=4)
        inner.rowconfigure(0, weight=1)
        inner.columnconfigure(0, weight=1)

        vsb = ttk.Scrollbar(inner, orient="vertical")
        hsb = ttk.Scrollbar(inner, orient="horizontal")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        st = ttk.Style()
        st.configure("LT.Treeview",
                      background=ROW_EVEN, foreground=TEXT,
                      fieldbackground=ROW_EVEN, rowheight=24,
                      font=("Consolas", 9))
        st.configure("LT.Treeview.Heading",
                      background=PANEL2, foreground=ACCENT,
                      font=("Segoe UI", 9, "bold"), relief="flat")
        st.map("LT.Treeview",
                background=[("selected", "#e0e7ff")],
                foreground=[("selected", ACCENT)])

        self.tree = ttk.Treeview(inner, style="LT.Treeview",
                                  yscrollcommand=vsb.set,
                                  xscrollcommand=hsb.set,
                                  selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

    def _populate_table(self, rows: list):
        """Clear and repopulate Treeview. Always clear first."""
        self.tree.delete(*self.tree.get_children())
        self._iterations = rows or []
        if not rows:
            return
        cols = list(rows[0].keys())
        self.tree["columns"] = cols
        self.tree["show"]    = "headings"
        for c in cols:
            w = 290 if c == "Remark" else max(72, len(c) * 9 + 12)
            self.tree.heading(c, text=c, anchor="center")
            self.tree.column(c, width=w, anchor="center",
                              minwidth=55, stretch=(c == "Remark"))
        for i, row in enumerate(rows):
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end",
                              values=list(row.values()), tags=(tag,))
        self.tree.tag_configure("even", background=ROW_EVEN)
        self.tree.tag_configure("odd",  background=ROW_ODD)

    def _get_iteration_points(self):
        """Return (iteration_label, x_value) points for the current method."""
        if not self._iterations or self._f is None:
            return []

        method = self.method_var.get()
        points = []

        for row in self._iterations:
            x = None
            if method in ("Bisection", "Regula-Falsi") and "XR" in row:
                x = row["XR"]
            elif method == "Newton-Raphson" and "Xi" in row:
                x = row["Xi"]
            elif method == "Secant" and "Xi" in row:
                x = row["Xi"]
            elif method == "Incremental Search":
                # Use the actual scan endpoint for the marker, not the midpoint
                # of the bracket, so the plotted point matches the iteration step.
                if "XU" in row:
                    x = row["XU"]
                elif "XL" in row:
                    x = row["XL"]
                else:
                    x = None
            elif "Xi" in row:
                x = row["Xi"]

            if x is None:
                continue
            try:
                x = float(x)
            except Exception:
                continue

            points.append((row.get("Iteration", ""), x))

        return points

    # ── Equation handling ─────────────────────────────────────────────────

    def _on_eq_keyrelease(self, _=None):
        """Debounce live graph preview — waits 600 ms after last keystroke."""
        if self._eq_debounce_id:
            self.after_cancel(self._eq_debounce_id)
        self._eq_debounce_id = self.after(600, self._try_parse_and_plot)

    def _try_parse_and_plot(self):
        """Parse equation silently; update graph only if valid."""
        try:
            f_, fp_ = parse_equation(self.eq_var.get())
            self._f  = f_
            self._fp = fp_
            self._root = None
            self._iterations = []
            self._draw_plot()
        except Exception:
            pass  # Don't interrupt the user while they're typing

    # ── Method change ─────────────────────────────────────────────────────

    def _on_method_change(self):
        self._refresh_param_fields()
        self._draw_plot()

    def _refresh_param_fields(self):
        m      = self.method_var.get()
        cfg    = self._PARAM_CONFIG[m]
        lbl_b  = cfg["label_b"]
        show_dx = cfg["show_dx"]

        self.lbl_a.config(text=cfg["label_a"])

        if lbl_b is None:
            self.lbl_b.grid_remove()
            self.ent_b.grid_remove()
        else:
            self.lbl_b.config(text=lbl_b)
            self.lbl_b.grid()
            self.ent_b.grid()

        if show_dx:
            self.lbl_dx.grid()
            self.ent_dx.grid()
        else:
            self.lbl_dx.grid_remove()
            self.ent_dx.grid_remove()

    # ── Clear ─────────────────────────────────────────────────────────────

    def _clear_inputs(self):
        self.eq_var.set("")
        self._f = None
        self._fp = None
        self.var_a.set("")
        self.var_b.set("")
        self.var_dx.set("0.1")
        self.var_mi.set("100")
        self.var_tol.set("0.0001")
        self.result_lbl.config(text="", fg=SUBTEXT)
        self._iterations = []
        self._draw_plot()

    # ── Calculate Root ────────────────────────────────────────────────────

    def _run(self):
        """
        Parse equation fresh on every Calculate click — no Apply button needed.
        Then dispatch to the selected numerical method.
        """
        # 1. Parse equation
        try:
            f_, fp_ = parse_equation(self.eq_var.get())
            self._f  = f_
            self._fp = fp_
        except Exception as e:
            messagebox.showerror(
                "Equation Error",
                f"Cannot parse  f(x) = {self.eq_var.get()}\n\n{e}")
            return

        m   = self.method_var.get()
        cfg = self._PARAM_CONFIG[m]

        # 2. Validate numeric parameters
        try:
            a   = float(self.var_a.get())
            mi  = int(self.var_mi.get())
            tol = float(self.var_tol.get())
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Enter valid numeric values for all parameters.")
            return

        b = None
        if cfg["label_b"] is not None:
            try:
                b = float(self.var_b.get())
            except ValueError:
                messagebox.showerror(
                    "Input Error",
                    f"Invalid value for '{cfg['label_b']}'.")
                return

        dx = None
        if cfg["show_dx"]:
            try:
                dx = float(self.var_dx.get())
                if dx <= 0:
                    raise ValueError("Step size must be positive.")
            except ValueError as e:
                messagebox.showerror("Input Error",
                                      f"Invalid Step Size:\n{e}")
                return

        # 3. Run method
        try:
            root, tbl = self._dispatch(m, a, b, tol, mi, dx)
        except Exception:
            messagebox.showerror("Method Error", traceback.format_exc())
            return

        # 4. Update UI
        self._root = root
        self._populate_table(tbl)
        self._draw_plot(root)

        if root is not None:
            try:
                froot = self._f(root)
            except Exception:
                froot = float("nan")
            self.result_lbl.config(
                text=(f"Root ≈ {root:.8f}\n"
                      f"f(root) = {froot:.4e}\n"
                      f"Iterations: {len(tbl)}"),
                fg=GREEN)
        else:
            self.result_lbl.config(
                text="No root found in the given range.", fg=RED)

    def _dispatch(self, m, a, b, tol, mi, dx=None):
        f, fp = self._f, self._fp
        if m == "Incremental Search":
            step = dx if dx is not None else 0.1
            return incremental_search(f, a, xu=None, dx=step, tol=tol)
        if m == "Bisection":
            return bisection(f, a, b, tol, mi)
        if m == "Regula-Falsi":
            return regula_falsi(f, a, b, tol, mi)
        if m == "Newton-Raphson":
            return newton_raphson(f, fp, a, tol, mi)
        if m == "Secant":
            return secant(f, a, b, tol, mi)
        raise ValueError(f"Unknown method: {m}")

    # ── Plot ──────────────────────────────────────────────────────────────

    def _draw_plot(self, root=None):
        self.ax.clear()
        self.ax.set_facecolor(M_AX)

        if self._f is None:
            self.ax.text(
                0.5, 0.5,
                "Enter an equation in the box above\nand click ▶ Calculate Root",
                transform=self.ax.transAxes,
                ha="center", va="center",
                color=SUBTEXT, fontsize=10,
                bbox=dict(fc=PANEL, ec=BORDER,
                           boxstyle="round,pad=0.7"))
            self.canvas.draw_idle()
            return

        m   = self.method_var.get()
        cfg = self._PARAM_CONFIG[m]

        try:
            a = float(self.var_a.get()) if self.var_a.get() else -5.0
        except ValueError:
            a = -5.0
        try:
            b_str = self.var_b.get()
            b = (float(b_str)
                 if cfg["label_b"] and b_str else a + 6.0)
        except ValueError:
            b = a + 6.0

        span   = max(abs(b - a), 1.0)
        margin = span * 0.7
        xmin   = a - margin
        xmax   = b + margin

        xs = np.linspace(xmin, xmax, 700)
        try:
            ys = np.array([self._f(xi) for xi in xs], dtype=float)
        except Exception:
            self.ax.text(0.5, 0.5, "⚠ f(x) evaluation error",
                          transform=self.ax.transAxes,
                          ha="center", va="center", color=RED)
            self.canvas.draw_idle()
            return

        fin = ys[np.isfinite(ys)]
        if len(fin) == 0:
            self.canvas.draw_idle()
            return
        lo  = np.percentile(fin, 2)
        hi  = np.percentile(fin, 98)
        pad = max((hi - lo) * 0.25, 0.5)
        ylo = lo - pad
        yhi = hi + pad
        ysc = np.clip(ys, ylo - 1, yhi + 1)

        # Plot iteration points from the current numerical method
        iteration_points = self._get_iteration_points()
        if iteration_points:
            iter_x = [x for _, x in iteration_points]
            self.ax.scatter(iter_x, [self._f(x) for x in iter_x],
                           color=ACCENT_LT, edgecolors="white",
                           s=56, zorder=6, label="Iteration points")
            for label, x in iteration_points:
                try:
                    y = float(self._f(x))
                except Exception:
                    continue
                yoff = 4 if y >= 0 else -4
                self.ax.annotate(
                    str(label), xy=(x, y), xytext=(0, yoff),
                    textcoords="offset points", ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=7, color=ACCENT, zorder=7)

        # Fill areas
        self.ax.fill_between(xs, ysc, 0, where=(ysc >= 0),
                              alpha=0.18, color=M_FILL_P,
                              interpolate=True)
        self.ax.fill_between(xs, ysc, 0, where=(ysc <= 0),
                              alpha=0.18, color=M_FILL_N,
                              interpolate=True)

        # Function curve
        eq_str = self.eq_var.get()
        self.ax.plot(xs, ysc, color=M_LINE, lw=2.2, zorder=3,
                      label=f"f(x) = {eq_str}")

        # ── Bold, visible X and Y axes ─────────────────────────────
        self.ax.axhline(0, color=M_AXIS, lw=1.8, zorder=4, alpha=0.9)
        self.ax.axvline(0, color=M_AXIS, lw=1.8, zorder=4, alpha=0.9)

        # Bracket markers
        if cfg["label_b"] is not None:
            try:
                fa_ = self._f(a)
                fb_ = self._f(b)
                self.ax.axvline(a, color=YELLOW, lw=1.2,
                                 ls="--", alpha=0.7, zorder=3)
                self.ax.axvline(b, color=YELLOW, lw=1.2,
                                 ls="--", alpha=0.7, zorder=3)
                self.ax.plot([a, b], [fa_, fb_], "D",
                              color=YELLOW, ms=5, zorder=5,
                              label="XL / XU")
            except Exception:
                pass

        # Root marker
        if root is not None:
            try:
                fy = self._f(root)
                self.ax.plot([root, root], [0, fy],
                              color=M_ROOT, lw=1.4,
                              ls="--", alpha=0.8, zorder=4)
                self.ax.plot(root, fy, "o", color=M_ROOT, ms=10,
                              zorder=6, markeredgewidth=2,
                              markeredgecolor="white")
                ox = (xmax - xmin) * 0.04
                oy = (yhi  - ylo)  * 0.10
                self.ax.annotate(
                    f" x ≈ {root:.6f}\n f ≈ {fy:.3e}",
                    xy=(root, fy),
                    xytext=(root + ox, fy + oy),
                    color=TEXT, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4",
                               fc=PANEL, ec=M_ROOT,
                               lw=1.2, alpha=0.95),
                    arrowprops=dict(arrowstyle="->",
                                     color=M_ROOT, lw=1.3),
                    zorder=7)
            except Exception:
                pass

        self.ax.set_ylim(ylo, yhi)
        self.ax.set_title(f"f(x) = {eq_str}",
                           color=TEXT, fontsize=9, pad=5,
                           fontfamily="Consolas")
        self.ax.set_xlabel("x",    color=TEXT,
                            fontsize=9, fontweight="bold")
        self.ax.set_ylabel("f(x)", color=TEXT,
                            fontsize=9, fontweight="bold")
        self.ax.tick_params(colors=SUBTEXT, labelsize=8)
        for sp_ in self.ax.spines.values():
            sp_.set_edgecolor(BORDER)
        self.ax.legend(fontsize=7.5, facecolor=PANEL,
                        labelcolor=TEXT, edgecolor=BORDER,
                        loc="upper left")
        self.ax.grid(True, color=M_GRID, ls="--",
                      lw=0.7, alpha=1.0, zorder=0)
        self.canvas.draw_idle()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   PART B — MATRIX OPERATIONS TAB                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MatrixTab(tk.Frame):

    OPERATIONS = [
        "Matrix Addition",
        "Matrix Multiplication",
        "Adjoint",
        "Inverse Matrix",
        "Determinants",
        "Power of Matrix",
        "Equations",
        "Transpose of Matrix",
    ]
    # Default matrices left empty by user request
    DA  = ""
    DB  = ""
    DVB = ""

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        r = 0

        # Title
        make_label(self, "Matrix Operations",
                   fg=ACCENT, font=FT, bg=BG).grid(
            row=r, column=0, sticky="w", padx=16, pady=(12, 6))
        r += 1

        # Operation row
        op_sec = make_section(self, "Operation", bg=BG)
        op_sec.grid(row=r, column=0, sticky="ew", padx=14, pady=(0, 6))
        op_sec.columnconfigure(0, weight=1)
        r += 1

        ctrl = tk.Frame(op_sec, bg=BG)
        ctrl.pack(fill="x", padx=10, pady=8)

        make_label(ctrl, "Operation:", bg=BG).pack(side="left")
        self.op_var = tk.StringVar(value=self.OPERATIONS[0])
        cb = ttk.Combobox(ctrl, textvariable=self.op_var,
                           values=self.OPERATIONS, state="readonly",
                           width=26, font=FUI)
        cb.pack(side="left", padx=(6, 16))
        cb.bind("<<ComboboxSelected>>", self._on_op_change)

        # Power n (hidden by default)
        self._pwr_frame = tk.Frame(ctrl, bg=BG)
        self._pwr_frame.pack(side="left")
        make_label(self._pwr_frame, "Power  n :", bg=BG).pack(side="left")
        self.var_n = tk.StringVar(value="2")
        self.ent_n = make_entry(self._pwr_frame, width=5,
                                 textvariable=self.var_n)
        self.ent_n.pack(side="left", padx=(4, 16), ipady=2)
        self._pwr_frame.pack_forget()

        make_button(ctrl, "▶  Compute", self._run).pack(
            side="left", padx=(0, 8))
        make_button(ctrl, "✕  Clear", self._clear,
                    bg=PANEL2, fg=TEXT).pack(side="left")

        # Matrix inputs
        self._inp_outer = tk.Frame(self, bg=BG)
        self._inp_outer.grid(row=r, column=0, sticky="ew",
                              padx=14, pady=(0, 4))
        self._inp_outer.columnconfigure(0, weight=1)
        self._inp_outer.columnconfigure(1, weight=1)
        r += 1

        sec_a = make_section(self._inp_outer,
                              "Matrix A  —  rows on separate lines",
                              bg=BG)
        sec_a.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.ta = self._make_matrix_box(sec_a, self.DA)

        self._sec_b = make_section(self._inp_outer, "Matrix B", bg=BG)
        self._sec_b.grid(row=0, column=1, sticky="nsew")
        self.tb = self._make_matrix_box(self._sec_b, self.DB)

        make_label(self,
                   "Each row on its own line, values space-separated"
                   "  —  e.g.  2 1 0  then  1 3 1  then  0 1 2",
                   fg=SUBTEXT, font=FS, bg=BG).grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4))
        r += 1

        res = make_section(self, "Result", bg=BG)
        res.grid(row=r, column=0, sticky="nsew", padx=14, pady=(0, 12))
        res.rowconfigure(0, weight=1)
        res.columnconfigure(0, weight=1)

        self.res_txt = scrolledtext.ScrolledText(
            res, height=10, bg=INPUT_BG, fg=GREEN,
            font=FM, relief="flat", bd=4,
            state="disabled", wrap="none")
        self.res_txt.pack(fill="both", expand=True, padx=6, pady=6)

        self._on_op_change()

    def _make_matrix_box(self, parent, default: str) -> tk.Text:
        t = tk.Text(
            parent, height=6, bg=INPUT_BG, fg=TEXT,
            insertbackground=TEXT, font=FM, relief="flat", bd=4,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, wrap="none")
        t.pack(fill="both", expand=True, padx=6, pady=6)
        t.insert("1.0", default)
        return t

    def _on_op_change(self, _=None):
        op     = self.op_var.get()
        need_b = ("Multiplication" in op) or ("Addition" in op) or ("Equations" in op)
        need_n = "Power" in op

        if need_b:
            self._sec_b.grid()
            if "Equations" in op:
                self._sec_b.config(
                    text="  Vector b  (one value per line)  ")
            else:
                self._sec_b.config(text="  Matrix B  ")
        else:
            self._sec_b.grid_remove()

        if need_n:
            self._pwr_frame.pack(side="left")
        else:
            self._pwr_frame.pack_forget()

        # Clear Matrix A and B and the result output when operation changes
        self.ta.delete("1.0", "end")
        self.tb.delete("1.0", "end")
        self.res_txt.config(state="normal")
        self.res_txt.delete("1.0", "end")
        self.res_txt.config(state="disabled")

    def _parse_matrix(self, widget: tk.Text) -> np.ndarray:
        raw  = widget.get("1.0", "end").strip()
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append([float(v) for v in line.split()])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid value in row '{line}': {e}") from e
        if not rows:
            raise ValueError("Matrix is empty.")
        col_counts = {len(r) for r in rows}
        if len(col_counts) > 1:
            raise ValueError(
                "Inconsistent column counts. "
                "Each row must have the same number of values.")
        return np.array(rows, dtype=float)

    def _show_result(self, label: str, value):
        self.res_txt.config(state="normal")
        self.res_txt.delete("1.0", "end")
        sep = "─" * 58
        self.res_txt.insert("end", f"{sep}\n  {label}\n{sep}\n\n")
        if isinstance(value, np.ndarray):
            self.res_txt.insert(
                "end",
                np.array2string(value, precision=6,
                                 suppress_small=True,
                                 separator="   ") + "\n")
        else:
            self.res_txt.insert("end", str(value) + "\n")
        self.res_txt.config(state="disabled")

    def _clear(self):
        # Clear matrix inputs
        self.ta.delete("1.0", "end")
        self.tb.delete("1.0", "end")
        self.var_n.set("")
        # Clear result
        self.res_txt.config(state="normal")
        self.res_txt.delete("1.0", "end")
        self.res_txt.config(state="disabled")

    def _run(self):
        op = self.op_var.get()
        try:
            A = self._parse_matrix(self.ta)
        except Exception as e:
            messagebox.showerror("Parse Error — Matrix A", str(e))
            return

        B = None
        if ("Multiplication" in op) or ("Addition" in op) or ("Equations" in op):
            try:
                B = self._parse_matrix(self.tb)
            except Exception as e:
                messagebox.showerror(
                    "Parse Error — Matrix B / vector b", str(e))
                return

        try:
            result = self._compute(op, A, B)
            self._show_result(op, result)
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))

    def _compute(self, op: str, A: np.ndarray, B):
        mo = MatrixOps
        if   "Addition"       in op: return mo.add(A, B)
        elif "Multiplication" in op: return mo.multiply(A, B)
        elif "Adjoint"        in op: return mo.adjoint(A)
        elif "Inverse"        in op: return mo.inverse(A)
        elif "Determinants"   in op:
            return f"det(A)  =  {mo.determinant(A):.10f}"
        elif "Power"          in op:
            try:
                n = int(self.var_n.get())
            except ValueError:
                raise ValueError("Power n must be an integer.")
            return mo.power(A, n)
        elif "Equations"      in op: return mo.solve(A, B.flatten())
        elif "Transpose"      in op: return mo.transpose(A)
        raise ValueError(f"Unknown operation: {op}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         APPLICATION ROOT                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Methods")
        self.geometry("1280x820")
        self.minsize(1024, 700)
        self.configure(bg=BG)
        self._apply_styles()
        self._build_ui()

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        # Notebook
        s.configure("TNotebook",
                     background=BG, borderwidth=0)
        s.configure("TNotebook.Tab",
                     background=PANEL2, foreground=SUBTEXT,
                     padding=[18, 7], font=FB)
        s.map("TNotebook.Tab",
               background=[("selected", ACCENT)],
               foreground=[("selected", "white")])

        # Scrollbars
        s.configure("Vertical.TScrollbar",
                     background=PANEL2, troughcolor=BG,
                     arrowcolor=SUBTEXT, relief="flat", borderwidth=0)
        s.configure("Horizontal.TScrollbar",
                     background=PANEL2, troughcolor=BG,
                     arrowcolor=SUBTEXT, relief="flat", borderwidth=0)

        # Combobox
        s.configure("TCombobox",
                     fieldbackground=INPUT_BG, background=PANEL2,
                     foreground=TEXT, selectbackground=ACCENT,
                     selectforeground="white", relief="flat")
        s.map("TCombobox",
               fieldbackground=[("readonly", INPUT_BG)],
               background=[("readonly", PANEL2)])

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=HDR_BG, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="  🔢  Numerical Methods",
            bg=HDR_BG, fg="white",
            font=("Segoe UI", 13, "bold")).pack(side="left", padx=12)
        # No version/library label on the right — removed per request

        # ── Notebook ──────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_a = NumericalTab(nb)
        nb.add(self.tab_a, text="  Part A — Root Finding  ")

        self.tab_b = MatrixTab(nb)
        nb.add(self.tab_b, text="  Part B — Matrix Ops  ")

        # ── Footer ────────────────────────────────────────────────────
        ft = tk.Frame(self, bg=PANEL2, height=24)
        ft.pack(fill="x", side="bottom")
        ft.pack_propagate(False)
        tk.Label(
            ft,
            text=("  ①  Enter f(x)   "
                  "②  Fill parameters   "
                  "③  Click Calculate Root   |   "
                  "Scroll over graph to zoom"),
            bg=PANEL2, fg=SUBTEXT, font=FS).pack(side="left")
        tk.Label(
            ft,
            text="v6  Production  ",
            bg=PANEL2, fg=SUBTEXT, font=FS).pack(side="right")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
