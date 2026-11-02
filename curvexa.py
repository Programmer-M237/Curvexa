#!/usr/bin/env python3
"""
Curvexa - a graph-plotting studio built with PySide6 + NumPy + Matplotlib + SymPy

Input methods : y=f(x) | parametric | polar | implicit F(x,y)=G | data table (paste / CSV / fill)
Extras        : math keyboard, live LaTeX preview, derivative overlay, polynomial fit,
                project save/load, image export (PNG/JPG/SVG/PDF/TIFF), data export (CSV)
LaTeX         : matplotlib mathtext by default; real compiled LaTeX if latex+dvipng+ghostscript exist.

Install : pip install PySide6 numpy matplotlib sympy
Run     : python curvexa.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass, field

os.environ.setdefault("QT_API", "pyside6")

import matplotlib as mpl
import matplotlib.style  # noqa: F401  (needed for mpl.style)
import numpy as np
import sympy as sp

mpl.use("QtAgg")
from matplotlib import mathtext
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QColor, QIcon, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath,
    QPalette, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# LaTeX capability detection
# =============================================================================
HAVE_REAL_LATEX = bool(
    shutil.which("latex")
    and shutil.which("dvipng")
    and (shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c"))
)

# =============================================================================
# Math engine
# =============================================================================
X, Y, T, TH = sp.symbols("x y t theta", real=True)


class MathError(Exception):
    pass


def _log(a, b=10):
    return sp.log(a) / sp.log(b)


NAMESPACE = {
    "x": X, "y": Y, "t": T, "theta": TH, "θ": TH,
    "pi": sp.pi, "π": sp.pi, "e": sp.E,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "acot": sp.acot, "atan2": sp.atan2,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "coth": sp.coth,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "ln": sp.log, "log": _log, "log10": _log,
    "log2": lambda a: sp.log(a) / sp.log(2),
    "sqrt": sp.sqrt, "cbrt": sp.cbrt,
    "abs": sp.Abs, "sign": sp.sign, "floor": sp.floor, "ceil": sp.ceiling,
    "min": sp.Min, "max": sp.Max, "mod": sp.Mod, "Heaviside": sp.Heaviside,
}
TRANSFORMS = standard_transformations + (
    convert_xor, implicit_multiplication, implicit_application, function_exponentiation,
)
_ALLOWED = re.compile(r"^[\w\s+\-*/^().,×÷−²³θπ]*$")


def parse_math(text: str, allowed=()) -> sp.Expr:
    s = text.strip()
    if not s:
        raise MathError("Empty expression")
    s = (s.replace("×", "*").replace("÷", "/").replace("−", "-")
         .replace("²", "**2").replace("³", "**3"))
    if "__" in s or not _ALLOWED.match(s):
        raise MathError("Illegal character in expression")
    try:
        expr = parse_expr(s, local_dict=dict(NAMESPACE), transformations=TRANSFORMS)
    except Exception as ex:  # noqa: BLE001
        raise MathError(f"Syntax error: {ex}") from None
    if not isinstance(expr, sp.Expr):
        raise MathError("Not a valid expression")
    bad = expr.free_symbols - set(allowed)
    if bad:
        raise MathError("Unknown symbol(s): " + ", ".join(sorted(map(str, bad))))
    return expr


def parse_number(text: str) -> float:
    try:
        v = parse_math(text, ())
        return float(v)
    except MathError:
        raise
    except Exception:  # noqa: BLE001
        raise MathError(f"'{text}' is not a number") from None


def parse_implicit(text: str):
    """'lhs = rhs' or just an expression (= 0). Returns (F, latex)."""
    if text.count("=") > 1:
        raise MathError("Use a single '='")
    if "=" in text:
        l, r = text.split("=")
        lhs, rhs = parse_math(l, (X, Y)), parse_math(r, (X, Y))
        return lhs - rhs, sp.latex(lhs) + " = " + sp.latex(rhs)
    e = parse_math(text, (X, Y))
    return e, sp.latex(e) + " = 0"


def eval_expr(expr, args, *arrays):
    f = sp.lambdify(args, expr, modules="numpy")
    shape = np.broadcast(*arrays).shape
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = np.asarray(f(*arrays))
            if np.iscomplexobj(out):
                out = np.where(np.abs(out.imag) <= 1e-9 * np.maximum(1, np.abs(out.real)),
                               out.real, np.nan)
            out = np.broadcast_to(out.astype(float), shape).copy()
        except Exception as ex:  # noqa: BLE001
            raise MathError(f"Cannot evaluate numerically: {ex}") from None
    out[~np.isfinite(out)] = np.nan
    return out


def break_discontinuities(y: np.ndarray) -> np.ndarray:
    """Insert NaN at poles (sign flip between two large values) so tan(x), 1/x don't draw vertical lines."""
    y = y.copy()
    a = np.abs(y)
    fin = a[np.isfinite(a)]
    if fin.size < 10:
        return y
    med = max(np.median(fin), 1e-12)
    with np.errstate(invalid="ignore"):
        flip = (y[:-1] * y[1:] < 0) & (a[:-1] > 3 * med) & (a[1:] > 3 * med)
    y[1:][flip] = np.nan
    return y


# =============================================================================
# Data model
# =============================================================================
LINESTYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
TABLE_STYLES = ["Markers", "Line", "Line + markers", "Step", "Stems", "Bars"]
MARKERS = ["o", "s", "^", "v", "D", "x", "+", "."]
TAB10 = [mpl.colors.to_hex(c) for c in mpl.colormaps["tab10"].colors]
LEGEND_LOCS = ["best", "upper right", "upper left", "lower left", "lower right",
               "center left", "center right", "lower center", "upper center", "center"]


@dataclass
class Series:
    kind: str                       # function | parametric | polar | implicit | table
    label: str = ""                 # empty = automatic LaTeX label
    color: str = "#1f77b4"
    linestyle: str = "-"
    linewidth: float = 2.0
    visible: bool = True
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = self.params
        if self.kind == "function":
            return "y = " + p["expr"] + ("   (+ f')" if p.get("deriv") else "")
        if self.kind == "parametric":
            return f"x = {p['xexpr']},  y = {p['yexpr']}"
        if self.kind == "polar":
            return "r = " + p["rexpr"]
        if self.kind == "implicit":
            return p["expr"]
        return f"table ({len(p['xs'])} points)"


@dataclass
class PlotSettings:
    title: str = ""
    xlabel: str = "x"
    ylabel: str = "y"
    font_size: int = 12
    grid: bool = True
    minor_grid: bool = False
    legend: bool = True
    legend_loc: str = "best"
    equal_aspect: bool = False
    axes_style: str = "Box"         # Box | Cross at origin
    xscale: str = "linear"
    yscale: str = "linear"
    xlim_auto: bool = True
    xmin: str = "-10"
    xmax: str = "10"
    ylim_auto: bool = True
    ymin: str = "-10"
    ymax: str = "10"
    style: str = "default"
    usetex: bool = False


@dataclass
class Curve:
    x: np.ndarray
    y: np.ndarray
    label: str
    role: str = "main"              # main | deriv | fit


def _tex(e) -> str:
    return sp.latex(e)


def poly_latex(c) -> str:
    n = len(c) - 1
    parts = []
    for i, a in enumerate(c):
        p = n - i
        if abs(a) < 1e-12:
            continue
        coef = f"{abs(a):.4g}"
        if p == 0:
            term = coef
        else:
            term = ("" if coef == "1" else coef) + ("x" if p == 1 else f"x^{{{p}}}")
        parts.append(("-" if a < 0 else "+", term))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sg, t in parts[1:]:
        s += f" {sg} {t}"
    return s


_CURVE_CACHE: dict = {}


def series_curves(s: Series) -> list[Curve]:
    """Compute numeric curves for function/parametric/polar/table series (cached)."""
    key = (s.kind, json.dumps(s.params, sort_keys=True))
    if key in _CURVE_CACHE:
        return _CURVE_CACHE[key]
    p = s.params
    out: list[Curve] = []
    if s.kind == "function":
        e = parse_math(p["expr"], (X,))
        a, b = parse_number(p["xmin"]), parse_number(p["xmax"])
        if not a < b:
            raise MathError("x min must be less than x max")
        xs = np.linspace(a, b, int(p["n"]))
        out.append(Curve(xs, break_discontinuities(eval_expr(e, (X,), xs)), f"$y = {_tex(e)}$"))
        if p.get("deriv"):
            d = sp.diff(e, X)
            out.append(Curve(xs, break_discontinuities(eval_expr(d, (X,), xs)),
                             f"$y' = {_tex(d)}$", "deriv"))
    elif s.kind == "parametric":
        ex, ey = parse_math(p["xexpr"], (T,)), parse_math(p["yexpr"], (T,))
        a, b = parse_number(p["tmin"]), parse_number(p["tmax"])
        if not a < b:
            raise MathError("t min must be less than t max")
        ts = np.linspace(a, b, int(p["n"]))
        out.append(Curve(eval_expr(ex, (T,), ts), eval_expr(ey, (T,), ts),
                         f"$x = {_tex(ex)},\\ y = {_tex(ey)}$"))
    elif s.kind == "polar":
        er = parse_math(p["rexpr"], (TH,))
        a, b = parse_number(p["tmin"]), parse_number(p["tmax"])
        if not a < b:
            raise MathError("θ min must be less than θ max")
        th = np.linspace(a, b, int(p["n"]))
        r = eval_expr(er, (TH,), th)
        out.append(Curve(r * np.cos(th), r * np.sin(th), f"$r = {_tex(er)}$"))
    elif s.kind == "table":
        xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
        out.append(Curve(xs, ys, f"data ({len(xs)} pts)"))
        deg = int(p.get("fit_degree", 0))
        if deg > 0:
            if len(xs) <= deg:
                raise MathError(f"Need more than {deg} points for a degree-{deg} fit")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = np.polyfit(xs, ys, deg)
            xf = np.linspace(xs.min(), xs.max(), 400)
            yf = np.polyval(c, xf)
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1 - float(np.sum((ys - np.polyval(c, xs)) ** 2)) / ss_tot if ss_tot > 0 else 1.0
            out.append(Curve(xf, yf, f"fit: $y = {poly_latex(c)}$  ($R^2 = {r2:.4f}$)", "fit"))
    if len(_CURVE_CACHE) > 64:
        _CURVE_CACHE.clear()
    _CURVE_CACHE[key] = out
    return out


def safe_text(s: str) -> str:
    """Return s if it is valid mathtext, else escape the dollars so it shows literally."""
    if "$" not in s:
        return s
    try:
        mathtext.MathTextParser("path").parse(s)
        return s
    except Exception:  # noqa: BLE001
        return s.replace("$", r"\$")


# =============================================================================
# Plot builder (works on ANY Figure -> used for screen and for export)
# =============================================================================
def apply_rc(st: PlotSettings):
    mpl.rcdefaults()
    if st.style != "default" and st.style in mpl.style.available:
        mpl.style.use(st.style)
    use_tex = bool(st.usetex and HAVE_REAL_LATEX)
    mpl.rcParams["text.usetex"] = use_tex
    if use_tex:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        mpl.rcParams["mathtext.fontset"] = "cm"   # Computer Modern look


def draw_plot(fig: Figure, series: list[Series], st: PlotSettings) -> list[str]:
    msgs: list[str] = []
    apply_rc(st)
    fig.clear()
    fig.set_layout_engine("constrained")
    fig.set_facecolor(mpl.rcParams["figure.facecolor"])
    ax = fig.add_subplot(111)
    fs = st.font_size
    y_samples = []
    y_other = []
    labelled = False

    for s in series:
        if not s.visible:
            continue
        try:
            if s.kind == "implicit":
                F, tex = parse_implicit(s.params["expr"])
                a, b = parse_number(s.params["xmin"]), parse_number(s.params["xmax"])
                c, d = parse_number(s.params["ymin"]), parse_number(s.params["ymax"])
                if not (a < b and c < d):
                    raise MathError("Invalid implicit ranges")
                n = int(s.params["res"])
                XX, YY = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
                Z = eval_expr(F, (X, Y), XX, YY)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.contour(XX, YY, Z, levels=[0], colors=[s.color],
                               linewidths=s.linewidth, linestyles=[s.linestyle])
                lab = safe_text(s.label or f"${tex}$")
                ax.plot([], [], color=s.color, ls=s.linestyle, lw=s.linewidth, label=lab)
                labelled = True
                continue

            curves = series_curves(s)
            for cv in curves:
                lab = safe_text(s.label if (s.label and cv.role == "main") else cv.label)
                labelled = True
                if s.kind == "table":
                    y_other.append(cv.y)
                if cv.role == "fit":
                    ax.plot(cv.x, cv.y, color=s.color, ls="--", lw=max(1.0, s.linewidth - 0.5),
                            alpha=0.9, label=lab)
                elif cv.role == "deriv":
                    ax.plot(cv.x, cv.y, color=s.color, ls="--", lw=s.linewidth,
                            alpha=0.65, label=lab)
                    y_samples.append(cv.y)
                elif s.kind == "table":
                    sty, mk = s.params.get("style", "Markers"), s.params.get("marker", "o")
                    if sty == "Markers":
                        ax.plot(cv.x, cv.y, ls="none", marker=mk, color=s.color, ms=6, label=lab)
                    elif sty == "Line":
                        ax.plot(cv.x, cv.y, color=s.color, ls=s.linestyle, lw=s.linewidth, label=lab)
                    elif sty == "Line + markers":
                        ax.plot(cv.x, cv.y, color=s.color, ls=s.linestyle, lw=s.linewidth,
                                marker=mk, ms=6, label=lab)
                    elif sty == "Step":
                        ax.step(cv.x, cv.y, where="mid", color=s.color, ls=s.linestyle,
                                lw=s.linewidth, label=lab)
                    elif sty == "Stems":
                        ml, sl, _ = ax.stem(cv.x, cv.y, basefmt=" ", label=lab)
                        ml.set_color(s.color)
                        sl.set_color(s.color)
                    else:  # Bars
                        dx = np.diff(np.sort(cv.x))
                        w = 0.8 * (dx[dx > 0].min() if (dx > 0).any() else 1.0)
                        ax.bar(cv.x, cv.y, width=w, color=s.color, alpha=0.8, label=lab)
                else:
                    ax.plot(cv.x, cv.y, color=s.color, ls=s.linestyle, lw=s.linewidth, label=lab)
                    if s.kind == "function":
                        y_samples.append(cv.y)
                    else:
                        y_other.append(cv.y)
        except MathError as ex:
            msgs.append(f"{s.summary()}: {ex}")

    # text
    ax.set_title(safe_text(st.title), fontsize=fs + 3)
    cross = st.axes_style.startswith("Cross")
    ax.set_xlabel(safe_text(st.xlabel), fontsize=fs, loc="right" if cross else "center")
    ax.set_ylabel(safe_text(st.ylabel), fontsize=fs, loc="top" if cross else "center")
    ax.tick_params(labelsize=max(6, fs - 2))

    # scales / limits
    try:
        ax.set_xscale(st.xscale)
        ax.set_yscale(st.yscale)
    except Exception as ex:  # noqa: BLE001
        msgs.append(f"Axis scale: {ex}")
    if not st.xlim_auto:
        try:
            ax.set_xlim(parse_number(st.xmin), parse_number(st.xmax))
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"X limits: {ex}")
    if not st.ylim_auto:
        try:
            ax.set_ylim(parse_number(st.ymin), parse_number(st.ymax))
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"Y limits: {ex}")
    elif y_samples:
        # robust autoscale: stops tan(x), 1/x from squashing the plot
        v = np.concatenate(y_samples)
        v = v[np.isfinite(v)]
        if v.size > 20:
            lo, hi = np.percentile(v, [5, 95])
            span = hi - lo
            if span > 0 and (v.max() - v.min()) > 20 * span:
                lo, hi = lo - 0.5 * span, hi + 0.5 * span
                o = np.concatenate(y_other) if y_other else np.array([])
                o = o[np.isfinite(o)]
                if o.size:                      # never clip table / parametric / polar data
                    lo, hi = min(lo, o.min()), max(hi, o.max())
                    pad = 0.05 * (hi - lo)
                    lo, hi = lo - pad, hi + pad
                ax.set_ylim(lo, hi)

    if st.equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if st.grid:
        ax.grid(True, which="major", alpha=0.45)
    if st.minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.18)
    if cross:
        ax.spines[["left", "bottom"]].set_position("zero")
        ax.spines[["top", "right"]].set_visible(False)
    if st.legend and labelled:
        ax.legend(loc=st.legend_loc, fontsize=max(6, fs - 2))
    return msgs



# =============================================================================
# Theme (Fusion + QSS)
# =============================================================================
THEMES = {
    "Dark": dict(bg="#171a21", panel="#1f232c", card="#2a2f3a", card_h="#343a48", border="#3a4152",
                 text="#e8ebf2", muted="#8f99ab", accent="#4f8cff", accent_h="#6ba0ff",
                 accent_text="#ffffff", num="#2f3544", op="#3b4256", var="#2b3a5c", fn="#26303f",
                 danger="#ff6b6b"),
    "Light": dict(bg="#eef1f6", panel="#ffffff", card="#f3f5f9", card_h="#e6eaf2", border="#d3d9e4",
                  text="#1d2330", muted="#667085", accent="#2f6df6", accent_h="#1f5ae0",
                  accent_text="#ffffff", num="#ffffff", op="#e3e8f1", var="#dce8ff", fn="#eef1f7",
                  danger="#d92d20"),
}


def make_arrows(c: dict) -> dict:
    """Write tiny SVG chevrons to a temp folder (QSS can only draw arrows from image files)."""
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "curvexa_icons")
    os.makedirs(d, exist_ok=True)
    out = {}
    for name, pts in (("up", "2,7 5,3 8,7"), ("down", "2,3 5,7 8,3")):
        for tone in ("muted", "accent"):
            path = os.path.join(d, f"{name}_{tone}_{c[tone][1:]}.svg").replace("\\", "/")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
                            f'<polyline points="{pts}" fill="none" stroke="{c[tone]}" stroke-width="1.6" '
                            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
            out[f"{name}_{tone}"] = path
    return out


def build_qss(c: dict) -> str:
    ar = make_arrows(c)
    return f"""
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", "Ubuntu", sans-serif; font-size: 13px; }}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QWidget {{ color: {c['text']}; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

QMenuBar {{ background: {c['bg']}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {c['card_h']}; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 8px; }}

QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 10px; background: {c['panel']}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {c['muted']}; padding: 8px 16px; margin-right: 2px;
                border: none; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QTabBar::tab:selected {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px;
    padding: 6px 9px; min-height: 20px; selection-background-color: {c['accent']};
    selection-color: {c['accent_text']}; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {c['muted']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({ar['down_muted']}); width: 10px; height: 10px; margin-right: 8px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;
    width: 24px; border: none; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
    width: 24px; border: none; background: transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({ar['up_muted']}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({ar['down_muted']}); width: 10px; height: 10px; }}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{ image: url({ar['up_accent']}); }}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{ image: url({ar['down_accent']}); }}
QComboBox QAbstractItemView {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 6px;
    selection-background-color: {c['accent']}; selection-color: {c['accent_text']}; outline: 0; padding: 4px; }}

QPushButton {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px;
               padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {c['card_h']}; border-color: {c['muted']}; }}
QPushButton:pressed {{ background: {c['border']}; }}
QPushButton[primary="true"] {{ background: {c['accent']}; color: {c['accent_text']}; border: 1px solid {c['accent']}; }}
QPushButton[primary="true"]:hover {{ background: {c['accent_h']}; border-color: {c['accent_h']}; }}
QPushButton[danger="true"]:hover {{ border-color: {c['danger']}; color: {c['danger']}; }}

QToolButton {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 2px; }}
QToolButton:hover {{ background: {c['card_h']}; }}
QToolButton:pressed {{ background: {c['accent']}; color: {c['accent_text']}; }}
QToolButton[kind="num"] {{ background: {c['num']}; }}
QToolButton[kind="op"]  {{ background: {c['op']}; }}
QToolButton[kind="var"] {{ background: {c['var']}; color: {c['accent']}; font-style: italic; }}
QToolButton[kind="fn"]  {{ background: {c['fn']}; font-size: 14px; }}
QToolButton[kind="ctl"] {{ background: {c['card_h']}; }}
QToolButton[kind="add"] {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; }}
QToolButton[kind="num"]:hover, QToolButton[kind="op"]:hover, QToolButton[kind="var"]:hover,
QToolButton[kind="fn"]:hover, QToolButton[kind="ctl"]:hover {{ border-color: {c['accent']}; }}
QToolButton[kind="add"]:hover {{ background: {c['accent_h']}; }}
QToolButton[kind]:pressed {{ background: {c['accent']}; color: {c['accent_text']}; }}

NavigationToolbar2QT {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 10px; spacing: 4px; padding: 3px; }}
NavigationToolbar2QT QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; }}
NavigationToolbar2QT QToolButton:hover {{ background: {c['card_h']}; }}
NavigationToolbar2QT QToolButton:checked {{ background: {c['accent']}; }}

QListWidget, QTableWidget {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 10px;
    padding: 4px; outline: 0; gridline-color: {c['border']}; alternate-background-color: {c['panel']}; }}
QListWidget::item {{ padding: 6px 6px; border-radius: 6px; }}
QListWidget::item:hover {{ background: {c['card_h']}; }}
QListWidget::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
QTableWidget::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
QHeaderView::section {{ background: {c['panel']}; color: {c['muted']}; border: none;
    border-bottom: 1px solid {c['border']}; padding: 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {c['panel']}; border: none; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
#preview {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 10px; }}
#preview QWidget {{ background: transparent; }}
#canvasFrame {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 12px; }}

QDockWidget {{ font-weight: 700; }}
QDockWidget::title {{ background: {c['panel']}; padding: 8px 12px; border-top-left-radius: 10px;
                      border-top-right-radius: 10px; border: 1px solid {c['border']}; border-bottom: none; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QMainWindow::separator {{ background: transparent; height: 8px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {c['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QCheckBox {{ spacing: 8px; }}
QStatusBar {{ background: {c['bg']}; }}
QStatusBar::item {{ border: none; }}
QLabel[muted="true"] {{ color: {c['muted']}; }}
"""


def apply_theme(app: QApplication, name: str):
    c = THEMES[name]
    app.setStyle("Fusion")
    p = QPalette()
    for role, key in [(QPalette.Window, "bg"), (QPalette.Base, "card"), (QPalette.AlternateBase, "panel"),
                      (QPalette.Button, "card"), (QPalette.ToolTipBase, "panel"), (QPalette.Highlight, "accent")]:
        p.setColor(role, QColor(c[key]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.ToolTipText):
        p.setColor(role, QColor(c["text"]))
    p.setColor(QPalette.HighlightedText, QColor(c["accent_text"]))
    p.setColor(QPalette.PlaceholderText, QColor(c["muted"]))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(c["muted"]))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(c["muted"]))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(c["muted"]))
    app.setPalette(p)
    app.setStyleSheet(build_qss(c))



# =============================================================================
# App icon (drawn in code -> no external file needed)
# =============================================================================
def render_icon(size: int) -> QPixmap:
    s = float(size)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    bg = QLinearGradient(0, 0, s, s)
    bg.setColorAt(0.0, QColor("#4f8cff"))
    bg.setColorAt(1.0, QColor("#8b4dff"))
    p.setBrush(bg)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.23, s * 0.23)
    # axes
    pen = QPen(QColor(255, 255, 255, 120), max(1.0, s * 0.035))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    axes = QPainterPath(QPointF(s * 0.22, s * 0.20))
    axes.lineTo(s * 0.22, s * 0.78)
    axes.lineTo(s * 0.82, s * 0.78)
    p.drawPath(axes)
    # the curve
    wave = QPainterPath(QPointF(s * 0.20, s * 0.64))
    wave.cubicTo(s * 0.30, s * 0.18, s * 0.45, s * 0.18, s * 0.52, s * 0.50)
    wave.cubicTo(s * 0.60, s * 0.82, s * 0.72, s * 0.82, s * 0.82, s * 0.32)
    pen = QPen(QColor("#ffffff"), max(1.5, s * 0.08))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(wave)
    # highlighted point on the curve
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#ffd166"))
    r = s * 0.065
    p.drawEllipse(QPointF(s * 0.82, s * 0.32), r, r)
    p.end()
    return pm


def make_app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(render_icon(size))
    return icon


# =============================================================================
# Small widgets
# =============================================================================
class LatexPreview(QScrollArea):
    """Live formula preview rendered with matplotlib mathtext."""

    def __init__(self):
        super().__init__()
        self.setObjectName("preview")
        self.setFixedHeight(72)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setTextFormat(Qt.RichText)
        self.setWidget(self.label)

    def set_tex(self, tex: str):
        try:
            color = self.palette().color(QPalette.WindowText).name()
            buf = io.BytesIO()
            with mpl.rc_context({"text.usetex": False, "mathtext.fontset": "cm"}):
                fig = Figure(figsize=(0.1, 0.1))
                fig.text(0, 0, f"${tex}$", fontsize=16, color=color)
                fig.savefig(buf, format="png", dpi=144, transparent=True,
                            bbox_inches="tight", pad_inches=0.04)
            pm = QPixmap()
            pm.loadFromData(buf.getvalue())
            pm.setDevicePixelRatio(1.5)
            self.label.setContentsMargins(12, 0, 12, 0)
            self.label.setPixmap(pm)
        except Exception:  # noqa: BLE001
            self.label.setText(f"<i>{tex}</i>")

    def set_error(self, msg: str):
        self.label.setContentsMargins(12, 0, 12, 0)
        self.label.setText(f"<span style='color:#ff6b6b'>{msg}</span>")

    def clear_preview(self):
        self.label.clear()


class ColorButton(QPushButton):
    changed = Signal()

    def __init__(self, color="#1f77b4"):
        super().__init__()
        self._c = color
        self.setFixedWidth(44)
        self.clicked.connect(self._pick)
        self._paint()

    def color(self) -> str:
        return self._c

    def set_color(self, c: str):
        self._c = c
        self._paint()

    def _paint(self):
        self.setStyleSheet(f"QPushButton {{ background-color:{self._c}; border:2px solid rgba(255,255,255,70); border-radius:8px; }}")

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._c), self, "Series colour")
        if c.isValid():
            self.set_color(c.name())
            self.changed.emit()


# =============================================================================
# Math keyboard
# =============================================================================
KEY_PAGES = {
    "Basic": [
        [("7", "7"), ("8", "8"), ("9", "9"), ("÷", "/"), ("(", "("), (")", ")")],
        [("4", "4"), ("5", "5"), ("6", "6"), ("×", "*"), ("xʸ", "^"), ("x²", "^2")],
        [("1", "1"), ("2", "2"), ("3", "3"), ("−", "-"), ("√", "sqrt()", 1), ("|x|", "abs()", 1)],
        [("0", "0"), (".", "."), (",", ","), ("+", "+"), ("a/b", "()/()", 4), ("1/x", "1/()", 1)],
        [("x", "x"), ("y", "y"), ("t", "t"), ("θ", "theta"), ("π", "pi"), ("e", "e")],
    ],
    "Trig": [
        [("sin", "sin()", 1), ("cos", "cos()", 1), ("tan", "tan()", 1), ("sec", "sec()", 1), ("csc", "csc()", 1), ("cot", "cot()", 1)],
        [("asin", "asin()", 1), ("acos", "acos()", 1), ("atan", "atan()", 1), ("atan2", "atan2(,)", 2), ("sin²", "sin()^2", 3), ("cos²", "cos()^2", 3)],
        [("sinh", "sinh()", 1), ("cosh", "cosh()", 1), ("tanh", "tanh()", 1), ("asinh", "asinh()", 1), ("acosh", "acosh()", 1), ("atanh", "atanh()", 1)],
    ],
    "Exp / Log / Other": [
        [("exp", "exp()", 1), ("ln", "ln()", 1), ("log₁₀", "log()", 1), ("log₂", "log2()", 1), ("logₐb", "log(,)", 2), ("∛", "cbrt()", 1)],
        [("eˣ", "e^()", 1), ("10ˣ", "10^()", 1), ("2ˣ", "2^()", 1), ("ⁿ√x", "()^(1/())", 8), ("x³", "^3", 0), ("x⁻¹", "^(-1)", 0)],
        [("sign", "sign()", 1), ("floor", "floor()", 1), ("ceil", "ceil()", 1), ("min", "min(,)", 2), ("max", "max(,)", 2), ("mod", "mod(,)", 2)],
        [("step", "Heaviside()", 1), ("abs", "abs()", 1)],
    ],
}


class MathKeyboard(QWidget):
    enter_pressed = Signal()

    def __init__(self):
        super().__init__()
        self.target: QLineEdit | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(12)
        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for name, rows in KEY_PAGES.items():
            page = QWidget()
            g = QGridLayout(page)
            g.setSpacing(6)
            g.setContentsMargins(10, 10, 10, 10)
            ncols = max(len(r) for r in rows)
            for c in range(ncols):
                g.setColumnStretch(c, 1)           # every column shares the full width equally
            for r, row in enumerate(rows):
                for c, key in enumerate(row):
                    label, text = key[0], key[1]
                    back = key[2] if len(key) > 2 else 0
                    tip = f"inserts: {text}" + ("   (log = base 10, ln = natural)" if label.startswith("log") or label == "ln" else "")
                    b = self._btn(label, tip, self._kind(label, text))
                    b.clicked.connect(lambda _=False, t=text, k=back: self.insert(t, k))
                    g.addWidget(b, r, c)
            g.setRowStretch(len(rows), 1)          # keep rows compact on pages with few rows
            tabs.addTab(page, name)
        lay.addWidget(tabs, 1)

        ctl = QGridLayout()
        ctl.setSpacing(6)
        ctl.setColumnStretch(0, 1)
        ctl.setColumnStretch(1, 1)
        for i, (lab, fn, tip) in enumerate([("◀", lambda: self._move(-1), "Cursor left"),
                                            ("▶", lambda: self._move(1), "Cursor right"),
                                            ("⌫", self._bksp, "Backspace"),
                                            ("AC", self._clear, "Clear field")]):
            b = self._btn(lab, tip, "ctl")
            b.clicked.connect(fn)
            ctl.addWidget(b, i // 2, i % 2)
        add = self._btn("Add ⏎", "Add the current input as a new series", "add")
        add.clicked.connect(self.enter_pressed.emit)
        ctl.addWidget(add, 2, 0, 1, 2)
        ctl.setRowStretch(3, 1)
        side = QWidget()
        side.setLayout(ctl)
        side.setMinimumWidth(190)
        side.setMaximumWidth(260)
        lay.addWidget(side)

    @staticmethod
    def _kind(label: str, text: str) -> str:
        if text.isdigit() or text in (".", ","):
            return "num"
        if text in ("+", "-", "*", "/", "^", "^2", "(", ")"):
            return "op"
        if label in ("x", "y", "t", "θ", "π", "e"):
            return "var"
        return "fn"

    @staticmethod
    def _btn(text, tip="", kind="fn"):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setProperty("kind", kind)
        b.setFocusPolicy(Qt.NoFocus)          # keep the caret in the expression field
        b.setMinimumHeight(32)
        b.setMaximumHeight(52)
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        f = b.font(); f.setPointSize(11); f.setBold(True); b.setFont(f)
        return b

    def insert(self, text: str, back: int = 0):
        e = self.target
        if e is None:
            return
        e.insert(text)
        e.setCursorPosition(e.cursorPosition() - back)
        e.setFocus()

    def _move(self, d):
        if self.target:
            self.target.setCursorPosition(self.target.cursorPosition() + d)
            self.target.setFocus()

    def _bksp(self):
        if self.target:
            self.target.backspace()
            self.target.setFocus()

    def _clear(self):
        if self.target:
            self.target.clear()
            self.target.setFocus()


# =============================================================================
# Input tabs
# =============================================================================
class BaseTab(QWidget):
    kind = ""
    changed = Signal()

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.preview = LatexPreview()
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(12, 12, 12, 12)
        self.root.setSpacing(8)
        self.form.setVerticalSpacing(10)
        cap = QLabel("PREVIEW")
        cap.setProperty("muted", True)
        self.root.addLayout(self.form)
        self.root.addWidget(cap)
        self.root.addWidget(self.preview)
        self.root.addStretch(1)

    def math_edit(self, text="", placeholder=""):
        e = QLineEdit(text)
        e.setPlaceholderText(placeholder)
        self.main.register_math_edit(e)
        e.textChanged.connect(self.refresh_preview)
        return e

    @staticmethod
    def spin(lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(max(1, val // 10))
        return s

    def refresh_preview(self):
        pass

    def read(self) -> dict:
        raise NotImplementedError

    def load(self, p: dict):
        raise NotImplementedError


class FunctionTab(BaseTab):
    kind = "function"

    def __init__(self, main):
        super().__init__(main)
        self.expr = self.math_edit("sin(x)/x", "e.g. sin(x)/x   or   x^2 - 3x + 1   or   exp(-x^2)")
        self.xmin, self.xmax = self.math_edit("-10"), self.math_edit("10")
        self.n = self.spin(10, 200000, 2000)
        self.deriv = QCheckBox("also plot the derivative f'(x)")
        self.form.addRow("y = f(x) =", self.expr)
        r = QHBoxLayout()
        r.addWidget(self.xmin)
        r.addWidget(QLabel("to"))
        r.addWidget(self.xmax)
        self.form.addRow("x from", r)
        self.form.addRow("Samples", self.n)
        self.form.addRow("", self.deriv)
        self.refresh_preview()

    def refresh_preview(self):
        t = self.expr.text()
        if not t.strip():
            return self.preview.clear_preview()
        try:
            self.preview.set_tex("y = " + sp.latex(parse_math(t, (X,))))
        except MathError as ex:
            self.preview.set_error(str(ex))

    def read(self):
        parse_math(self.expr.text(), (X,))
        a, b = parse_number(self.xmin.text()), parse_number(self.xmax.text())
        if not a < b:
            raise MathError("x min must be less than x max")
        return dict(expr=self.expr.text().strip(), xmin=self.xmin.text(), xmax=self.xmax.text(),
                    n=self.n.value(), deriv=self.deriv.isChecked())

    def load(self, p):
        self.expr.setText(p["expr"]); self.xmin.setText(p["xmin"]); self.xmax.setText(p["xmax"])
        self.n.setValue(p["n"]); self.deriv.setChecked(p.get("deriv", False))


class ParametricTab(BaseTab):
    kind = "parametric"

    def __init__(self, main):
        super().__init__(main)
        self.xe = self.math_edit("cos(t)", "x(t)")
        self.ye = self.math_edit("sin(3t)", "y(t)")
        self.tmin, self.tmax = self.math_edit("0"), self.math_edit("2*pi")
        self.n = self.spin(10, 200000, 2000)
        self.form.addRow("x(t) =", self.xe)
        self.form.addRow("y(t) =", self.ye)
        r = QHBoxLayout()
        r.addWidget(self.tmin); r.addWidget(QLabel("to")); r.addWidget(self.tmax)
        self.form.addRow("t from", r)
        self.form.addRow("Samples", self.n)
        self.refresh_preview()

    def refresh_preview(self):
        try:
            ex, ey = parse_math(self.xe.text(), (T,)), parse_math(self.ye.text(), (T,))
            self.preview.set_tex(f"x(t) = {sp.latex(ex)},\\quad y(t) = {sp.latex(ey)}")
        except MathError as ex:
            self.preview.set_error(str(ex)) if (self.xe.text() or self.ye.text()) else self.preview.clear_preview()

    def read(self):
        parse_math(self.xe.text(), (T,)); parse_math(self.ye.text(), (T,))
        if not parse_number(self.tmin.text()) < parse_number(self.tmax.text()):
            raise MathError("t min must be less than t max")
        return dict(xexpr=self.xe.text().strip(), yexpr=self.ye.text().strip(),
                    tmin=self.tmin.text(), tmax=self.tmax.text(), n=self.n.value())

    def load(self, p):
        self.xe.setText(p["xexpr"]); self.ye.setText(p["yexpr"])
        self.tmin.setText(p["tmin"]); self.tmax.setText(p["tmax"]); self.n.setValue(p["n"])


class PolarTab(BaseTab):
    kind = "polar"

    def __init__(self, main):
        super().__init__(main)
        self.re = self.math_edit("1 + cos(theta)", "r(theta)  - use 'theta' or θ")
        self.tmin, self.tmax = self.math_edit("0"), self.math_edit("2*pi")
        self.n = self.spin(10, 200000, 2000)
        self.form.addRow("r(θ) =", self.re)
        r = QHBoxLayout()
        r.addWidget(self.tmin); r.addWidget(QLabel("to")); r.addWidget(self.tmax)
        self.form.addRow("θ from", r)
        self.form.addRow("Samples", self.n)
        self.refresh_preview()

    def refresh_preview(self):
        t = self.re.text()
        if not t.strip():
            return self.preview.clear_preview()
        try:
            self.preview.set_tex(r"r(\theta) = " + sp.latex(parse_math(t, (TH,))))
        except MathError as ex:
            self.preview.set_error(str(ex))

    def read(self):
        parse_math(self.re.text(), (TH,))
        if not parse_number(self.tmin.text()) < parse_number(self.tmax.text()):
            raise MathError("θ min must be less than θ max")
        return dict(rexpr=self.re.text().strip(), tmin=self.tmin.text(), tmax=self.tmax.text(),
                    n=self.n.value())

    def load(self, p):
        self.re.setText(p["rexpr"]); self.tmin.setText(p["tmin"]); self.tmax.setText(p["tmax"])
        self.n.setValue(p["n"])


class ImplicitTab(BaseTab):
    kind = "implicit"

    def __init__(self, main):
        super().__init__(main)
        self.ex = self.math_edit("x^2 + y^2 = 4", "e.g. x^2 + y^2 = 4   or   x^3 + y^3 = 3xy")
        self.xmin, self.xmax = self.math_edit("-5"), self.math_edit("5")
        self.ymin, self.ymax = self.math_edit("-5"), self.math_edit("5")
        self.res = self.spin(50, 3000, 800)
        self.form.addRow("F(x, y) =", self.ex)
        r = QHBoxLayout(); r.addWidget(self.xmin); r.addWidget(QLabel("to")); r.addWidget(self.xmax)
        self.form.addRow("x from", r)
        r = QHBoxLayout(); r.addWidget(self.ymin); r.addWidget(QLabel("to")); r.addWidget(self.ymax)
        self.form.addRow("y from", r)
        self.form.addRow("Grid resolution", self.res)
        self.refresh_preview()

    def refresh_preview(self):
        t = self.ex.text()
        if not t.strip():
            return self.preview.clear_preview()
        try:
            self.preview.set_tex(parse_implicit(t)[1])
        except MathError as ex:
            self.preview.set_error(str(ex))

    def read(self):
        parse_implicit(self.ex.text())
        if not (parse_number(self.xmin.text()) < parse_number(self.xmax.text())
                and parse_number(self.ymin.text()) < parse_number(self.ymax.text())):
            raise MathError("Invalid ranges")
        return dict(expr=self.ex.text().strip(), xmin=self.xmin.text(), xmax=self.xmax.text(),
                    ymin=self.ymin.text(), ymax=self.ymax.text(), res=self.res.value())

    def load(self, p):
        self.ex.setText(p["expr"]); self.xmin.setText(p["xmin"]); self.xmax.setText(p["xmax"])
        self.ymin.setText(p["ymin"]); self.ymax.setText(p["ymax"]); self.res.setValue(p["res"])


class TableTab(BaseTab):
    kind = "table"

    def __init__(self, main):
        super().__init__(main)
        self.root.removeWidget(self.preview)
        self.preview.hide()
        for i in reversed(range(self.root.count())):       # drop 'Preview' label + stretch
            it = self.root.itemAt(i)
            w = it.widget()
            if (w is not None and isinstance(w, QLabel)) or it.spacerItem():
                self.root.takeAt(i)
                if w:
                    w.hide()
        self.table = QTableWidget(8, 2)
        self.table.setHorizontalHeaderLabels(["x", "y"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(170)
        self.root.addWidget(self.table, 1)

        row = QHBoxLayout()
        for text, fn in [("+ Row", self.add_row), ("− Row", self.del_rows), ("Clear", self.clear),
                         ("Paste", self.paste), ("Import CSV…", self.import_csv), ("Sort by x", self.sort_x)]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        self.root.addLayout(row)

        fill = QHBoxLayout()
        self.f0, self.fstep = QLineEdit("0"), QLineEdit("1")
        self.fn = QSpinBox(); self.fn.setRange(1, 100000); self.fn.setValue(10)
        for e in (self.f0, self.fstep):
            main.register_math_edit(e)
        b = QPushButton("Fill x"); b.clicked.connect(self.fill_x)
        for w in (QLabel("start"), self.f0, QLabel("step"), self.fstep, QLabel("count"), self.fn, b):
            fill.addWidget(w)
        self.root.addLayout(fill)

        opt = QFormLayout()
        self.style = QComboBox(); self.style.addItems(TABLE_STYLES)
        self.marker = QComboBox(); self.marker.addItems(MARKERS)
        self.fit = QSpinBox(); self.fit.setRange(0, 15)
        self.fit.setSpecialValueText("off")
        opt.addRow("Draw as", self.style)
        opt.addRow("Marker", self.marker)
        opt.addRow("Polynomial fit degree", self.fit)
        self.root.addLayout(opt)

    # table helpers
    def add_row(self):
        self.table.insertRow(self.table.rowCount())

    def del_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def clear(self):
        self.table.setRowCount(0)
        self.table.setRowCount(8)

    def set_data(self, xs, ys):
        self.table.setRowCount(max(len(xs), 8))
        for i in range(self.table.rowCount()):
            for c, arr in enumerate((xs, ys)):
                self.table.setItem(i, c, QTableWidgetItem(f"{arr[i]:.10g}" if i < len(arr) else ""))

    def paste(self):
        text = QApplication.clipboard().text()
        xs, ys = [], []
        for line in text.splitlines():
            parts = [p for p in re.split(r"[,\t;\s]+", line.strip()) if p]
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0])); ys.append(float(parts[1]))
                except ValueError:
                    pass
        if not xs:
            return QMessageBox.information(self, "Paste", "Clipboard has no 'x y' numeric rows.")
        self.set_data(xs, ys)

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV / text (*.csv *.txt *.dat);;All (*)")
        if not path:
            return
        xs, ys = [], []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f, delimiter=","):
                    if len(row) == 1:
                        row = re.split(r"[;\t\s]+", row[0].strip())
                    try:
                        xs.append(float(row[0])); ys.append(float(row[1]))
                    except (ValueError, IndexError):
                        continue            # header / junk line
        except OSError as ex:
            return QMessageBox.warning(self, "Import CSV", str(ex))
        if not xs:
            return QMessageBox.information(self, "Import CSV", "No numeric 'x,y' rows found.")
        self.set_data(xs, ys)

    def sort_x(self):
        try:
            xs, ys = self.collect()
        except MathError as ex:
            return QMessageBox.warning(self, "Sort", str(ex))
        o = np.argsort(xs)
        self.set_data(list(np.array(xs)[o]), list(np.array(ys)[o]))

    def fill_x(self):
        try:
            a, d, n = parse_number(self.f0.text()), parse_number(self.fstep.text()), self.fn.value()
        except MathError as ex:
            return QMessageBox.warning(self, "Fill x", str(ex))
        ys = []
        for i in range(max(n, self.table.rowCount())):
            it = self.table.item(i, 1) if i < self.table.rowCount() else None
            ys.append(it.text() if it else "")
        self.table.setRowCount(max(n, 8))
        for i in range(self.table.rowCount()):
            self.table.setItem(i, 0, QTableWidgetItem(f"{a + i * d:.10g}" if i < n else ""))

    def collect(self):
        xs, ys = [], []
        for r in range(self.table.rowCount()):
            tx = self.table.item(r, 0).text().strip() if self.table.item(r, 0) else ""
            ty = self.table.item(r, 1).text().strip() if self.table.item(r, 1) else ""
            if not tx and not ty:
                continue
            try:
                xs.append(parse_number(tx)); ys.append(parse_number(ty))
            except MathError:
                raise MathError(f"Row {r + 1}: both x and y must be numbers (pi, 1/3, 2*pi are fine)") from None
        if len(xs) < 2:
            raise MathError("Enter at least 2 complete rows")
        return xs, ys

    def read(self):
        xs, ys = self.collect()
        return dict(xs=xs, ys=ys, style=self.style.currentText(), marker=self.marker.currentText(),
                    fit_degree=self.fit.value())

    def load(self, p):
        self.set_data(p["xs"], p["ys"])
        self.style.setCurrentText(p.get("style", "Markers"))
        self.marker.setCurrentText(p.get("marker", "o"))
        self.fit.setValue(p.get("fit_degree", 0))


# =============================================================================
# Export dialog
# =============================================================================
EXPORT_FILTERS = {
    "PNG image (*.png)": ".png", "JPEG image (*.jpg)": ".jpg", "SVG vector (*.svg)": ".svg",
    "PDF vector (*.pdf)": ".pdf", "TIFF image (*.tiff)": ".tiff",
}


class ExportDialog(QDialog):
    def __init__(self, parent, w_in, h_in):
        super().__init__(parent)
        self.setWindowTitle("Export image")
        f = QFormLayout(self)
        self.w = QDoubleSpinBox(); self.w.setRange(1, 60); self.w.setValue(round(w_in, 2)); self.w.setSuffix(" in")
        self.h = QDoubleSpinBox(); self.h.setRange(1, 60); self.h.setValue(round(h_in, 2)); self.h.setSuffix(" in")
        self.dpi = QSpinBox(); self.dpi.setRange(50, 1200); self.dpi.setValue(300)
        self.transparent = QCheckBox("Transparent background (not for JPEG)")
        self.tight = QCheckBox("Crop tightly around the plot"); self.tight.setChecked(True)
        self.px = QLabel()
        for wd in (self.w, self.h, self.dpi):
            wd.valueChanged.connect(self._px)
        f.addRow("Width", self.w); f.addRow("Height", self.h); f.addRow("DPI", self.dpi)
        f.addRow("Pixels", self.px); f.addRow(self.transparent); f.addRow(self.tight)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._px()

    def _px(self):
        self.px.setText(f"{int(self.w.value() * self.dpi.value())} × {int(self.h.value() * self.dpi.value())} px (raster formats)")


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Curvexa")
        self.setWindowIcon(make_app_icon())
        self.resize(1400, 860)
        self.series: list[Series] = []
        self.st = PlotSettings()
        self.math_edits: set[QLineEdit] = set()
        self._color_i = 0
        self._loading = False

        self.fig = Figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right = QWidget()
        self.right_layout = rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 8, 8, 0)
        rl.setSpacing(8)
        rl.addWidget(self.toolbar)
        frame = QWidget()
        frame.setObjectName("canvasFrame")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(6, 6, 6, 6)
        fl.addWidget(self.canvas)
        rl.addWidget(frame, 1)
        self.theme_name = "Dark"
        self.st.style = "dark_background"

        # ---- left: Series tab + Plot settings tab
        self.left_tabs = QTabWidget()
        self.left_tabs.setContentsMargins(0, 0, 0, 0)
        self.left_tabs.addTab(self._scrollable(self._build_series_tab(), 560), "Series")
        self.left_tabs.addTab(self._scrollable(self._build_settings_tab(), 520), "Plot settings")

        split = QSplitter()
        split.addWidget(self.left_tabs)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([540, 860])
        split.setChildrenCollapsible(False)
        split.setContentsMargins(8, 0, 0, 0)
        self.setCentralWidget(split)

        # ---- keyboard dock
        self.keyboard = MathKeyboard()
        self.keyboard.enter_pressed.connect(self.add_series)
        self.dock = QDockWidget("Math keyboard", self)
        self.dock.setWidget(self.keyboard)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.keyboard.target = self.tab_function.expr
        QApplication.instance().focusChanged.connect(self._focus_changed)

        self.redraw_timer = QTimer(self, singleShot=True, interval=250)
        self.redraw_timer.timeout.connect(self.redraw)

        self.status_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.status_lbl, 1)
        self._build_menu()
        self.sync_settings_to_ui()
        self.redraw()

    # ------------------------------------------------------------------ UI build
    def register_math_edit(self, e: QLineEdit):
        self.math_edits.add(e)
        e.returnPressed.connect(self.add_series)

    def _focus_changed(self, _old, new):
        if isinstance(new, QLineEdit) and new in self.math_edits:
            self.keyboard.target = new

    @staticmethod
    def _scrollable(inner: QWidget, min_h: int) -> QScrollArea:
        """Wrap a pane so a short window scrolls instead of squashing/overlapping the controls."""
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        return sa

    def _build_series_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.input_tabs = QTabWidget()
        self.tab_function = FunctionTab(self)
        self.tab_param = ParametricTab(self)
        self.tab_polar = PolarTab(self)
        self.tab_implicit = ImplicitTab(self)
        self.tab_table = TableTab(self)
        self.tabs_by_kind = {t.kind: t for t in (self.tab_function, self.tab_param, self.tab_polar,
                                                 self.tab_implicit, self.tab_table)}
        for t, name in [(self.tab_function, "y = f(x)"), (self.tab_param, "Parametric"),
                        (self.tab_polar, "Polar"), (self.tab_implicit, "Implicit"),
                        (self.tab_table, "Table")]:
            self.input_tabs.addTab(t, name)
        self.input_tabs.currentChanged.connect(self._tab_changed)
        lay.addWidget(self.input_tabs, 1)

        # style row
        sr = QHBoxLayout()
        sr.setSpacing(8)
        self.lab_edit = QLineEdit()
        self.lab_edit.setPlaceholderText("Legend label (optional, LaTeX ok: $\\alpha x$)")
        self.color_btn = ColorButton(TAB10[0])
        self.color_btn.setFixedSize(40, 34)
        self.ls_combo = QComboBox(); self.ls_combo.addItems(LINESTYLES)
        self.lw_spin = QDoubleSpinBox(); self.lw_spin.setRange(0.5, 10); self.lw_spin.setValue(2.0)
        self.lw_spin.setSingleStep(0.5)
        for wd in (self.lab_edit, self.color_btn, self.ls_combo, self.lw_spin):
            sr.addWidget(wd)
        sr.setStretch(0, 1)
        lay.addLayout(sr)

        br = QHBoxLayout()
        for text, fn in [("Add series", self.add_series), ("Update selected", self.update_series),
                         ("Edit selected", self.load_selected), ("Remove", self.remove_series)]:
            b = QPushButton(text); b.clicked.connect(fn); br.addWidget(b)
            if text == "Add series":
                b.setProperty("primary", True)
            if text == "Remove":
                b.setProperty("danger", True)
        br.setSpacing(8)
        lay.addLayout(br)

        self.list = QListWidget()
        self.list.setMinimumHeight(120)
        self.list.setSpacing(2)
        self.list.itemChanged.connect(self._item_toggled)
        self.list.itemDoubleClicked.connect(lambda _i: self.load_selected())
        lay.addWidget(self.list, 1)
        return w

    def _build_settings_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(14, 14, 14, 14)
        f.setVerticalSpacing(10)
        hint = QLabel("Math goes between dollar signs, e.g.  <b>$\\omega\\ (\\mathrm{rad/s})$</b>")
        hint.setWordWrap(True)
        f.addRow(hint)
        self.e_title, self.e_xl, self.e_yl = QLineEdit(), QLineEdit(), QLineEdit()
        self.s_font = QSpinBox(); self.s_font.setRange(6, 40)
        f.addRow("Graph title", self.e_title)
        f.addRow("X-axis label", self.e_xl)
        f.addRow("Y-axis label", self.e_yl)
        f.addRow("Font size", self.s_font)

        def lim_row():
            auto = QCheckBox("auto")
            lo, hi = QLineEdit(), QLineEdit()
            self.register_math_edit(lo); self.register_math_edit(hi)
            r = QHBoxLayout()
            for x in (auto, lo, QLabel("to"), hi):
                r.addWidget(x)
            return auto, lo, hi, r
        self.xa, self.xlo, self.xhi, r = lim_row(); f.addRow("X limits", r)
        self.ya, self.ylo, self.yhi, r = lim_row(); f.addRow("Y limits", r)

        self.c_xs, self.c_ys = QComboBox(), QComboBox()
        for c in (self.c_xs, self.c_ys):
            c.addItems(["linear", "log", "symlog"])
        f.addRow("X scale", self.c_xs); f.addRow("Y scale", self.c_ys)
        self.k_grid, self.k_mgrid = QCheckBox("Grid"), QCheckBox("Minor grid")
        self.k_leg = QCheckBox("Legend"); self.c_loc = QComboBox(); self.c_loc.addItems(LEGEND_LOCS)
        self.k_eq = QCheckBox("Equal aspect ratio (circles look round)")
        self.c_axes = QComboBox(); self.c_axes.addItems(["Box", "Cross at origin"])
        r = QHBoxLayout(); r.addWidget(self.k_grid); r.addWidget(self.k_mgrid); f.addRow(r)
        r = QHBoxLayout(); r.addWidget(self.k_leg); r.addWidget(self.c_loc); f.addRow(r)
        f.addRow(self.k_eq)
        f.addRow("Axes style", self.c_axes)
        self.c_style = QComboBox()
        wanted = ["default", "ggplot", "bmh", "classic", "dark_background", "fivethirtyeight",
                  "seaborn-v0_8-whitegrid", "grayscale"]
        self.c_style.addItems([s for s in wanted if s == "default" or s in mpl.style.available])
        f.addRow("Plot style", self.c_style)
        self.k_tex = QCheckBox("Use real compiled LaTeX (slower)")
        if not HAVE_REAL_LATEX:
            self.k_tex.setEnabled(False)
            self.k_tex.setToolTip("Needs latex + dvipng + ghostscript on PATH. Using matplotlib's "
                                  "built-in LaTeX renderer (Computer Modern) instead.")
            self.k_tex.setText("Real compiled LaTeX (latex/dvipng/ghostscript not found)")
        f.addRow(self.k_tex)

        for wd in (self.e_title, self.e_xl, self.e_yl, self.xlo, self.xhi, self.ylo, self.yhi):
            wd.textChanged.connect(self.settings_changed)
        for wd in (self.xa, self.ya, self.k_grid, self.k_mgrid, self.k_leg, self.k_eq, self.k_tex):
            wd.toggled.connect(self.settings_changed)
        for wd in (self.c_xs, self.c_ys, self.c_loc, self.c_axes, self.c_style):
            wd.currentTextChanged.connect(self.settings_changed)
        self.s_font.valueChanged.connect(self.settings_changed)
        return w

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")

        def act(menu, text, fn, key=None):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            menu.addAction(a)
            return a
        act(fm, "&New", self.new_project, "Ctrl+N")
        act(fm, "&Open project…", self.open_project, "Ctrl+O")
        act(fm, "&Save project…", self.save_project, "Ctrl+S")
        fm.addSeparator()
        act(fm, "&Export image…", self.export_image, "Ctrl+E")
        act(fm, "Copy image to clipboard", self.copy_image, "Ctrl+Shift+C")
        act(fm, "Export plotted data (CSV)…", self.export_csv)
        fm.addSeparator()
        act(fm, "&Quit", self.close, "Ctrl+Q")
        vm = mb.addMenu("&View")
        vm.addAction(self.dock.toggleViewAction())
        vm.addSeparator()
        for name in THEMES:
            act(vm, f"{name} theme", lambda _=False, n=name: self.set_theme(n))
        hm = mb.addMenu("&Help")
        act(hm, "Load demo plots", self.load_demo)
        act(hm, "Syntax help", self.syntax_help)
        act(hm, "About Curvexa", self.about)

    def set_theme(self, name: str):
        self.theme_name = name
        apply_theme(QApplication.instance(), name)
        if self.st.style in ("default", "dark_background"):      # keep the graph in tune with the app
            self.st.style = "dark_background" if name == "Dark" else "default"
            self.sync_settings_to_ui()
            self.redraw()
        # matplotlib recolours toolbar icons at creation time -> rebuild it
        new_tb = NavigationToolbar2QT(self.canvas, self)
        self.right_layout.replaceWidget(self.toolbar, new_tb)
        self.toolbar.deleteLater()
        self.toolbar = new_tb
        for t in self.tabs_by_kind.values():
            t.refresh_preview()

    # ------------------------------------------------------------------ settings <-> UI
    def sync_settings_to_ui(self):
        self._loading = True
        s = self.st
        self.e_title.setText(s.title); self.e_xl.setText(s.xlabel); self.e_yl.setText(s.ylabel)
        self.s_font.setValue(s.font_size)
        self.xa.setChecked(s.xlim_auto); self.xlo.setText(s.xmin); self.xhi.setText(s.xmax)
        self.ya.setChecked(s.ylim_auto); self.ylo.setText(s.ymin); self.yhi.setText(s.ymax)
        self.c_xs.setCurrentText(s.xscale); self.c_ys.setCurrentText(s.yscale)
        self.k_grid.setChecked(s.grid); self.k_mgrid.setChecked(s.minor_grid)
        self.k_leg.setChecked(s.legend); self.c_loc.setCurrentText(s.legend_loc)
        self.k_eq.setChecked(s.equal_aspect); self.c_axes.setCurrentText(s.axes_style)
        self.c_style.setCurrentText(s.style); self.k_tex.setChecked(s.usetex and HAVE_REAL_LATEX)
        self._loading = False

    def settings_changed(self, *_):
        if self._loading:
            return
        s = self.st
        s.title, s.xlabel, s.ylabel = self.e_title.text(), self.e_xl.text(), self.e_yl.text()
        s.font_size = self.s_font.value()
        s.xlim_auto, s.xmin, s.xmax = self.xa.isChecked(), self.xlo.text(), self.xhi.text()
        s.ylim_auto, s.ymin, s.ymax = self.ya.isChecked(), self.ylo.text(), self.yhi.text()
        s.xscale, s.yscale = self.c_xs.currentText(), self.c_ys.currentText()
        s.grid, s.minor_grid = self.k_grid.isChecked(), self.k_mgrid.isChecked()
        s.legend, s.legend_loc = self.k_leg.isChecked(), self.c_loc.currentText()
        s.equal_aspect, s.axes_style = self.k_eq.isChecked(), self.c_axes.currentText()
        s.style, s.usetex = self.c_style.currentText(), self.k_tex.isChecked()
        self.redraw_timer.start()        # debounce (typing "$\alpha$" passes through invalid states)

    # ------------------------------------------------------------------ series management
    def _style_from_ui(self):
        return dict(label=self.lab_edit.text().strip(), color=self.color_btn.color(),
                    linestyle=LINESTYLES[self.ls_combo.currentText()], linewidth=self.lw_spin.value())

    def _current_tab(self) -> BaseTab:
        return self.input_tabs.currentWidget()

    def _tab_changed(self, _i):
        t = self._current_tab()
        if t.kind == "table":
            return
        for e in self.math_edits:
            if t.isAncestorOf(e):
                self.keyboard.target = e
                break

    def _read_current(self) -> Series:
        tab = self._current_tab()
        try:
            params = tab.read()
        except MathError as ex:
            raise MathError(str(ex)) from None
        return Series(kind=tab.kind, params=params, **self._style_from_ui())

    def add_series(self):
        try:
            s = self._read_current()
        except MathError as ex:
            self.flash(str(ex), True)
            return
        self.series.append(s)
        self._color_i += 1
        self.color_btn.set_color(TAB10[self._color_i % len(TAB10)])
        self.lab_edit.clear()
        self.refresh_list()
        self.list.setCurrentRow(len(self.series) - 1)
        self.redraw()

    def update_series(self):
        r = self.list.currentRow()
        if r < 0:
            return self.flash("Select a series in the list first", True)
        try:
            s = self._read_current()
        except MathError as ex:
            return self.flash(str(ex), True)
        s.visible = self.series[r].visible
        self.series[r] = s
        self.refresh_list()
        self.list.setCurrentRow(r)
        self.redraw()

    def load_selected(self):
        r = self.list.currentRow()
        if r < 0:
            return
        s = self.series[r]
        tab = self.tabs_by_kind[s.kind]
        self.input_tabs.setCurrentWidget(tab)
        tab.load(s.params)
        self.lab_edit.setText(s.label)
        self.color_btn.set_color(s.color)
        self.ls_combo.setCurrentText(next(k for k, v in LINESTYLES.items() if v == s.linestyle))
        self.lw_spin.setValue(s.linewidth)

    def remove_series(self):
        r = self.list.currentRow()
        if r >= 0:
            del self.series[r]
            self.refresh_list()
            self.redraw()

    def refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for s in self.series:
            it = QListWidgetItem(s.summary())
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s.visible else Qt.Unchecked)
            pm = QPixmap(14, 14); pm.fill(QColor(s.color))
            it.setIcon(QIcon(pm))
            self.list.addItem(it)
        self.list.blockSignals(False)

    def _item_toggled(self, item):
        r = self.list.row(item)
        if 0 <= r < len(self.series):
            self.series[r].visible = item.checkState() == Qt.Checked
            self.redraw()

    # ------------------------------------------------------------------ drawing
    def flash(self, msg: str, error=False):
        self.status_lbl.setText(f"<span style='color:{'#ff6b6b' if error else '#3fb950'}'>{msg}</span>")

    def redraw(self):
        try:
            msgs = draw_plot(self.fig, self.series, self.st)
            self.canvas.draw()
        except Exception as ex:  # noqa: BLE001 - usually a LaTeX failure
            if self.st.usetex:
                self.st.usetex = False
                self.k_tex.blockSignals(True); self.k_tex.setChecked(False); self.k_tex.blockSignals(False)
                self.flash(f"LaTeX rendering failed, switched to built-in renderer: {str(ex)[:160]}", True)
                return self.redraw()
            self.flash(f"Draw error: {ex}", True)
            return
        self.flash("; ".join(msgs), True) if msgs else self.flash(f"{sum(s.visible for s in self.series)} series drawn")

    # ------------------------------------------------------------------ export
    def export_image(self):
        w, h = self.fig.get_size_inches()
        dlg = ExportDialog(self, w, h)
        if dlg.exec() != QDialog.Accepted:
            return
        path, flt = QFileDialog.getSaveFileName(self, "Export image", "graph.png",
                                                ";;".join(EXPORT_FILTERS))
        if not path:
            return
        ext = EXPORT_FILTERS.get(flt, ".png")
        if not os.path.splitext(path)[1]:
            path += ext
        is_jpg = path.lower().endswith((".jpg", ".jpeg"))
        try:
            fig = Figure(figsize=(dlg.w.value(), dlg.h.value()))
            draw_plot(fig, self.series, self.st)
            kw = dict(dpi=dlg.dpi.value(), transparent=dlg.transparent.isChecked() and not is_jpg)
            if dlg.tight.isChecked():
                kw["bbox_inches"] = "tight"
            fig.savefig(path, **kw)
        except Exception as ex:  # noqa: BLE001
            return QMessageBox.critical(self, "Export failed", str(ex))
        self.flash(f"Saved {path}")

    def copy_image(self):
        buf = io.BytesIO()
        fig = Figure(figsize=self.fig.get_size_inches())
        draw_plot(fig, self.series, self.st)
        fig.savefig(buf, format="png", dpi=200)
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        self.flash("Image copied to clipboard")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export plotted data", "data.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for s in self.series:
                    if not s.visible or s.kind == "implicit":
                        continue
                    for cv in series_curves(s):
                        w.writerow([f"# {s.summary()} [{cv.role}]"])
                        w.writerow(["x", "y"])
                        w.writerows(zip(cv.x.tolist(), cv.y.tolist()))
        except (OSError, MathError) as ex:
            return QMessageBox.critical(self, "Export failed", str(ex))
        self.flash(f"Saved {path} (implicit curves are not exported)")

    # ------------------------------------------------------------------ project
    def new_project(self):
        self.series.clear(); self.st = PlotSettings(style="dark_background" if self.theme_name == "Dark" else "default")
        self.sync_settings_to_ui(); self.refresh_list(); self.redraw()

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "plot.json", "Project (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "settings": asdict(self.st),
                       "series": [asdict(s) for s in self.series]}, f, indent=1)
        self.flash(f"Saved {path}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Project (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self.st = PlotSettings(**d["settings"])
            self.series = [Series(**s) for s in d["series"]]
        except Exception as ex:  # noqa: BLE001
            return QMessageBox.critical(self, "Open failed", str(ex))
        self.sync_settings_to_ui(); self.refresh_list(); self.redraw()

    # ------------------------------------------------------------------ help
    def load_demo(self):
        self.series = [
            Series("function", params=dict(expr="sin(x)/x", xmin="-20", xmax="20", n=2000, deriv=False),
                   color=TAB10[0]),
            Series("function", params=dict(expr="tan(x)", xmin="-10", xmax="10", n=3000, deriv=False),
                   color=TAB10[1]),
            Series("parametric", params=dict(xexpr="cos(t)", yexpr="sin(3t)", tmin="0", tmax="2*pi", n=2000),
                   color=TAB10[2]),
            Series("table", params=dict(xs=[0, 1, 2, 3, 4, 5], ys=[0.1, 1.2, 3.8, 9.1, 15.7, 25.2],
                                        style="Markers", marker="o", fit_degree=2), color=TAB10[3]),
        ]
        self.st.title = "Demo: $\\frac{\\sin x}{x}$, $\\tan x$, a Lissajous curve and a fitted data set"
        self.sync_settings_to_ui(); self.refresh_list(); self.redraw()

    def about(self):
        box = QMessageBox(self)
        box.setWindowTitle("About Curvexa")
        box.setIconPixmap(render_icon(72))
        box.setText("<b style='font-size:16px'>Curvexa</b><br>Graph-plotting studio")
        box.setInformativeText(
            f"PySide6 {__import__('PySide6').__version__} · Matplotlib {mpl.__version__} · "
            f"NumPy {np.__version__} · SymPy {sp.__version__}<br>"
            f"Compiled LaTeX: {'available' if HAVE_REAL_LATEX else 'not found (using built-in renderer)'}")
        box.exec()

    def syntax_help(self):
        QMessageBox.information(self, "Syntax", (
            "Operators:  + - * / ^   (2x, 3sin(x), (x+1)(x-1) work: * is optional)\n"
            "Constants:  pi (π), e\n"
            "Variables:  x (function / implicit), y (implicit), t (parametric), theta or θ (polar)\n"
            "Functions:  sin cos tan sec csc cot asin acos atan atan2 sinh cosh tanh ...\n"
            "            exp ln log(=log10) log2 log(x,base) sqrt cbrt abs sign floor ceil\n"
            "            min max mod Heaviside\n"
            "Ranges accept expressions:  -2*pi , pi/2\n"
            "Implicit:  x^2 + y^2 = 4     (or just  x^2 + y^2 - 4)\n"
            "Table:     paste two columns, import a CSV, or use 'Fill x'.\n"
            "Labels:    anything between $...$ is LaTeX, e.g. $\\omega\\ (rad/s)$"))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Curvexa")
    app.setApplicationDisplayName("Curvexa")
    if sys.platform == "win32":                      # own taskbar identity so the icon shows correctly
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("curvexa.graphstudio.1")
        except Exception:  # noqa: BLE001
            pass
    app.setWindowIcon(make_app_icon())
    apply_theme(app, "Dark")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
