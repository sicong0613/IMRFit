from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer


@dataclass
class PlotHandles:
    exp_line: object | None = None
    sim_line: object | None = None
    fit_best_line: object | None = None
    fit_window_fill: object | None = None
    fit_window_vline_start: object | None = None
    fit_window_vline_end: object | None = None


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(constrained_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.handles = PlotHandles()

        # data bounds for zoom reference (set after each plot)
        self._data_xlim: tuple | None = None
        self._data_ylim: tuple | None = None

        self.ax.set_xlabel("t (s)")
        self.ax.set_ylabel("R (m)")
        self.ax.grid(True, alpha=0.3)

        # drag state for fit-window lines
        self._drag_target: str | None = None  # "start" or "end"
        self._drag_callback: Callable[[str, float], None] | None = None
        self._pick_tolerance = 0.02  # fraction of x-axis range
        self._drag_xlim: tuple[float, float] | None = None  # clamp range

        # pan state
        self._pan_start: tuple | None = None   # (event.x, event.y) in display coords
        self._pan_xlim: list | None = None
        self._pan_ylim: list | None = None

        # MATLAB-like delayed data tips for plotted curves.
        self._data_tip_delay_ms = 500
        self._data_tip_pick_radius_px = 14.0
        self._data_tip_candidate: dict | None = None
        self._data_tip_annotation = None
        self._data_tip_pins: list[dict] = []
        self._data_tip_timer = QTimer(self)
        self._data_tip_timer.setSingleShot(True)
        self._data_tip_timer.timeout.connect(self._show_pending_data_tip)

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("axes_leave_event", self._on_axes_leave)

    # ---- draggable fit-window support ------------------------------------

    def set_drag_callback(self, cb: Callable[[str, float], None] | None):
        """Register *cb(which, x_data)* called when a fit-window line is
        dragged.  *which* is ``"start"`` or ``"end"``."""
        self._drag_callback = cb

    def set_drag_limits(self, x_min: float, x_max: float):
        """Clamp dragged fit-window lines to ``[x_min, x_max]``."""
        self._drag_xlim = (min(x_min, x_max), max(x_min, x_max))

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        h = self.handles

        # Check if near a fit-window line first
        if (
            self._drag_callback is not None
            and (h.fit_window_vline_start is not None or h.fit_window_vline_end is not None)
        ):
            xlim = self.ax.get_xlim()
            tol = (xlim[1] - xlim[0]) * self._pick_tolerance
            x = event.xdata

            dist_start = dist_end = float("inf")
            if h.fit_window_vline_start is not None:
                dist_start = abs(x - h.fit_window_vline_start.get_xdata()[0])
            if h.fit_window_vline_end is not None:
                dist_end = abs(x - h.fit_window_vline_end.get_xdata()[0])

            best = min(dist_start, dist_end)
            if best <= tol:
                self._drag_target = "start" if dist_start <= dist_end else "end"
                return

        # Not near a fit line — start pan
        data_tip = self._nearest_data_tip(event)
        if data_tip is not None:
            pinned = self._matching_pinned_data_tip(data_tip)
            self._data_tip_timer.stop()
            self._hide_data_tip(draw=False)
            self._data_tip_candidate = None
            if pinned is not None:
                self._remove_pinned_data_tip(pinned, draw=False)
            else:
                self._pin_data_tip(data_tip, draw=False)
            self.draw_idle()
            return

        self._hide_data_tip()
        self._pan_start = (event.x, event.y)
        self._pan_xlim = list(self.ax.get_xlim())
        self._pan_ylim = list(self.ax.get_ylim())

    def _on_release(self, event):
        self._drag_target = None
        self._pan_start = None

    def _on_motion(self, event):
        # --- fit-window line drag ---
        if self._drag_target is not None:
            if event.inaxes != self.ax:
                return
            x = event.xdata
            if x is None:
                return
            if self._drag_xlim is not None:
                x = max(self._drag_xlim[0], min(x, self._drag_xlim[1]))

            vline = (
                self.handles.fit_window_vline_start
                if self._drag_target == "start"
                else self.handles.fit_window_vline_end
            )
            if vline is not None:
                vline.set_xdata([x, x])

            self._update_fill_region()
            self.draw_idle()

            if self._drag_callback is not None:
                self._drag_callback(self._drag_target, x)
            return

        # --- pan ---
        if self._pan_start is not None and event.x is not None:
            dx_disp = event.x - self._pan_start[0]
            dy_disp = event.y - self._pan_start[1]
            xlim = self._pan_xlim
            ylim = self._pan_ylim
            bbox = self.ax.get_window_extent()
            if bbox.width > 0 and bbox.height > 0:
                dx_data = -dx_disp * (xlim[1] - xlim[0]) / bbox.width
                dy_data = -dy_disp * (ylim[1] - ylim[0]) / bbox.height
                self.ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
                self.ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
                self.draw_idle()
            return

        self._update_hover_data_tip(event)

    def _on_axes_leave(self, _event):
        self._data_tip_timer.stop()
        self._data_tip_candidate = None
        self._hide_data_tip()

    def clear_data_tip(self):
        """Forget all data tips before plot artists are rebuilt."""
        self._data_tip_timer.stop()
        self._data_tip_candidate = None
        self._data_tip_annotation = None
        self._data_tip_pins.clear()

    def capture_pinned_data_tips(self) -> list[dict]:
        """Describe pinned tips so they can be restored after plot rebuilding."""
        lines = list(self.ax.lines)
        captured: list[dict] = []
        for pinned in self._data_tip_pins:
            line = pinned.get("line")
            try:
                line_index = lines.index(line)
            except ValueError:
                continue
            label = str(line.get_label())
            label_occurrence = sum(
                1 for previous in lines[:line_index] if str(previous.get_label()) == label
            )
            captured.append({
                "line_index": line_index,
                "line_label": label,
                "label_occurrence": label_occurrence,
                "x": float(pinned["x"]),
                "y": float(pinned["y"]),
            })
        return captured

    def restore_pinned_data_tips(self, captured: list[dict] | None):
        """Restore fixed tips onto the corresponding newly created line artists."""
        if not captured:
            return
        lines = list(self.ax.lines)
        restored = False
        for saved in captured:
            label = str(saved.get("line_label", ""))
            line_index = int(saved.get("line_index", -1))
            line = None
            if 0 <= line_index < len(lines) and str(lines[line_index].get_label()) == label:
                line = lines[line_index]
            else:
                matches = [candidate for candidate in lines if str(candidate.get_label()) == label]
                occurrence = int(saved.get("label_occurrence", 0))
                if 0 <= occurrence < len(matches):
                    line = matches[occurrence]
            if line is None or not line.get_visible():
                continue
            self._pin_data_tip(
                {
                    "line": line,
                    "x": float(saved["x"]),
                    "y": float(saved["y"]),
                },
                draw=False,
            )
            restored = True
        if restored:
            self.draw_idle()

    def _update_hover_data_tip(self, event):
        candidate = self._nearest_data_tip(event)
        if candidate is None:
            self._data_tip_timer.stop()
            self._data_tip_candidate = None
            self._hide_data_tip()
            return

        if self._matching_pinned_data_tip(candidate) is not None:
            self._data_tip_timer.stop()
            self._data_tip_candidate = candidate
            self._hide_data_tip()
            return

        previous = self._data_tip_candidate
        same_line = previous is not None and previous.get("line") is candidate.get("line")
        self._data_tip_candidate = candidate
        if self._data_tip_annotation is not None and same_line:
            self._show_data_tip(candidate)
        elif not same_line:
            self._hide_data_tip()
            self._data_tip_timer.start(self._data_tip_delay_ms)
        elif not self._data_tip_timer.isActive():
            self._data_tip_timer.start(self._data_tip_delay_ms)

    def _nearest_data_tip(self, event) -> dict | None:
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return None
        mouse = np.array([float(event.x), float(event.y)], dtype=float)
        best: dict | None = None
        best_distance_sq = self._data_tip_pick_radius_px ** 2

        for line in self.ax.lines:
            if not line.get_visible() or str(line.get_label()).startswith("_"):
                continue
            if str(line.get_linestyle()).lower() in ("none", "", " ") and str(
                line.get_marker()
            ).lower() in ("none", "", " "):
                continue
            try:
                x = np.asarray(line.get_xdata(orig=False), dtype=float).reshape(-1)
                y = np.asarray(line.get_ydata(orig=False), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            n = min(x.size, y.size)
            if n == 0:
                continue
            x = x[:n]
            y = y[:n]
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue
            x = x[finite]
            y = y[finite]
            display = line.get_transform().transform(np.column_stack((x, y)))

            linestyle = str(line.get_linestyle()).lower()
            is_continuous = linestyle not in ("none", "", " ") and x.size >= 2
            if is_continuous:
                starts = display[:-1]
                vectors = display[1:] - starts
                lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
                projection = np.zeros(vectors.shape[0], dtype=float)
                valid_segments = lengths_sq > 0.0
                if np.any(valid_segments):
                    offsets = mouse - starts[valid_segments]
                    projection[valid_segments] = np.clip(
                        np.einsum("ij,ij->i", offsets, vectors[valid_segments])
                        / lengths_sq[valid_segments],
                        0.0,
                        1.0,
                    )
                closest = starts + projection[:, None] * vectors
                distances_sq = np.sum((closest - mouse) ** 2, axis=1)
                index = int(np.argmin(distances_sq))
                distance_sq = float(distances_sq[index])
                fraction = float(projection[index])
                x_value = float(x[index] + fraction * (x[index + 1] - x[index]))
                y_value = float(y[index] + fraction * (y[index + 1] - y[index]))
            else:
                distances_sq = np.sum((display - mouse) ** 2, axis=1)
                index = int(np.argmin(distances_sq))
                distance_sq = float(distances_sq[index])
                x_value = float(x[index])
                y_value = float(y[index])

            if distance_sq <= best_distance_sq:
                best_distance_sq = distance_sq
                best = {"line": line, "x": x_value, "y": y_value}
        return best

    def _show_pending_data_tip(self):
        if (
            self._data_tip_candidate is not None
            and self._matching_pinned_data_tip(self._data_tip_candidate) is None
        ):
            self._show_data_tip(self._data_tip_candidate)

    def _show_data_tip(self, candidate: dict):
        self._hide_data_tip(draw=False)
        self._data_tip_annotation = self._create_data_tip_annotation(candidate)
        self.draw_idle()

    def _matching_pinned_data_tip(self, candidate: dict) -> dict | None:
        line = candidate.get("line")
        try:
            candidate_display = line.get_transform().transform(
                (float(candidate["x"]), float(candidate["y"]))
            )
        except (AttributeError, TypeError, ValueError):
            return None

        tolerance_sq = self._data_tip_pick_radius_px ** 2
        for pinned in self._data_tip_pins:
            if pinned.get("line") is not line:
                continue
            try:
                pinned_display = line.get_transform().transform(
                    (float(pinned["x"]), float(pinned["y"]))
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if float(np.sum((candidate_display - pinned_display) ** 2)) <= tolerance_sq:
                return pinned
        return None

    def _pin_data_tip(self, candidate: dict, *, draw: bool = True):
        pinned = {
            "line": candidate["line"],
            "x": float(candidate["x"]),
            "y": float(candidate["y"]),
        }
        pinned["annotation"] = self._create_data_tip_annotation(pinned)
        self._data_tip_pins.append(pinned)
        if draw:
            self.draw_idle()

    def _remove_pinned_data_tip(self, pinned: dict, *, draw: bool = True):
        annotation = pinned.get("annotation")
        if annotation is not None:
            try:
                annotation.remove()
            except (ValueError, AttributeError):
                pass
        try:
            self._data_tip_pins.remove(pinned)
        except ValueError:
            pass
        if draw:
            self.draw_idle()

    def _create_data_tip_annotation(self, candidate: dict):
        x = float(candidate["x"])
        y = float(candidate["y"])
        x_label = self.ax.get_xlabel().strip() or "x"
        y_label = self.ax.get_ylabel().strip() or "y"
        x_name = x_label.split(" (")[0]
        y_name = y_label.split(" (")[0]
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_fraction = (x - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
        y_fraction = (y - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.5
        x_offset = -10 if x_fraction > 0.72 else 10
        y_offset = -10 if y_fraction > 0.72 else 10
        return self.ax.annotate(
            f"{x_name} = {x:.6g}\n{y_name} = {y:.6g}",
            xy=(x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            va="top" if y_offset < 0 else "bottom",
            fontsize=9,
            color="black",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#555555", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
            zorder=20,
        )

    def _hide_data_tip(self, *, draw: bool = True):
        annotation = self._data_tip_annotation
        self._data_tip_annotation = None
        if annotation is not None:
            try:
                annotation.remove()
            except (ValueError, AttributeError):
                pass
            if draw:
                self.draw_idle()

    def _update_fill_region(self):
        """Redraw the shaded region between the two vlines."""
        if self.handles.fit_window_fill is not None:
            try:
                self.handles.fit_window_fill.remove()
            except Exception:
                pass
            self.handles.fit_window_fill = None

        vs = self.handles.fit_window_vline_start
        ve = self.handles.fit_window_vline_end
        if vs is not None and ve is not None:
            x0 = vs.get_xdata()[0]
            x1 = ve.get_xdata()[0]
            self.handles.fit_window_fill = self.ax.axvspan(
                min(x0, x1), max(x0, x1),
                alpha=0.08, color="gray", zorder=0,
            )

    # ---- simulation curve ------------------------------------------------

    def clear_sim(self):
        if self.handles.sim_line is not None:
            try:
                self.handles.sim_line.remove()
            except Exception:
                pass
            self.handles.sim_line = None

    def plot_experiment(self, t: np.ndarray, R: np.ndarray):
        if self.handles.exp_line is not None:
            try:
                self.handles.exp_line.remove()
            except Exception:
                pass

        (line,) = self.ax.plot(
            t,
            R,
            linestyle="None",
            marker="s",
            markersize=2,
            color="k",
            label="exp data",
        )
        self.handles.exp_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    def plot_simulation(self, t: np.ndarray, R: np.ndarray):
        self.clear_sim()
        (line,) = self.ax.plot(t, R, "-", linewidth=1.5, label="simulation", color="C0")
        self.handles.sim_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    # ---- best-fit curve (live during fitting) ----------------------------

    def plot_fit_best(self, t: np.ndarray, R: np.ndarray):
        self.clear_fit_best()
        (line,) = self.ax.plot(
            t, R, "--", linewidth=1.5, label="best fit", color="#2ca02c",
        )
        self.handles.fit_best_line = line
        self.ax.legend(loc="best")
        self.draw_idle()

    def clear_fit_best(self):
        if self.handles.fit_best_line is not None:
            try:
                self.handles.fit_best_line.remove()
            except Exception:
                pass
            self.handles.fit_best_line = None

    # ---- fit time window -------------------------------------------------

    def draw_fit_window(self, t_start: float, t_end: float):
        self.clear_fit_window()
        self.handles.fit_window_fill = self.ax.axvspan(
            t_start, t_end, alpha=0.08, color="gray", zorder=0,
        )
        self.handles.fit_window_vline_start = self.ax.axvline(
            t_start, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        )
        self.handles.fit_window_vline_end = self.ax.axvline(
            t_end, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        )
        self.draw_idle()

    def clear_fit_window(self):
        for attr in ("fit_window_fill", "fit_window_vline_start", "fit_window_vline_end"):
            obj = getattr(self.handles, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self.handles, attr, None)

    # ---- zoom support -------------------------------------------------------

    def capture_view_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the currently visible x/y limits."""
        return tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim())

    def restore_view_limits(
        self,
        limits: tuple[tuple[float, float], tuple[float, float]] | None,
    ):
        """Restore an exact viewport after a redraw in the same coordinate mode."""
        if limits is None:
            return
        xlim, ylim = limits
        if all(np.isfinite(xlim)) and xlim[0] != xlim[1]:
            self.ax.set_xlim(xlim)
        if all(np.isfinite(ylim)) and ylim[0] != ylim[1]:
            self.ax.set_ylim(ylim)
        self.draw_idle()

    def capture_relative_view(self) -> dict[str, tuple[float, float]] | None:
        """Express the visible viewport relative to the current full data bounds."""
        if self._data_xlim is None or self._data_ylim is None:
            return None

        def relative(view, full):
            span = float(full[1] - full[0])
            if not np.isfinite(span) or span == 0.0:
                return (0.0, 1.0)
            return (
                float((view[0] - full[0]) / span),
                float((view[1] - full[0]) / span),
            )

        return {
            "x": relative(self.ax.get_xlim(), self._data_xlim),
            "y": relative(self.ax.get_ylim(), self._data_ylim),
        }

    def restore_relative_view(self, view: dict[str, tuple[float, float]] | None):
        """Apply a relative viewport to newly calculated full data bounds."""
        if view is None or self._data_xlim is None or self._data_ylim is None:
            return

        def absolute(relative, full):
            span = float(full[1] - full[0])
            return (
                float(full[0] + relative[0] * span),
                float(full[0] + relative[1] * span),
            )

        self.restore_view_limits((
            absolute(view["x"], self._data_xlim),
            absolute(view["y"], self._data_ylim),
        ))

    def set_data_bounds(self, xlim: tuple, ylim: tuple):
        """Store the full data bounds for zoom reference."""
        self._data_xlim = xlim
        self._data_ylim = ylim

    def zoom_x(self, fraction: float):
        """Zoom x-axis anchored at the left edge (x_min stays fixed)."""
        if self._data_xlim is None:
            return
        x_left = self._data_xlim[0]
        x_range = (self._data_xlim[1] - self._data_xlim[0]) * max(0.01, fraction)
        self.ax.set_xlim(x_left, x_left + x_range)
        self.draw_idle()

    def set_x_view(self, left: float, right: float):
        """Set the visible x-axis interval without changing full data bounds."""
        if np.isfinite(left) and np.isfinite(right) and left != right:
            self.ax.set_xlim(min(left, right), max(left, right))
            self.draw_idle()

    def zoom_y(self, fraction: float):
        """Zoom y-axis anchored at the bottom edge (y_min stays fixed)."""
        if self._data_ylim is None:
            return
        y_bottom = self._data_ylim[0]
        y_range = (self._data_ylim[1] - self._data_ylim[0]) * max(0.01, fraction)
        self.ax.set_ylim(y_bottom, y_bottom + y_range)
        self.draw_idle()

    def set_y_view(self, bottom: float, top: float):
        """Set the visible y-axis interval without changing full data bounds."""
        if np.isfinite(bottom) and np.isfinite(top) and bottom != top:
            self.ax.set_ylim(min(bottom, top), max(bottom, top))
            self.draw_idle()

    def reset_zoom(self):
        """Restore full data view."""
        if self._data_xlim is not None:
            self.ax.set_xlim(self._data_xlim)
        if self._data_ylim is not None:
            self.ax.set_ylim(self._data_ylim)
        self.draw_idle()

    def wheelEvent(self, event):  # noqa: N802
        """Zoom in/out centered on mouse cursor (like MATLAB)."""
        if self._data_xlim is None and self._data_ylim is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        factor = 0.85 if delta > 0 else 1.0 / 0.85
        ax = self.ax
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        # Get mouse position in data coordinates
        try:
            pos = event.position()
            x_pix = pos.x()
            y_pix = pos.y()
            # Convert Qt top-left origin to matplotlib bottom-left origin
            y_mpl = self.height() - y_pix
            x_data, y_data = ax.transData.inverted().transform((x_pix, y_mpl))
            # Clamp to current view
            x_data = max(xlim[0], min(x_data, xlim[1]))
            y_data = max(ylim[0], min(y_data, ylim[1]))
        except Exception:
            x_data = (xlim[0] + xlim[1]) / 2
            y_data = (ylim[0] + ylim[1]) / 2

        new_xlim = [x_data + (x - x_data) * factor for x in xlim]
        new_ylim = [y_data + (y - y_data) * factor for y in ylim]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.draw_idle()
        event.accept()
