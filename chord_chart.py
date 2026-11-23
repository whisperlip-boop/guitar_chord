"""
QPainter chord charts for a single guitar voicing.

One widget class draws either panel of the viewer:
  - PANEL_FINGERING: chord box with finger dots (Dots / Fingers / Fingers + Barre).
  - PANEL_NOTES:     same box geometry, each dot labelled with its note name.
Both use the start fret of the ASCII renderer in diagram.py, always MAX_FRET_COUNT
frets wide, and its row order: strings run horizontally (high e on top), frets
increase left to right.

The fingering panel can be made editable (set_editable): clicking a fret cell or the
open/muted marker emits frets_edited with the new shape, and the mouse wheel shifts
the visible fret window.
"""

import logging
import math
import re
from typing import NamedTuple, Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QFont,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QSizePolicy, QWidget

from chord_parser import ParsedChord, semitone_to_name
from diagram import MAX_FRET_COUNT, decide_render_window, render_left, render_right
from voicings import OPEN_STRING_SEMITONES, STRING_NAMES, Voicing

logger = logging.getLogger(__name__)

PANEL_FINGERING = "fingering"
PANEL_NOTES = "notes"
_PANEL_TITLES: dict[str, str] = {PANEL_FINGERING: "Fingering chart", PANEL_NOTES: "Note chart"}

MODE_DOTS = "dots"
MODE_FINGERS = "fingers"
MODE_BARRE = "barre"
DISPLAY_MODES: list[tuple[str, str]] = [
    ("Dots", MODE_DOTS),
    ("Fingers", MODE_FINGERS),
    ("Fingers + Barre", MODE_BARRE),
]

PALETTE: dict[str, str] = {
    "background": "#ffffff",
    "border":     "#d0d7de",
    "fret":       "#8c959f",
    "nut":        "#24292f",
    "string":     "#57606a",
    "dot":        "#24292f",
    "root":       "#2f6fed",
    "dot_text":   "#ffffff",
    "label":      "#24292f",
    "dim_label":  "#8c959f",
    "preview":    "#2f6fed",
}

CUSTOM_SOURCE = "custom"   # Voicing.source of a shape built by clicking the fretboard
MAX_EDIT_FRET = 24         # highest fret reachable by clicking (standard 24-fret neck)
_PREVIEW_ALPHA = 70        # hover preview dot opacity (0-255)
_WHEEL_STEP = 120          # angleDelta of one wheel notch

# Geometry is expressed in units of one string gap (u) so the chart scales with the widget.
_FRET_RATIO = 1.6      # fret column width / string gap
_LEFT_UNITS = 1.8      # string-name column + open/muted marker column, left of the nut
_RIGHT_UNITS = 1.4     # per-string label column right of the box
_TOP_UNITS = 0.6
_BOTTOM_UNITS = 1.0    # "Nfr" label under the first fret column
_LABEL_GAP = 0.3       # gap between the low E string and the "Nfr" label; also ends the low-E click zone
_PADDING_PX = 10
_DOT_RADIUS = {PANEL_FINGERING: 0.36, PANEL_NOTES: 0.42}   # in u; note names need a bit more room


class _Geometry(NamedTuple):
    """Pixel layout of one chart; shared by painting and hit-testing."""

    u: float          # string gap
    start: int        # first fret shown
    n_frets: int
    fret_w: float
    left: float       # x of the nut / first fret line
    top: float        # y of the high e string
    box_w: float
    box_h: float
    marker_x: float   # centre of the open/muted marker column

    def y_of(self, string_idx: int) -> float:
        """Row y of a string (high e on top, as in render_left)."""
        return self.top + string_idx * self.u

    def x_of_fret(self, fret: int) -> float:
        """Cell centre x of a fret (mid-way between two fret lines)."""
        return self.left + (fret - self.start + 0.5) * self.fret_w


def _clamp_start(start: int) -> int:
    """Keep a window start inside 1..MAX_EDIT_FRET - MAX_FRET_COUNT + 1."""
    return min(max(start, 1), MAX_EDIT_FRET - MAX_FRET_COUNT + 1)


def _pitch(string_idx: int, fret: int) -> int:
    """Pitch class sounded by a string at a fret."""
    return (OPEN_STRING_SEMITONES[string_idx] + fret) % 12


class ChordChartView(QWidget):
    """Graphic chord box (fingering or note panel) with copy/save actions.

    Signals:
        status_message(str): Short English feedback after copy/save actions.
        frets_edited(list): Editable panel only; the new six frets (e..E, -1 muted,
            0 open, n fret) after a click.
        view_changed(int): Editable panel only; new window start after a wheel/button shift.
    """

    status_message = Signal(str)
    frets_edited = Signal(list)
    view_changed = Signal(int)

    def __init__(self, panel: str = PANEL_FINGERING, parent: Optional[QWidget] = None) -> None:
        """Create an empty chart.

        Args:
            panel: PANEL_FINGERING or PANEL_NOTES.
            parent: Optional parent widget.

        Raises:
            ValueError: If panel is not a known panel kind.
        """
        super().__init__(parent)
        if panel not in _PANEL_TITLES:
            raise ValueError(f"Unknown panel kind: {panel!r}")
        self._panel = panel
        self._voicing: Optional[Voicing] = None
        self._parsed: Optional[ParsedChord] = None
        self._mode: str = MODE_DOTS
        self._editable = False
        self._view_start: Optional[int] = None    # None = start chosen by decide_render_window
        self._hover: Optional[tuple[int, int]] = None   # (string, fret); fret 0 = marker column
        self._wheel_accum = 0
        self.setMinimumSize(280, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # ------------------------------------------------------------------ state
    def set_voicing(
        self, voicing: Optional[Voicing], parsed: Optional[ParsedChord], keep_view: bool = False
    ) -> None:
        """Show a voicing, or the empty state when voicing is None.

        The fingering panel also draws a voicing without a chord (unidentified custom
        shape: no root colour); the note panel needs both.

        Args:
            voicing: Voicing to draw.
            parsed: Chord it belongs to (root colour and note spelling).
            keep_view: Keep a fret window set by clicking/scrolling instead of
                returning to the automatic window.
        """
        self._voicing = voicing
        self._parsed = parsed
        if not keep_view:
            self._view_start = None
        self.update()

    def set_editable(self, editable: bool) -> None:
        """Enable click editing, hover preview and wheel scrolling (fingering panel only).

        Args:
            editable: True to make the chart an input surface.

        Raises:
            ValueError: If called with True on the note panel.
        """
        if editable and self._panel != PANEL_FINGERING:
            raise ValueError("Only the fingering panel can be edited")
        self._editable = editable
        self._hover = None
        self.setMouseTracking(editable)
        self.update()

    def shift_view(self, delta: int) -> None:
        """Move the visible fret window by delta frets (editable panel only).

        The start is clamped to 1..MAX_EDIT_FRET - MAX_FRET_COUNT + 1.

        Args:
            delta: Frets to move; negative = toward the nut.
        """
        if not self._editable or delta == 0:
            return
        start = _clamp_start(self._window()[0] + delta)
        if start != self._window()[0]:
            self.set_view_start(start)
            self.view_changed.emit(start)

    def view_start(self) -> Optional[int]:
        """Return the fixed window start, or None while the automatic start is used."""
        return self._view_start

    def set_view_start(self, start: Optional[int]) -> None:
        """Fix the window start (clamped like shift_view), or None for the automatic start.

        Args:
            start: First fret to show, or None.
        """
        self._view_start = None if start is None else _clamp_start(start)
        self.update()

    def set_mode(self, mode: str) -> None:
        """Switch the fingering panel between MODE_DOTS, MODE_FINGERS and MODE_BARRE.

        The note panel stores the mode but does not use it.

        Args:
            mode: One of the values in DISPLAY_MODES.

        Raises:
            ValueError: If mode is not a known display mode.
        """
        if mode not in {value for _, value in DISPLAY_MODES}:
            raise ValueError(f"Unknown display mode: {mode!r}")
        self._mode = mode
        self.update()

    def has_chord(self) -> bool:
        """Return True when a voicing is shown (the note panel also needs its chord)."""
        return self._voicing is not None and (self._parsed is not None or self._panel == PANEL_FINGERING)

    def missing_data_hint(self) -> str:
        """Return an English hint when the fingering mode needs data the voicing lacks."""
        v = self._voicing
        # A clicked shape has no fingering by design and is simply drawn as dots.
        if v is None or v.source == CUSTOM_SOURCE or self._panel != PANEL_FINGERING or self._mode == MODE_DOTS:
            return ""
        # Fingers present but barres None (e.g. a malformed chords-db "barres" field) is drawn as
        # fingers without a bar (see _paint_chart); only missing fingers get a hint.
        return "No fingering data for this voicing" if v.fingers is None else ""

    # ------------------------------------------------------------------ actions
    def copy_image(self) -> None:
        """Copy the rendered chart to the clipboard as an image."""
        if not self.has_chord():
            return
        QApplication.clipboard().setPixmap(self._grab_clean())
        self.status_message.emit(f"{_PANEL_TITLES[self._panel]} image copied to clipboard.")

    def ascii_text(self) -> str:
        """Return the matching ASCII diagram (render_left / render_right), or "" when empty."""
        if not self.has_chord():
            return ""
        if self._panel == PANEL_FINGERING:
            return render_left(self._voicing)
        return render_right(self._voicing, self._parsed)

    def copy_text(self) -> None:
        """Copy the matching ASCII diagram to the clipboard."""
        if not self.has_chord():
            return
        QApplication.clipboard().setText(self.ascii_text())
        self.status_message.emit(f"{_PANEL_TITLES[self._panel]} text copied to clipboard.")

    def save_png(self) -> None:
        """Ask for a path and save the rendered chart as PNG; report failures."""
        if not self.has_chord():
            return
        chord = re.sub(r'[\\/:*?"<>|\s]+', "_", self._parsed.raw) if self._parsed else CUSTOM_SOURCE
        path, _ = QFileDialog.getSaveFileName(
            self, "Save chord diagram", f"{chord}_{self._panel}.png", "PNG image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if self._grab_clean().save(path, "PNG"):
            logger.info("Saved %s to %s", _PANEL_TITLES[self._panel].lower(), path)
            self.status_message.emit(f"{_PANEL_TITLES[self._panel]} image saved.")
        else:
            logger.error("Failed to save %s to %s", _PANEL_TITLES[self._panel].lower(), path)
            QMessageBox.warning(
                self, "Save failed",
                f"Could not save the image to:\n{path}\n\nCheck the folder exists and is writable.",
            )

    def _grab_clean(self) -> QPixmap:
        """Render the chart without the hover preview (exports must not show the cursor cell)."""
        hover, self._hover = self._hover, None
        try:
            return self.grab()
        finally:
            self._hover = hover

    # ------------------------------------------------------------------ events
    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 (Qt override)
        """Offer copy image / copy text / save actions."""
        menu = QMenu(self)
        enabled = self.has_chord()
        actions = [
            (menu.addAction("Copy image"), self.copy_image),
            (menu.addAction("Copy text"), self.copy_text),
            (menu.addAction("Save as PNG…"), self.save_png),
        ]
        for action, _ in actions:
            action.setEnabled(enabled)
        chosen = menu.exec(event.globalPos())
        menu.deleteLater()
        for action, handler in actions:
            if chosen is action:
                handler()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        """Ctrl+C copies the chart image."""
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Double-click copies the chart image; on the editable panel it is a second click."""
        if event.button() == Qt.MouseButton.LeftButton and not self._editable:
            self.copy_image()
        super().mouseDoubleClickEvent(event)   # QWidget forwards to mousePressEvent

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Editable panel: place, move or clear a dot, or toggle open/muted."""
        if not self._editable or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        event.accept()
        target = self._hit_test(event.position())
        if target is None:
            return
        string_idx, fret = target
        frets = self._display_frets()
        if fret == 0:
            frets[string_idx] = 0 if frets[string_idx] == -1 else -1
        else:
            frets[string_idx] = 0 if frets[string_idx] == fret else fret
        # Freeze the window the user clicked in (already legal) so later lookups keep it.
        self._view_start = self._window()[0]
        logger.debug("Fretboard click string=%d fret=%d -> %s", string_idx, fret, frets)
        self.frets_edited.emit(frets)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Editable panel: track the hover target for the preview dot."""
        if self._editable:
            self._set_hover(self._hit_test(event.position()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt override)
        """Drop the hover preview when the mouse leaves the chart."""
        self._set_hover(None)
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt override)
        """Editable panel: wheel down / right shows higher frets, up / left lower frets."""
        if not self._editable:
            super().wheelEvent(event)
            return
        event.accept()
        delta = event.angleDelta().y() or -event.angleDelta().x()
        if delta * self._wheel_accum < 0:   # direction reversed: drop the old partial notch
            self._wheel_accum = 0
        self._wheel_accum += delta
        steps = int(self._wheel_accum / _WHEEL_STEP)   # touchpads send fractions of a notch
        if steps:
            self._wheel_accum -= steps * _WHEEL_STEP
            self.shift_view(-steps)
            self._set_hover(self._hit_test(event.position()))

    def _set_hover(self, target: Optional[tuple[int, int]]) -> None:
        """Store the hover target; repaint only when it changes."""
        if target != self._hover:
            self._hover = target
            self.update()

    # ------------------------------------------------------------------ geometry
    def _display_frets(self) -> list[int]:
        """Frets currently drawn: the voicing, or all open on an empty editable panel."""
        return list(self._voicing.frets) if self._voicing is not None else [0] * 6

    def _window(self) -> tuple[int, int]:
        """(start fret, fret count) of the visible window; always MAX_FRET_COUNT wide.

        The constant width keeps cells in place when a click fixes the start.
        """
        if self._view_start is not None:
            return self._view_start, MAX_FRET_COUNT
        if self._voicing is None:   # empty editable board
            return 1, MAX_FRET_COUNT
        return _clamp_start(decide_render_window(self._voicing)[0]), MAX_FRET_COUNT

    def _geometry(self) -> Optional[_Geometry]:
        """Layout for the current widget size and window; None when there is no room."""
        u = min(
            (self.width() - 2 * _PADDING_PX)
            / (_LEFT_UNITS + MAX_FRET_COUNT * _FRET_RATIO + _RIGHT_UNITS),  # fixed scale across voicings
            (self.height() - 2 * _PADDING_PX) / (_TOP_UNITS + 5 + _BOTTOM_UNITS),
        )
        if u <= 0:
            return None
        start, n_frets = self._window()
        fret_w = u * _FRET_RATIO
        total_w = (_LEFT_UNITS + n_frets * _FRET_RATIO + _RIGHT_UNITS) * u
        left = (self.width() - total_w) / 2 + _LEFT_UNITS * u
        top = (self.height() - (_TOP_UNITS + 5 + _BOTTOM_UNITS) * u) / 2 + _TOP_UNITS * u
        return _Geometry(u, start, n_frets, fret_w, left, top, n_frets * fret_w, 5 * u, left - 0.55 * u)

    def _hit_test(self, pos: QPointF) -> Optional[tuple[int, int]]:
        """Inverse of the paint layout: (string, fret) under pos, fret 0 = marker column.

        Returns None outside every target (string-name column, labels, margins).
        """
        g = self._geometry()
        if g is None:
            return None
        string_idx = math.floor((pos.y() - g.top) / g.u + 0.5)   # uniform half-row boundaries
        if not 0 <= string_idx <= 5 or pos.y() >= g.top + g.box_h + _LABEL_GAP * g.u:
            return None
        x = pos.x()
        if g.left <= x < g.left + g.box_w:
            return string_idx, g.start + min(int((x - g.left) / g.fret_w), g.n_frets - 1)
        if g.marker_x - 0.45 * g.u <= x < g.left:
            return string_idx, 0
        return None

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        """Draw the card background, then the chart or the empty state."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.setPen(QPen(QColor(PALETTE["border"]), 1))
            painter.setBrush(QColor(PALETTE["background"]))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
            if self.has_chord() or self._editable:
                self._paint_chart(painter)
            else:
                painter.setPen(QColor(PALETTE["dim_label"]))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No chord to display")
        finally:
            painter.end()

    def _font(self, unit: float, scale: float, bold: bool = False) -> QFont:
        """Widget font sized relative to the string gap."""
        font = QFont(self.font())
        font.setPixelSize(max(7, round(unit * scale)))
        font.setBold(bold)
        return font

    def _paint_chart(self, p: QPainter) -> None:
        """Draw the box, markers, dots (and barres) and labels for the current voicing."""
        v, parsed = self._voicing, self._parsed
        notes_panel = self._panel == PANEL_NOTES
        g = self._geometry()
        if g is None:
            return
        u, start, n_frets, fret_w, left, top, box_w, box_h, marker_x = g
        y_of, x_of_fret = g.y_of, g.x_of_fret
        frets = self._display_frets()
        radius = _DOT_RADIUS[self._panel] * u
        # No chord (empty editable board or unidentified custom shape): no root colour, sharps.
        root_pc = parsed.root_semitone % 12 if parsed else None
        use_flat = parsed.display_use_flat if parsed else False
        has_fingers = not notes_panel and v is not None and v.fingers is not None
        show_fingers = has_fingers and self._mode in (MODE_FINGERS, MODE_BARRE)
        # barres None (unknown) draws like () (none): fingers only, never a guessed bar.
        barres = (v.barres or ()) if has_fingers and self._mode == MODE_BARRE else ()

        def color_for(pc: int) -> QColor:
            return QColor(PALETTE["root"] if pc == root_pc else PALETTE["dot"])

        def under_barre(string_idx: int, fret: int) -> bool:
            return any(b_fret == fret and lo <= string_idx <= hi for b_fret, lo, hi in barres)

        # Frets (vertical), then nut on the left or the starting-fret label under the first column.
        p.setPen(QPen(QColor(PALETTE["fret"]), max(1.0, u / 30)))
        for i in range(n_frets + 1):
            x = left + i * fret_w
            p.drawLine(QPointF(x, top), QPointF(x, top + box_h))
        if start == 1:
            nut_w = max(3.0, u * 0.16)
            p.fillRect(QRectF(left - nut_w, top - 0.5, nut_w, box_h + 1), QColor(PALETTE["nut"]))
        else:
            p.setFont(self._font(u, 0.42))
            p.setPen(QColor(PALETTE["label"]))
            p.drawText(QRectF(left, top + box_h + _LABEL_GAP * u, fret_w, (_BOTTOM_UNITS - _LABEL_GAP) * u),
                       Qt.AlignmentFlag.AlignCenter, f"{start}fr")

        # Strings (horizontal): thicker toward low E.
        for s_idx in range(6):
            p.setPen(QPen(QColor(PALETTE["string"]), max(1.0, u / 40) * (1 + 0.35 * s_idx)))
            p.drawLine(QPointF(left, y_of(s_idx)), QPointF(left + box_w, y_of(s_idx)))

        # String names, then open / muted markers left of the nut (note panel: open-string note name).
        marker_r = 0.2 * u
        p.setBrush(Qt.BrushStyle.NoBrush)
        for s_idx, fret in enumerate(frets):
            cy = y_of(s_idx)
            p.setFont(self._font(u, 0.38))
            p.setPen(QColor(PALETTE["dim_label"]))
            p.drawText(QRectF(left - _LEFT_UNITS * u, cy - u / 2, 0.8 * u, u),
                       Qt.AlignmentFlag.AlignCenter, STRING_NAMES[s_idx])
            if fret == 0:
                pc = _pitch(s_idx, 0)
                color = QColor(PALETTE["root"] if pc == root_pc else PALETTE["label"])
                if notes_panel:
                    p.setFont(self._font(u, 0.4, bold=pc == root_pc))
                    p.setPen(color)
                    p.drawText(QRectF(marker_x - u / 2, cy - u / 2, u, u), Qt.AlignmentFlag.AlignCenter,
                               semitone_to_name(pc, use_flat=use_flat))
                else:
                    p.setPen(QPen(color, max(1.2, u / 22)))
                    p.drawEllipse(QPointF(marker_x, cy), marker_r, marker_r)
            elif fret < 0:
                p.setPen(QPen(QColor(PALETTE["dim_label"]), max(1.2, u / 22)))
                p.drawLine(QPointF(marker_x - marker_r, cy - marker_r), QPointF(marker_x + marker_r, cy + marker_r))
                p.drawLine(QPointF(marker_x - marker_r, cy + marker_r), QPointF(marker_x + marker_r, cy - marker_r))

        # Barres (from data only): vertical bar across the barred string rows, under the dots.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PALETTE["dot"]))
        for b_fret, lo, hi in barres:
            if not start <= b_fret < start + n_frets:
                continue
            cx = x_of_fret(b_fret)
            p.drawRoundedRect(
                QRectF(cx - radius * 0.85, y_of(lo) - radius, radius * 1.7, y_of(hi) - y_of(lo) + 2 * radius),
                radius * 0.85, radius * 0.85,
            )

        # Dots; frets outside the window are skipped exactly as in the text renderer.
        for s_idx, fret in enumerate(frets):
            if not start <= fret < start + n_frets:
                continue
            center = QPointF(x_of_fret(fret), y_of(s_idx))
            dot_rect = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
            pc = _pitch(s_idx, fret)
            if pc == root_pc or not under_barre(s_idx, fret):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color_for(pc))
                p.drawEllipse(center, radius, radius)
            label = ""
            if notes_panel:
                label = semitone_to_name(pc, use_flat=use_flat)
            elif show_fingers and v.fingers[s_idx] > 0:
                label = str(v.fingers[s_idx])
            if label:
                p.setFont(self._font(u, 0.42 if len(label) == 1 else 0.34, bold=True))
                p.setPen(QColor(PALETTE["dot_text"]))
                p.drawText(dot_rect, Qt.AlignmentFlag.AlignCenter, label)

        # Right-hand column per string: fingering panel = sounded note, note panel = fret number.
        col_x = left + box_w + 0.25 * u
        for s_idx, fret in enumerate(frets):
            if fret < 0 or (notes_panel and fret == 0):
                continue
            pc = _pitch(s_idx, fret)
            if notes_panel:
                text, bold, color = str(fret), False, PALETTE["label"]
            else:
                text, bold = semitone_to_name(pc, use_flat=use_flat), pc == root_pc
                color = PALETTE["root"] if bold else PALETTE["label"]
            p.setFont(self._font(u, 0.38, bold=bold))
            p.setPen(QColor(color))
            p.drawText(QRectF(col_x, y_of(s_idx) - u / 2, (_RIGHT_UNITS - 0.25) * u, u),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

        # A fretted string whose dot is off-window gets an edge tag with its fret in the
        # first / last cell (that row has no other dot, so nothing is covered).
        if any(f > 0 and not start <= f < start + n_frets for f in frets):
            p.setFont(self._font(u, 0.32, bold=True))
            for s_idx, fret in enumerate(frets):
                if fret <= 0 or start <= fret < start + n_frets:
                    continue
                below = fret < start
                cx = left + (0.5 * fret_w if below else box_w - 0.5 * fret_w)
                tag = QRectF(cx - 0.5 * u, y_of(s_idx) - 0.3 * u, u, 0.6 * u)
                p.setPen(QPen(QColor(PALETTE["dim_label"]), 1))
                p.setBrush(QColor(PALETTE["background"]))
                p.drawRoundedRect(tag, 0.2 * u, 0.2 * u)
                p.setPen(QColor(PALETTE["label"]))
                p.drawText(tag, Qt.AlignmentFlag.AlignCenter, f"‹{fret}" if below else f"{fret}›")
        if self._editable and self._hover is not None:
            h_string, h_fret = self._hover
            color = QColor(PALETTE["preview"])
            color.setAlpha(_PREVIEW_ALPHA)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            if h_fret == 0:
                p.drawEllipse(QPointF(marker_x, y_of(h_string)), marker_r * 1.6, marker_r * 1.6)
            elif start <= h_fret < start + n_frets:
                p.drawEllipse(QPointF(x_of_fret(h_fret), y_of(h_string)), radius, radius)
