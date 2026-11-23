"""
Guitar Chord Viewer — PySide6 GUI.

Features:
  - User types a chord (e.g. C, CM7, Bbm7, F#m7/B) and presses Enter or "Show".
  - Radio buttons select voicing source: DB / Auto / Both.
  - Two QPainter panels: fingering chart (Dots / Fingers / Fingers + Barre)
    and note chart (note name per string).
  - Left/Right arrow buttons navigate between voicings of the same chord.
  - "<current>/<total>" indicator below the diagrams.
  - At 1/N the left arrow is disabled; at N/N the right arrow is disabled.
  - Clicking the fingering chart builds a custom shape and identifies it live;
    the wheel or the fret buttons scroll the window, "Clear" resets it.
  - Ctrl+C copies a panel's image (double-click too on the note chart);
    right-click offers Copy image / Copy text (ASCII) / Save as PNG.
  - Log records go to %LOCALAPPDATA%/GuitarChordViewer/guitar_chord.log.
"""

import os
import sys
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chord_parser import parse_chord, ChordParseError, ParsedChord, chord_tones
from voicings import (
    OPEN_STRING_SEMITONES,
    Voicing,
    get_voicings,
    identify_chord_from_frets,
    identify_exact_chords,
)
from chord_chart import (
    CUSTOM_SOURCE,
    DISPLAY_MODES,
    MODE_DOTS,
    PANEL_FINGERING,
    PANEL_NOTES,
    ChordChartView,
)

logger = logging.getLogger(__name__)


MONO_FAMILY = "Consolas"
MONO_FONT = QFont(MONO_FAMILY, 12)

_STATUS_TIMEOUT_MS = 2500
_BOARD_PROMPT = "Click the fretboard to find a chord."
_EMPTY_NAME = "Please enter a chord name or click the fretboard."
_TOO_FEW_NOTES = "Click at least 2 strings with different notes."
_NO_MATCH = "No matching chord for these notes."
_MAX_CHIPS = 4

_LOG_DIR_NAME = "GuitarChordViewer"
_LOG_FILE_NAME = "guitar_chord.log"
_LOG_MAX_BYTES = 512 * 1024
_LOG_BACKUP_COUNT = 1

APP_STYLESHEET = """
QWidget {
    background-color: #f7f8fa;
    color: #24292f;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QLineEdit {
    background: white;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 4px 8px;
}
QLineEdit:focus {
    border: 1px solid #2f6fed;
}
QPushButton {
    background-color: #2f6fed;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 18px;
    font-weight: 600;
}
QPushButton:hover { background-color: #255ed1; }
QPushButton:pressed { background-color: #1d4bab; }
QPushButton:disabled { background-color: #c7d2e8; color: #eef1f8; }
QPushButton#navButton {
    background-color: white;
    color: #2f6fed;
    border: 1px solid #d0d7de;
    border-radius: 18px;
    font-size: 14pt;
    font-weight: bold;
    padding: 0px;
}
QPushButton#navButton:hover { background-color: #eef2ff; }
QPushButton#navButton:disabled {
    color: #c7ccd1;
    border-color: #e6e8eb;
    background: #f7f8fa;
}
QRadioButton { spacing: 6px; }
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 8px;
}
QRadioButton::indicator:unchecked {
    border: 1.5px solid #8b949e;
    background: white;
}
QRadioButton::indicator:checked {
    border: 1.5px solid #2f6fed;
    background: #2f6fed;
}
QPushButton#boardButton {
    background-color: white;
    color: #2f6fed;
    border: 1px solid #d0d7de;
    padding: 2px 10px;
}
QPushButton#boardButton:hover { background-color: #eef2ff; }
QPushButton#chordChip {
    background-color: white;
    color: #2f6fed;
    border: 1px solid #2f6fed;
    border-radius: 11px;
    padding: 2px 10px;
}
QPushButton#chordChip:hover { background-color: #eef2ff; }
QPushButton#chordChip:focus { background-color: #dbe6ff; }
QPushButton#chordChip:pressed { background-color: #c9d9ff; }
QLabel#titleLabel { font-weight: 600; }
QLabel#srcTag { color: #57606a; }
QLabel#tonesLabel { color: #32383f; }
QLabel#hintLabel { color: #9a6700; }
QLabel#statusLabel { color: #1a7f37; }
"""


def get_resource_path(relative_path: str) -> str:
    """Resolve a bundled-resource path for both dev and PyInstaller builds.

    Args:
        relative_path: Path relative to the project / bundle root.

    Returns:
        Absolute filesystem path to the resource.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


def configure_logging() -> None:
    """Attach a small rotating log file under %LOCALAPPDATA%/GuitarChordViewer.

    If the folder or file cannot be created, the app runs without a log file.
    """
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return
    log_dir = os.path.join(base, _LOG_DIR_NAME)
    try:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, _LOG_FILE_NAME),
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _log_unhandled_exception(exc_type: type, exc: BaseException, tb: object) -> None:
    """sys.excepthook: record exceptions escaping Qt slots (the exe has no console)."""
    logging.getLogger(__name__).critical("Unhandled exception", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)


class ChordApp(QWidget):
    """Main application window for Guitar Chord Viewer."""

    def __init__(self) -> None:
        """Initialise the application state and build the UI."""
        super().__init__()
        self.setWindowTitle("Guitar Chord Viewer")

        self.voicings: List[Voicing] = []
        self.parsed: Optional[ParsedChord] = None
        self.index: int = 0

        self._set_window_icon()
        self._build_ui()
        self.resize(920, 420)
        self.setMinimumSize(780, 350)

    # ------------------------------------------------------------------ icon
    def _set_window_icon(self) -> None:
        """Apply guitar.ico to the window and the Windows taskbar."""
        ico_path = get_resource_path("guitar.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))
        # Windows-specific: make the taskbar show the app icon instead of the
        # generic Python icon by setting a unique AppUserModelID.
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "GuitarChordViewer.1.0"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Construct all UI widgets and layouts."""
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        # ---- Top bar: chord input + source radio buttons ------------------
        top = QHBoxLayout()
        top.setSpacing(8)

        title_input = QLabel("Input Chord:")
        title_input.setObjectName("titleLabel")
        top.addWidget(title_input)

        self.entry = QLineEdit()
        self.entry.setFont(MONO_FONT)
        self.entry.setFixedWidth(140)
        self.entry.returnPressed.connect(self._show_chord)
        top.addWidget(self.entry)

        self.show_btn = QPushButton("Show")
        self.show_btn.clicked.connect(self._show_chord)
        top.addWidget(self.show_btn)

        top.addSpacing(14)
        title_source = QLabel("Fingering Source:")
        title_source.setObjectName("titleLabel")
        top.addWidget(title_source)

        self.source_group = QButtonGroup(self)
        for label, value in [("DB", "db"), ("Auto", "auto"), ("Both", "both")]:
            radio = QRadioButton(label)
            radio.setProperty("source_value", value)
            if value == "both":
                radio.setChecked(True)
            radio.toggled.connect(self._on_source_change)
            self.source_group.addButton(radio)
            top.addWidget(radio)

        top.addStretch(1)
        root.addLayout(top)

        self.fingering_chart = ChordChartView(PANEL_FINGERING)
        self.fingering_chart.set_editable(True)
        self.fingering_chart.frets_edited.connect(self._on_frets_edited)
        self.fingering_chart.view_changed.connect(self._on_view_changed)
        self.note_chart = ChordChartView(PANEL_NOTES)

        # ---- Display mode row ---------------------------------------------
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        title_mode = QLabel("Display:")
        title_mode.setObjectName("titleLabel")
        mode_row.addWidget(title_mode)

        self.mode_group = QButtonGroup(self)
        for label, value in DISPLAY_MODES:
            radio = QRadioButton(label)
            radio.setProperty("mode_value", value)
            if value == MODE_DOTS:
                radio.setChecked(True)
            radio.toggled.connect(self._on_mode_change)
            self.mode_group.addButton(radio)
            mode_row.addWidget(radio)

        mode_row.addSpacing(14)
        self.hint_label = QLabel("")
        self.hint_label.setObjectName("hintLabel")
        mode_row.addWidget(self.hint_label)
        mode_row.addStretch(1)

        # Fretboard input controls: scroll the clickable window, reset the clicked shape.
        for text, tip, delta in (("◀ Fret", "Show lower frets (or scroll the wheel)", -1),
                                 ("Fret ▶", "Show higher frets (or scroll the wheel)", 1)):
            btn = QPushButton(text)
            btn.setObjectName("boardButton")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, d=delta: self.fingering_chart.shift_view(d))
            mode_row.addWidget(btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("boardButton")
        self.clear_btn.setToolTip("Reset the fretboard to all open strings")
        self.clear_btn.clicked.connect(self._clear_board)
        mode_row.addWidget(self.clear_btn)
        root.addLayout(mode_row)

        # ---- Banner: chord name / status message --------------------------
        self.banner = QLabel(
            "Input Chord.  e.g.: C, CM7, Bbm7, F#m7/B, C△7, C°, Cø"
        )
        self.banner.setObjectName("banner")
        root.addWidget(self.banner)

        # ---- Alternative exact names of a clicked shape -------------------
        # Fixed-height row: chips appear/disappear without moving the charts.
        chip_row = QWidget()
        chip_layout = QHBoxLayout(chip_row)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.setSpacing(6)
        self.chips: List[QPushButton] = []
        for _ in range(_MAX_CHIPS):
            chip = QPushButton()
            chip.setObjectName("chordChip")
            chip.clicked.connect(lambda _checked=False, c=chip: self._on_chip_clicked(c))
            chip.hide()
            chip_layout.addWidget(chip)
            self.chips.append(chip)
        chip_layout.addStretch(1)
        chip_row.setFixedHeight(self.chips[0].sizeHint().height())
        root.addWidget(chip_row)

        # ---- Center: ◀  [fingering chart]  [note chart]  ▶ ------------------
        center = QHBoxLayout()
        center.setSpacing(10)

        self.left_btn = QPushButton("◀")
        self.left_btn.setObjectName("navButton")
        self.left_btn.setFixedSize(36, 36)
        self.left_btn.setEnabled(False)
        self.left_btn.clicked.connect(self._prev)
        center.addWidget(self.left_btn)

        for chart in (self.fingering_chart, self.note_chart):
            chart.status_message.connect(self._show_status)
            center.addWidget(chart, 1)

        self.right_btn = QPushButton("▶")
        self.right_btn.setObjectName("navButton")
        self.right_btn.setFixedSize(36, 36)
        self.right_btn.setEnabled(False)
        self.right_btn.clicked.connect(self._next)
        center.addWidget(self.right_btn)

        root.addLayout(center, 1)

        # ---- Bottom: counter + source tag + chord tones -------------------
        bottom = QHBoxLayout()

        self.counter = QLabel("0/0")
        self.counter.setObjectName("titleLabel")
        bottom.addWidget(self.counter)

        self.src_tag = QLabel("")
        self.src_tag.setObjectName("srcTag")
        bottom.addWidget(self.src_tag)
        bottom.addSpacing(12)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        bottom.addWidget(self.status_label)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self.status_label.clear)

        bottom.addStretch(1)

        self.tones_label = QLabel("")
        self.tones_label.setObjectName("tonesLabel")
        bottom.addWidget(self.tones_label)

        root.addLayout(bottom)

    # --------------------------------------------------------- input handling
    def _current_source(self) -> str:
        """Return the currently selected fingering source value."""
        checked = self.source_group.checkedButton()
        return checked.property("source_value") if checked else "both"

    # ------------------------------------------------------------------ events
    def _show_chord(self) -> None:
        """Show the voicings of the chord typed in the name field."""
        text = self.entry.text().strip()
        if not text:
            if self.parsed is None and self.fingering_chart.has_chord():
                # Custom-shape state: a stray Enter must not wipe the clicked shape or its banner.
                self._show_status(_EMPTY_NAME)
                return
            self._show_error(_EMPTY_NAME)
            return
        try:
            parsed = parse_chord(text)
        except ChordParseError as e:
            self._show_error(f"Input error: {e}")
            return

        voicings = get_voicings(parsed, source=self._current_source(), max_results=60)
        if not voicings:
            self.parsed = parsed
            self.voicings = []
            self.index = 0
            self._set_banner(
                f"Chord: {self._chord_title(parsed)}"
                "   — No voicings found in the selected source.",
                error=True,
            )
            self.tones_label.setText(f"Chord tones: {', '.join(chord_tones(parsed))}")
            self._render_empty()
            return

        self.parsed = parsed
        self.voicings = voicings
        self.index = 0
        self._set_banner(f"Chord: {self._chord_title(parsed)}", error=False)
        self.tones_label.setText(f"Chord tones: {', '.join(chord_tones(parsed))}")
        self._render_current()

    def _on_frets_edited(self, frets: List[int]) -> None:
        """Identify a shape clicked on the fingering chart and show it as a custom shape.

        Clears the chord-name field (the two inputs are mutually exclusive) and
        leaves lookup navigation until the next Show.

        Args:
            frets: 6 elements (e, B, G, D, A, E); -1 muted, 0 open, n fret.
        """
        self.entry.clear()
        self.voicings = []
        self.parsed = None   # custom state: source changes do not re-query
        self.index = 0
        shape = Voicing(frets=list(frets), source=CUSTOM_SOURCE)
        played: List[Optional[int]] = [f if f >= 0 else None for f in frets]
        try:
            exact = identify_exact_chords(played)
            # exact[0] is what identify_chord_from_frets returns; it only adds the fallback.
            result = (exact[0], True) if exact else identify_chord_from_frets(played)
            alternatives = exact[1:]
        except Exception:   # keep the board usable even if the lookup fails on an odd shape
            logger.exception("Reverse lookup failed for %s", frets)
            result, alternatives = None, []
        self._set_chips([p.raw for p in alternatives])
        if result is None:
            notes = {(OPEN_STRING_SEMITONES[i] + f) % 12 for i, f in enumerate(frets) if f >= 0}
            self._set_banner(_TOO_FEW_NOTES if len(notes) < 2 else _NO_MATCH, error=True)
            self.tones_label.setText("")
            self.fingering_chart.set_voicing(shape, None, keep_view=True)
            self.note_chart.set_voicing(None, None)
        else:
            parsed, is_exact = result
            suffix = "" if is_exact else "   — Exact match not found."
            self._set_banner(f"Chord: {self._chord_title(parsed)}{suffix}", error=not is_exact)
            self.tones_label.setText(f"Chord tones: {', '.join(chord_tones(parsed))}")
            self.fingering_chart.set_voicing(shape, parsed, keep_view=True)
            self.note_chart.set_voicing(shape, parsed)
            self.note_chart.set_view_start(self.fingering_chart.view_start())
        self.hint_label.setText("")
        self.counter.setText("-/-")
        self.src_tag.setText("[Custom shape]")
        self.left_btn.setEnabled(False)
        self.right_btn.setEnabled(False)

    def _on_view_changed(self, start: int) -> None:
        """Custom-shape state: keep the note chart on the fingering chart's fret window.

        Args:
            start: New first fret of the fingering chart.
        """
        if self.parsed is None and self.note_chart.has_chord():
            self.note_chart.set_view_start(start)

    def _set_chips(self, names: List[str]) -> None:
        """Show up to _MAX_CHIPS alternative chord names as "= name" chips; hide the rest.

        Args:
            names: Chord names that parse_chord accepts, best first.
        """
        for chip, name in zip(self.chips, names + [""] * _MAX_CHIPS):
            chip.setProperty("chord_name", name)
            chip.setText(f"= {name}")
            chip.setToolTip(f"Show voicings of {name}")
            chip.setVisible(bool(name))

    def _on_chip_clicked(self, chip: QPushButton) -> None:
        """Look up the chip's chord exactly as if it had been typed and Show pressed."""
        name = chip.property("chord_name")
        if name:
            self.entry.setText(name)
            self._show_chord()

    def _clear_board(self) -> None:
        """Reset to the empty state: all strings open, no chord identified."""
        self.entry.clear()
        self.voicings = []
        self.parsed = None
        self.index = 0
        self.tones_label.setText("")
        self._set_banner(_BOARD_PROMPT, error=False)
        self._render_empty()

    def _on_source_change(self) -> None:
        """Re-query voicings when the source radio button changes."""
        if self.parsed is None:
            return
        voicings = get_voicings(self.parsed, source=self._current_source(), max_results=60)
        self.voicings = voicings
        self.index = 0
        if voicings:
            self._render_current()
        else:
            self._set_banner(
                f"Chord: {self._chord_title(self.parsed)}"
                "   — No voicings found in the selected source.",
                error=True,
            )
            self._render_empty()

    def _on_mode_change(self) -> None:
        """Apply the selected display mode to the fingering chart."""
        checked = self.mode_group.checkedButton()
        self.fingering_chart.set_mode(checked.property("mode_value") if checked else MODE_DOTS)
        self.hint_label.setText(self.fingering_chart.missing_data_hint())

    def _show_status(self, message: str) -> None:
        """Show a short-lived status message (e.g. after copying).

        Args:
            message: Text to show; cleared after _STATUS_TIMEOUT_MS.
        """
        self.status_label.setText(message)
        self._status_timer.start(_STATUS_TIMEOUT_MS)

    def _prev(self) -> None:
        """Navigate to the previous voicing."""
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def _next(self) -> None:
        """Navigate to the next voicing."""
        if self.index < len(self.voicings) - 1:
            self.index += 1
            self._render_current()

    # ------------------------------------------------------------------ rendering
    def _set_banner(self, text: str, error: bool) -> None:
        """Update the banner text and color.

        Args:
            text: Message to display.
            error: True renders the text in red, False in the default color.
        """
        self.banner.setText(text)
        self.banner.setStyleSheet("color: #cf222e;" if error else "color: #24292f;")

    def _render_empty(self) -> None:
        """Clear both charts and reset all navigation controls."""
        self._set_chips([])
        self.fingering_chart.set_voicing(None, None)
        self.note_chart.set_voicing(None, None)
        self.hint_label.setText("")
        self.counter.setText("0/0")
        self.src_tag.setText("")
        self.left_btn.setEnabled(False)
        self.right_btn.setEnabled(False)

    def _render_current(self) -> None:
        """Render the voicing at self.index into both charts."""
        if self.parsed is None:
            return
        v = self.voicings[self.index]
        self._set_chips([])
        self.fingering_chart.set_voicing(v, self.parsed)
        self.note_chart.set_voicing(v, self.parsed)
        self.hint_label.setText(self.fingering_chart.missing_data_hint())
        total = len(self.voicings)
        self.counter.setText(f"{self.index + 1}/{total}")
        src_label = {"db": "DB", "auto": "Auto"}.get(v.source, v.source)
        self.src_tag.setText(f"[{src_label}]")
        self.left_btn.setEnabled(self.index > 0)
        self.right_btn.setEnabled(self.index < total - 1)

    def _show_error(self, msg: str) -> None:
        """Display an error message in the banner and clear the diagrams.

        Args:
            msg: Error text to show in the banner.
        """
        self._set_banner(msg, error=True)
        self.voicings = []
        self.parsed = None
        self.index = 0
        self.tones_label.setText("")
        self._render_empty()

    @staticmethod
    def _chord_title(p: ParsedChord) -> str:
        """Format a ParsedChord for display in the banner.

        Args:
            p: Parsed chord object.

        Returns:
            Human-readable title string.
        """
        bass_part = f", bass={p.bass}" if p.bass else ""
        return f"{p.raw}    (root={p.root}, quality={p.quality}{bass_part})"


def main() -> None:
    """Entry point: create the QApplication and run the event loop."""
    configure_logging()
    sys.excepthook = _log_unhandled_exception
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = ChordApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
