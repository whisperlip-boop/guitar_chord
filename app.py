"""
Guitar Chord Viewer — PySide6 GUI.

Features:
  - User types a chord (e.g. C, CM7, Bbm7, F#m7/B) and presses Enter or "Show".
  - Radio buttons select voicing source: DB / Auto / Both.
  - Two side-by-side ASCII diagrams in Consolas font (left:right ≈ 3:2).
  - Left/Right arrow buttons navigate between voicings of the same chord.
  - "<current>/<total>" indicator below the diagrams.
  - At 1/N the left arrow is disabled; at N/N the right arrow is disabled.
  - Double-click either diagram to select all its text for copying.
"""

import os
import sys
import ctypes
from typing import List, Optional

from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chord_parser import parse_chord, ChordParseError, ParsedChord, chord_tones
from voicings import get_voicings, Voicing, identify_chord_from_frets
from diagram import render_left, render_right


MONO_FAMILY = "Consolas"
MONO_FONT = QFont(MONO_FAMILY, 12)
LABEL_FONT = QFont(MONO_FAMILY, 11)

# Minimum diagram widths, in characters of MONO_FONT.  Distributed across the
# available space with a 3:2 stretch ratio (left:right).
_LEFT_DIAG_CHARS = 22
_RIGHT_DIAG_CHARS = 15
_DIAG_ROWS = 7  # 6 strings + 1 fret-indicator line

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
QPlainTextEdit {
    background: white;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12pt;
}
QLabel#titleLabel { font-weight: 600; }
QLabel#srcTag { color: #57606a; }
QLabel#tonesLabel { color: #32383f; }
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


class DiagramView(QPlainTextEdit):
    """Read-only diagram text box; double-click selects all text for copying."""

    def __init__(self, min_chars: int, parent: Optional[QWidget] = None) -> None:
        """Initialise a fixed-font, read-only diagram viewer.

        Args:
            min_chars: Minimum visible width, in monospace characters.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setFont(MONO_FONT)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        metrics = QFontMetrics(MONO_FONT)
        self.setMinimumWidth(metrics.horizontalAdvance("0") * min_chars + 24)
        self.setFixedHeight(metrics.lineSpacing() * _DIAG_ROWS + 20)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Select all diagram text on double-click for quick copying."""
        self.selectAll()
        super().mouseDoubleClickEvent(event)


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
        self.resize(627, 300)
        self.setMinimumSize(600, 280)

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
        self.entry.textChanged.connect(self._on_chord_entry_changed)
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

        # ---- Banner: chord name / status message --------------------------
        self.banner = QLabel(
            "Input Chord.  e.g.: C, CM7, Bbm7, F#m7/B, C△7, C°, Cø"
        )
        self.banner.setObjectName("banner")
        root.addWidget(self.banner)

        # ---- Center: ◀  [left diagram]  [right diagram]  [frets]  ▶ -------
        center = QHBoxLayout()
        center.setSpacing(10)

        self.left_btn = QPushButton("◀")
        self.left_btn.setObjectName("navButton")
        self.left_btn.setFixedSize(36, 36)
        self.left_btn.setEnabled(False)
        self.left_btn.clicked.connect(self._prev)
        center.addWidget(self.left_btn)

        self.left_diagram = DiagramView(_LEFT_DIAG_CHARS)
        center.addWidget(self.left_diagram, 3)

        self.right_diagram = DiagramView(_RIGHT_DIAG_CHARS)
        center.addWidget(self.right_diagram, 2)

        center.addLayout(self._build_fret_inputs())

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

        bottom.addStretch(1)

        self.tones_label = QLabel("")
        self.tones_label.setObjectName("tonesLabel")
        bottom.addWidget(self.tones_label)

        root.addLayout(bottom)

    def _build_fret_inputs(self) -> QGridLayout:
        """Create the 6 fret-entry boxes (e/B/G/D/A/E, high→low), digits only.

        Returns:
            A grid layout with one label+entry row per string.
        """
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        validator = QRegularExpressionValidator(QRegularExpression(r"^[0-9]{0,2}$"))
        self.fret_entries: List[QLineEdit] = []
        for row, string_name in enumerate(("e", "B", "G", "D", "A", "E")):
            label = QLabel(string_name)
            label.setFont(LABEL_FONT)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, row, 0)

            entry = QLineEdit()
            entry.setFont(MONO_FONT)
            entry.setFixedWidth(34)
            entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            entry.setValidator(validator)
            entry.returnPressed.connect(self._show_chord)
            entry.textChanged.connect(self._on_fret_entry_changed)
            grid.addWidget(entry, row, 1)
            self.fret_entries.append(entry)

        return grid

    # --------------------------------------------------------- input handling
    def _on_chord_entry_changed(self) -> None:
        """Clear fret boxes when the user edits the chord name field.

        Keeps the two input modes mutually exclusive so Show always knows
        which source to use.
        """
        if self.entry.text().strip():
            for e in self.fret_entries:
                e.blockSignals(True)
                e.clear()
                e.blockSignals(False)

    def _on_fret_entry_changed(self) -> None:
        """Clear the chord name field when the user edits any fret box."""
        if any(e.text().strip() for e in self.fret_entries):
            self.entry.blockSignals(True)
            self.entry.clear()
            self.entry.blockSignals(False)

    def _current_source(self) -> str:
        """Return the currently selected fingering source value."""
        checked = self.source_group.checkedButton()
        return checked.property("source_value") if checked else "both"

    # ------------------------------------------------------------------ events
    def _show_chord(self) -> None:
        """Show voicings: reverse-lookup from fret boxes if any are filled, else by chord name."""
        fret_inputs: List[Optional[int]] = []
        has_fret_input = False
        for entry in self.fret_entries:
            val = entry.text().strip()
            if val:
                fret_inputs.append(int(val))
                has_fret_input = True
            else:
                fret_inputs.append(None)

        if has_fret_input:
            self._show_chord_from_frets(fret_inputs)
            return

        text = self.entry.text().strip()
        if not text:
            self._show_error("Please enter a chord name or fill in the fret boxes.")
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

    def _show_chord_from_frets(self, fret_inputs: List[Optional[int]]) -> None:
        """Identify chord from fret inputs and display its voicings.

        Args:
            fret_inputs: 6 elements (e, B, G, D, A, E). None = not entered.
        """
        result = identify_chord_from_frets(fret_inputs)
        if result is None:
            self._show_error("Not enough notes. Enter at least 2 different strings.")
            return

        parsed, is_exact = result
        voicings = get_voicings(parsed, source=self._current_source(), max_results=60)

        suffix = "" if is_exact else "   — Exact match not found."

        # Reflect identified chord name in the Input Chord entry.
        self.entry.blockSignals(True)
        self.entry.setText(parsed.raw)
        self.entry.blockSignals(False)

        self.parsed = parsed
        self.voicings = voicings
        self.index = 0
        self.tones_label.setText(f"Chord tones: {', '.join(chord_tones(parsed))}")

        if not voicings:
            self._set_banner(
                f"Chord: {self._chord_title(parsed)}"
                f"{suffix}   — No voicings found in the selected source.",
                error=True,
            )
            self._render_empty()
            return

        self._set_banner(f"Chord: {self._chord_title(parsed)}{suffix}", error=not is_exact)
        self._render_current()

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
        """Clear both diagrams and reset all navigation controls."""
        self.left_diagram.setPlainText("")
        self.right_diagram.setPlainText("")
        self.counter.setText("0/0")
        self.src_tag.setText("")
        self.left_btn.setEnabled(False)
        self.right_btn.setEnabled(False)

    def _render_current(self) -> None:
        """Render the voicing at self.index into both diagram widgets."""
        if self.parsed is None:
            return
        v = self.voicings[self.index]
        self.left_diagram.setPlainText(render_left(v))
        self.right_diagram.setPlainText(render_right(v, self.parsed))
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
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = ChordApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
