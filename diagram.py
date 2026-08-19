"""
Render two ASCII diagrams for a Voicing:

  LEFT:  fretboard with 'o' markers
  RIGHT: per-string fret info "n(NOTE)" or "(NOTE)" for open, blank for muted

Layout rule (user-defined):
  Each row starts with the string name, then ONE column for o/x/space,
  then the diagram body. That way o/x never shifts the body alignment.

Left:
  e |---|---|---|---|
  eo|---|---|---|---|
  Ex|---|---|---|---|

Right:
  e |------|
  eo|--(E)-|
  E |-1(C)-|
"""

from typing import List, Optional, Tuple
from voicings import Voicing, STRING_NAMES, OPEN_STRING_SEMITONES
from chord_parser import ParsedChord, semitone_to_name


# How many frets to show in the left diagram.
# Defaults to 3, expands up to 5 if a fingered fret requires it.
DEFAULT_FRET_COUNT = 3
MAX_FRET_COUNT = 5


def _decide_render_window(v: Voicing) -> Tuple[int, int]:
    """Return (start_fret, num_frets_displayed).

    Policy: default 3 frets; expand to 4 or at most 5 as needed to show
    all fingered positions.
    """
    fingered = v.fingered_frets()
    if not fingered:
        return 1, DEFAULT_FRET_COUNT
    lo, hi = min(fingered), max(fingered)
    span = hi - lo + 1  # minimum frets needed to cover all positions

    if hi <= MAX_FRET_COUNT:
        # Low/mid position: start at fret 1 and expand enough to include hi.
        return 1, max(DEFAULT_FRET_COUNT, hi)
    # High position: start at lo, cover actual span but cap at MAX_FRET_COUNT.
    return lo, min(MAX_FRET_COUNT, max(DEFAULT_FRET_COUNT, span))


def _left_marker(f: int) -> str:
    """Marker for the open/mute column at the start of the diagram."""
    if f == -1:
        return "x"
    if f == 0:
        return "o"
    return " "


def render_left(v: Voicing) -> str:
    """Render the fretboard diagram (left side)."""
    start, n = _decide_render_window(v)
    lines: List[str] = []

    for s_idx, name in enumerate(STRING_NAMES):
        marker = _left_marker(v.frets[s_idx])
        cells: List[str] = []
        for i in range(n):
            fret = start + i
            if v.frets[s_idx] == fret:
                cells.append("-o-")
            else:
                cells.append("---")
        # Build: "e |---|---|---|"
        body = "|" + "|".join(cells) + "|"
        lines.append(f"{name}{marker}{body}")

    # Optional starting-fret indicator line at the bottom
    if start != 1:
        # Place the number under the first fret column.
        # The diagram starts at column index = len(name)+1 = 2, then "|", then cell of width 3.
        # Center of first cell is at column 2 + 1 + 1 = 4 (0-indexed), but text easier:
        # Format: 3 chars indent, then number under cell 1.
        # Format per row: name(1) + marker(1) + "|"(1) + cell(3) + ...
        # Center of the first cell is at index 4.  prefix covers name+marker (2),
        # so indent_to_center needs 2 more chars to land at position 4.
        prefix = "  "
        indent_to_center = " " * 2
        lines.append(prefix + indent_to_center + str(start))
    return "\n".join(lines)


def _note_at(string_idx: int, fret: int) -> int:
    """Pitch class produced by playing a string at a given fret."""
    return (OPEN_STRING_SEMITONES[string_idx] + fret) % 12


def render_right(v: Voicing, parsed: ParsedChord) -> str:
    """Render the per-string note-and-fret info (right side).

    Format per row:
      "<name><marker>|<cell>|"
    where <cell> is one of:
      - "-n(NN)-"  for a fingered fret (NN may be 1-2 chars; we keep cell width consistent)
      - "--(NN)-"  for open string (no fret number)
      - "------"   for muted string

    Cell width adapts to the largest content (e.g. "12(C#)" needs width 6+
    with dashes around). For two-digit frets and accidentals, width may be 7.
    """
    use_flat = parsed.display_use_flat

    # Pre-compute the inner texts and find the max length.
    inners: List[Optional[str]] = []
    for s_idx in range(6):
        f = v.frets[s_idx]
        if f == -1:
            inners.append(None)
        elif f == 0:
            note = semitone_to_name(_note_at(s_idx, 0), use_flat=use_flat)
            inners.append(f"({note})")
        else:
            note = semitone_to_name(_note_at(s_idx, f), use_flat=use_flat)
            inners.append(f"{f}({note})")

    # Cell width: at least 6, plus 2 dashes around the longest inner.
    max_inner_len = max((len(t) for t in inners if t is not None), default=4)
    cell_width = max(6, max_inner_len + 2)

    rows: List[str] = []
    for s_idx, name in enumerate(STRING_NAMES):
        marker = _left_marker(v.frets[s_idx])
        inner = inners[s_idx]
        if inner is None:
            cell = "-" * cell_width
        else:
            cell = _center_pad(inner, cell_width, pad="-")
        rows.append(f"{name}{marker}|{cell}|")
    return "\n".join(rows)


def _center_pad(text: str, width: int, pad: str = "-") -> str:
    """Center `text` in a field of `width` using `pad`.

    When padding is asymmetric (odd remainder), the extra character goes on the
    LEFT so that the note letter visually aligns with fingered-fret entries.
    e.g. _center_pad("(E)", 6) -> "--(E)-" (not "-(E)--")
    """
    if len(text) >= width:
        return text[:width]
    total = width - len(text)
    right = total // 2      # smaller half on right
    left = total - right    # larger half on left
    return pad * left + text + pad * right


def render_both(v: Voicing, parsed: ParsedChord, gap: int = 4) -> str:
    """Render left and right diagrams side by side, aligned line-by-line."""
    left = render_left(v).split("\n")
    right = render_right(v, parsed).split("\n")

    # Pad both to the same number of lines
    n = max(len(left), len(right))
    left += [""] * (n - len(left))
    right += [""] * (n - len(right))

    left_width = max(len(line) for line in left) if left else 0
    spacer = " " * gap
    out_lines = []
    for line_l, line_r in zip(left, right):
        out_lines.append(f"{line_l:<{left_width}}{spacer}{line_r}")
    return "\n".join(out_lines)


if __name__ == "__main__":
    from chord_parser import parse_chord
    from voicings import get_voicings

    for s in ["C", "G", "F", "Am", "CM7", "C/G", "Bbm7"]:
        p = parse_chord(s)
        vs = get_voicings(p, source="both", max_results=4)
        print(f"\n========== {s} ==========")
        for i, v in enumerate(vs):
            print(f"\n--- voicing {i+1}/{len(vs)} (source={v.source}, base_fret={v.base_fret}) ---")
            print(render_both(v, p))
