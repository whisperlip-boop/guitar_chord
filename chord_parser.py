"""
Chord input string parser.

Rules:
- First char: A-G (uppercase, required) = root note
- Optional accidental: '#' or 'b'
- Optional quality: 'M'=maj, 'm'=min, '7', 'sus2', 'sus4', 'dim', 'aug', 'add9', '6', '9', etc.
- Optional '/<bass>' for slash chord
- 'mM' (minor-major) is NOT supported per user spec.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List

# ---------------------------------------------------------------------------
# Input normalisation: convert alternate notations to the canonical ASCII form
# that the parser understands.  Applied before charset validation.
# ---------------------------------------------------------------------------

# Simple one-to-one symbol replacements (order matters: longest first).
_SYMBOL_ALIASES: list[tuple[str, str]] = [
    ("△", "M"),    # C△7 = CM7
    ("Δ", "M"),    # alternate Unicode triangle
    ("°", "dim"),  # C° = Cdim
    ("ø", "m7b5"), # Cø = Cm7b5 (half-diminished)
    ("+", "aug"),  # C+ = Caug, C+7 = Caug7
    ("6/9", "69"), # C6/9 = C69; must run before the slash-chord split ("/9" is never a valid bass)
]

# '-' immediately after a root note (A-G optionally followed by # or b) means minor.
# e.g. C-7 -> Cm7, Bb- -> Bbm
# Lookbehind can't handle variable-width [#b]?, so we capture the note prefix instead.
_MINOR_DASH_RE = re.compile(r"([A-G][#b]?)-")

# Full-word quality spellings (case-insensitive). Longest first to prevent
# partial matches (e.g. "minor" before "min" would not overlap, but order
# is kept consistent with replace priority).
_WORD_ALIAS_RE = re.compile(r"augmented|diminished|major|minor", re.IGNORECASE)
_WORD_ALIAS_MAP: dict[str, str] = {
    "augmented":  "aug",
    "diminished": "dim",
    "major":      "maj",
    "minor":      "min",
}


def _preprocess_input(s: str) -> str:
    """Normalize alternate chord notations to the canonical ASCII input format.

    Supported aliases (examples):
        C△7, CΔ7          -> CM7       (major-7 triangle symbol)
        C°, C°7           -> Cdim, Cdim7
        Cø                -> Cm7b5     (half-diminished)
        C+, C+7           -> Caug, Caug7
        C-7, Bb-          -> Cm7, Bbm  (dash for minor)
        C6/9, Cm6/9       -> C69, Cm69
        Cmajor7, Cminor7  -> Cmaj7, Cmin7
        Caugmented        -> Caug
        Cdiminished       -> Cdim

    Args:
        s: Raw (stripped) chord string from user input.

    Returns:
        Equivalent string using only the ASCII chars the parser accepts.
    """
    for old, new in _SYMBOL_ALIASES:
        s = s.replace(old, new)
    s = _MINOR_DASH_RE.sub(lambda m: m.group(1) + "m", s)
    s = _WORD_ALIAS_RE.sub(lambda m: _WORD_ALIAS_MAP[m.group().lower()], s)
    return s


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Map note string -> semitone offset from C (0..11)
NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}


@dataclass
class ParsedChord:
    """Parsed chord representation."""
    raw: str                        # Original input
    root: str                       # e.g. "C", "Bb", "F#"
    root_semitone: int              # 0..11
    quality: str                    # canonical quality token, e.g. "maj", "min", "7", "maj7", "min7", "sus4", "dim", "aug", "add9", ...
    bass: Optional[str] = None      # e.g. "G" for C/G
    bass_semitone: Optional[int] = None
    intervals: List[int] = field(default_factory=list)   # semitone offsets from root for chord tones
    display_use_flat: bool = False  # if root uses 'b', display chord tones with flats


# Quality token -> intervals (semitones from root)
# Order matters for parsing (longer tokens first)
QUALITY_DEFS = [
    # token,        intervals
    ("maj7",  [0, 4, 7, 11]),
    ("maj9",  [0, 4, 7, 11, 14]),
    ("min7",  [0, 3, 7, 10]),
    ("min9",  [0, 3, 7, 10, 14]),
    ("sus2",  [0, 2, 7]),
    ("sus4",  [0, 5, 7]),
    ("add9",  [0, 4, 7, 14]),
    ("dim7",   [0, 3, 6, 9]),
    ("min7b5", [0, 3, 6, 10]),   # half-diminished (ø)
    ("aug7",   [0, 4, 8, 10]),
    ("dim",    [0, 3, 6]),
    ("aug",   [0, 4, 8]),
    ("maj",   [0, 4, 7]),
    ("min",   [0, 3, 7]),
    ("13",    [0, 4, 7, 10, 14, 21]),
    ("11",    [0, 4, 7, 10, 14, 17]),
    ("9",     [0, 4, 7, 10, 14]),
    ("7",     [0, 4, 7, 10]),
    ("6",     [0, 4, 7, 9]),
    ("5",     [0, 7]),                 # power chord
    # Qualities present in the external datasets. Appended (not interleaved) so
    # identify_chord_from_frets keeps its existing tie-breaking order.
    ("min6",     [0, 3, 7, 9]),
    ("min11",    [0, 3, 7, 10, 14, 17]),
    ("minadd9",  [0, 3, 7, 14]),
    ("min69",    [0, 3, 7, 9, 14]),
    ("add11",    [0, 4, 7, 17]),
    ("69",       [0, 4, 7, 9, 14]),
    ("7sus4",    [0, 5, 7, 10]),
    ("7sus2",    [0, 2, 7, 10]),
    ("sus2sus4", [0, 2, 5, 7]),
    ("7b5",      [0, 4, 6, 10]),
    ("7b9",      [0, 4, 7, 10, 13]),
    ("7#9",      [0, 4, 7, 10, 15]),
    ("9b5",      [0, 4, 6, 10, 14]),
    ("9#11",     [0, 4, 7, 10, 14, 18]),
    ("aug9",     [0, 4, 8, 10, 14]),
    ("maj7b5",   [0, 4, 6, 11]),
    ("maj7#5",   [0, 4, 8, 11]),
    ("maj7sus2", [0, 2, 7, 11]),
    ("maj11",    [0, 4, 7, 11, 14, 17]),
    ("maj13",    [0, 4, 7, 11, 14, 21]),  # 11th omitted, same convention as "13"
]


class ChordParseError(ValueError):
    pass


def _parse_note(s: str, pos: int) -> tuple[str, int, int]:
    """Parse a single note starting at s[pos]. Returns (note_str, semitone, new_pos)."""
    if pos >= len(s):
        raise ChordParseError("Empty note.")
    ch = s[pos]
    if ch not in "ABCDEFG":
        raise ChordParseError(f"Note must start with A-G (uppercase). Got '{ch}'.")
    note = ch
    pos += 1
    if pos < len(s) and s[pos] in "#b":
        note += s[pos]
        pos += 1
    if note not in NOTE_TO_SEMITONE:
        raise ChordParseError(f"Unknown note '{note}'.")
    return note, NOTE_TO_SEMITONE[note], pos


def _parse_quality(s: str, pos: int) -> tuple[str, List[int], int]:
    """
    Parse the quality portion starting at s[pos].
    Returns (canonical_quality_token, intervals, new_pos).
    Stops at '/' or end-of-string.
    """
    # Find the end (either '/' or end of string)
    end = s.find("/", pos)
    if end == -1:
        end = len(s)
    body = s[pos:end]

    # Empty body => major triad
    if body == "":
        return "maj", [0, 4, 7], end

    # Translate user shorthand to canonical tokens.
    # Special rules:
    #   'M' (uppercase) at start of body means 'maj'  (e.g. CM, CM7)
    #   'm' (lowercase) at start of body means 'min'  (e.g. Cm, Cm7)
    #   bare 'M' / 'm' (whole body) => maj / min triad
    if body == "M":
        return "maj", [0, 4, 7], end
    if body == "m":
        return "min", [0, 3, 7], end

    # CM7 -> maj7, Cm7 -> min7, CM9 -> maj9, etc.
    normalized = body
    if normalized.startswith("M") and len(normalized) > 1 and normalized[1].isdigit():
        # Disallow 'mM' combo per spec; here we only handle starts-with-M.
        normalized = "maj" + normalized[1:]
    elif normalized.startswith("m") and len(normalized) > 1 and normalized[1].isdigit():
        normalized = "min" + normalized[1:]
    elif normalized.startswith("madd"):
        # Cmadd9 -> minadd9 ("maj..." never starts with "madd", so no clash).
        normalized = "min" + normalized[1:]

    # Match against QUALITY_DEFS (longest token first by definition order).
    for token, intervals in QUALITY_DEFS:
        if normalized == token:
            return token, list(intervals), end

    raise ChordParseError(f"Unsupported chord quality: '{body}'.")


# Reject pure double-flats / double-sharps and 'mM' (minor-major) combos.
_INVALID_PATTERNS = [
    re.compile(r"bb"),       # double flat
    re.compile(r"##"),       # double sharp
    re.compile(r"mM"),       # minor-major combo, explicitly excluded by user
]


def _validate_charset(s: str) -> None:
    """Allowed chars: A-Z, a-z (but only specific letters used by rules), digits, '#', '/'."""
    if not s:
        raise ChordParseError("Empty input.")
    for ch in s:
        if not (ch.isalnum() or ch in "#/"):
            raise ChordParseError(f"Invalid character: '{ch}'. Allowed: digits, English letters, '#', '/'.")


def parse_chord(text: str) -> ParsedChord:
    """Parse a chord string into a ParsedChord.

    Accepts canonical ASCII notation and common alternate forms.
    See _preprocess_input for the full alias list.
    Examples: 'C', 'CM7', 'C△7', 'Cmaj7', 'Cmajor7', 'Bbm7', 'C-7', 'F#m7/B'.

    Args:
        text: User-supplied chord string (may contain Unicode music symbols).

    Returns:
        ParsedChord with root, quality, intervals, and optional bass note.

    Raises:
        ChordParseError: If the string cannot be interpreted as a valid chord.
    """
    s = text.strip()
    s = _preprocess_input(s)  # normalise aliases before validation
    _validate_charset(s)

    # Reject 'bb', '##', 'mM' anywhere (per user rules).
    for pat in _INVALID_PATTERNS:
        if pat.search(s):
            raise ChordParseError(
                f"Invalid sequence found in '{s}'. Double accidentals (bb/##) and minor-major (mM) are not supported."
            )

    pos = 0
    root, root_semi, pos = _parse_note(s, pos)
    quality, intervals, pos = _parse_quality(s, pos)

    bass = None
    bass_semi = None
    if pos < len(s):
        if s[pos] != "/":
            raise ChordParseError(f"Unexpected character '{s[pos]}' at position {pos}.")
        pos += 1
        bass, bass_semi, pos = _parse_note(s, pos)
        if pos != len(s):
            raise ChordParseError(f"Extra characters after bass note: '{s[pos:]}'.")

    use_flat = "b" in root or (bass is not None and "b" in bass)

    return ParsedChord(
        raw=text,
        root=root,
        root_semitone=root_semi,
        quality=quality,
        bass=bass,
        bass_semitone=bass_semi,
        intervals=intervals,
        display_use_flat=use_flat,
    )


def semitone_to_name(semitone: int, use_flat: bool = False) -> str:
    """Convert semitone (0..11, where 0=C) to note name string."""
    semitone %= 12
    return (NOTE_NAMES_FLAT if use_flat else NOTE_NAMES_SHARP)[semitone]


def chord_tones(parsed: ParsedChord) -> List[str]:
    """Return chord tones as note name strings."""
    return [
        semitone_to_name((parsed.root_semitone + iv) % 12, parsed.display_use_flat)
        for iv in parsed.intervals
    ]


if __name__ == "__main__":
    # Quick smoke test — canonical notation
    tests = ["C", "CM", "Cm", "C7", "CM7", "Cm7", "Bb", "Bbm7", "F#", "F#m7/B", "Csus4", "Cdim", "Caug", "Cadd9"]
    for t in tests:
        try:
            p = parse_chord(t)
            print(f"{t:12s} -> root={p.root}, quality={p.quality}, bass={p.bass}, tones={chord_tones(p)}")
        except ChordParseError as e:
            print(f"{t:12s} -> ERROR: {e}")

    # Alias notation smoke test
    alias_tests = [
        ("C△7",      "CM7"),
        ("CΔ7",      "CM7"),
        ("Cmaj7",    "CM7"),
        ("Cmajor7",  "CM7"),
        ("C-7",      "Cm7"),
        ("Cminor7",  "Cm7"),
        ("C°",       "Cdim"),
        ("C°7",      "Cdim7"),
        ("Cø",       "Cm7b5"),
        ("C+",       "Caug"),
        ("C+7",      "Caug7"),
        ("Bb-",      "Bbm"),
    ]
    print("\n--- alias tests ---")
    for alias, canonical in alias_tests:
        try:
            pa = parse_chord(alias)
            pc = parse_chord(canonical)
            ok = pa.quality == pc.quality and pa.root_semitone == pc.root_semitone
            print(f"  {alias:15s} -> quality={pa.quality}  {'OK' if ok else 'MISMATCH with ' + canonical}")
        except ChordParseError as e:
            print(f"  {alias:15s} -> ERROR: {e}")

    # Negative tests
    for t in ["Hb", "CmM7", "Cbb", "C##", "c", "X/G", "C/", "/G", ""]:
        try:
            p = parse_chord(t)
            print(f"{t:12s} -> UNEXPECTED OK: {p}")
        except ChordParseError as e:
            print(f"{t:12s} -> rejected: {e}")
