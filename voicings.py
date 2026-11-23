"""
Guitar voicing (fingering) model and generator.

A Voicing represents one way to play a chord on a 6-string guitar.
Strings are indexed 0..5 from HIGH to LOW (e, B, G, D, A, E) for display convenience.

Each string has a value:
  -1  -> muted (x)
   0  -> open (o)
   n  -> pressed at fret n (1..)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from chord_parser import (
    ParsedChord, semitone_to_name, NOTE_TO_SEMITONE,
    NOTE_NAMES_SHARP, QUALITY_DEFS,
)
from external_data import Barre, load_all_external

logger = logging.getLogger(__name__)


# Standard tuning. Index 0 = high e (1st string), Index 5 = low E (6th string).
# Semitone values are absolute (C4 = 60-style midi-like), but we only care about pitch class (mod 12).
# Open string pitch classes:
#   high e -> E (4)
#   B      -> B (11)
#   G      -> G (7)
#   D      -> D (2)
#   A      -> A (9)
#   low E  -> E (4)
STRING_NAMES = ["e", "B", "G", "D", "A", "E"]   # display order: top (high e) to bottom (low E)
OPEN_STRING_SEMITONES = [4, 11, 7, 2, 9, 4]      # pitch class of each open string


@dataclass
class Voicing:
    """A single chord voicing.

    frets[i] is for STRING_NAMES[i]:
      -1 = muted
       0 = open
       n = fret n

    fingers and barres come from external data only and are None when unknown
    (never inferred); they do not take part in signature().
    """
    frets: List[int]                         # length 6, index 0 = high e
    source: str = "db"                       # "db" or "auto"
    base_fret: int = 1                       # starting fret to render (1 if includes open/low frets)
    fingers: Optional[Tuple[int, ...]] = None  # length 6, index 0 = high e; 0 = no finger, 1-4 = finger
    barres: Optional[Tuple[Barre, ...]] = None  # () = data says no barre

    def fingered_frets(self) -> List[int]:
        return [f for f in self.frets if f > 0]

    def max_fret(self) -> int:
        ff = self.fingered_frets()
        return max(ff) if ff else 0

    def min_fingered_fret(self) -> int:
        ff = self.fingered_frets()
        return min(ff) if ff else 0

    def signature(self) -> Tuple[int, ...]:
        return tuple(self.frets)


# ---------------------------------------------------------------------------
# Voicing DB (hand-curated common voicings)
# Keys are canonical chord names like "C", "Cmaj", "Cm", "C7", "CM7", etc.
# We store them under (root, quality) form internally.
# ---------------------------------------------------------------------------

# Each entry: (frets_high_e_to_low_E,)
# Order: e, B, G, D, A, E
VOICING_DB: dict[Tuple[str, str, Optional[str]], List[List[int]]] = {
    # Major triads (open shapes)
    ("C", "maj", None): [
        [0, 1, 0, 2, 3, -1],     # open C
        [3, 5, 5, 5, 3, -1],     # A-shape barre at 3
        [8, 8, 9, 10, 10, 8],    # E-shape barre at 8
    ],
    ("D", "maj", None): [
        [2, 3, 2, 0, -1, -1],    # open D
        [5, 7, 7, 7, 5, -1],     # A-shape barre at 5
        [10, 10, 11, 12, 12, 10],
    ],
    ("E", "maj", None): [
        [0, 0, 1, 2, 2, 0],      # open E
        [7, 9, 9, 9, 7, -1],     # A-shape barre at 7
    ],
    ("F", "maj", None): [
        [1, 1, 2, 3, 3, 1],      # E-shape barre at 1
        [8, 10, 10, 10, 8, -1],  # A-shape barre at 8
    ],
    ("G", "maj", None): [
        [3, 0, 0, 0, 2, 3],      # open G
        [3, 3, 4, 5, 5, 3],      # E-shape barre at 3
        [10, 12, 12, 12, 10, -1],# A-shape barre at 10
    ],
    ("A", "maj", None): [
        [0, 2, 2, 2, 0, -1],     # open A
        [5, 5, 6, 7, 7, 5],      # E-shape barre at 5
    ],
    ("B", "maj", None): [
        [2, 4, 4, 4, 2, -1],     # A-shape barre at 2
        [7, 7, 8, 9, 9, 7],      # E-shape barre at 7
    ],

    # Minor triads (open shapes)
    ("A", "min", None): [
        [0, 1, 2, 2, 0, -1],     # open Am
        [5, 5, 5, 7, 7, 5],      # E-shape minor barre at 5
    ],
    ("D", "min", None): [
        [1, 3, 2, 0, -1, -1],    # open Dm
        [5, 6, 7, 7, 5, -1],     # A-shape minor barre at 5 (Am shape moved to D)
    ],
    ("E", "min", None): [
        [0, 0, 0, 2, 2, 0],      # open Em
        [7, 8, 9, 9, 7, -1],     # A-shape minor barre at 7
    ],
    ("C", "min", None): [
        [3, 4, 5, 5, 3, -1],     # A-shape minor barre at 3
        [8, 8, 8, 10, 10, 8],    # E-shape minor barre at 8
    ],
    ("F", "min", None): [
        [1, 1, 1, 3, 3, 1],      # E-shape minor barre at 1
    ],
    ("G", "min", None): [
        [3, 3, 3, 5, 5, 3],      # E-shape minor barre at 3
    ],
    ("B", "min", None): [
        [2, 3, 4, 4, 2, -1],     # A-shape minor barre at 2
    ],

    # 7th chords (open)
    ("C", "7", None):  [[0, 1, 3, 2, 3, -1]],   # C7 open
    ("D", "7", None):  [[2, 1, 2, 0, -1, -1]],  # D7 open
    ("E", "7", None):  [[0, 3, 1, 0, 2, 0]],    # E7 open (one common voicing)
    ("G", "7", None):  [[1, 0, 0, 0, 2, 3]],    # G7 open
    ("A", "7", None):  [[0, 2, 0, 2, 0, -1]],   # A7 open
    ("B", "7", None):  [[2, 0, 2, 1, 2, -1]],   # B7 open

    # Major 7th (open)
    ("C", "maj7", None): [[0, 0, 0, 2, 3, -1]],
    ("D", "maj7", None): [[2, 2, 2, 0, -1, -1]],
    ("G", "maj7", None): [[2, 0, 0, 0, 2, 3]],
    ("A", "maj7", None): [[0, 2, 1, 2, 0, -1]],

    # Minor 7th (open)
    ("A", "min7", None): [[0, 1, 0, 2, 0, -1]],
    ("D", "min7", None): [[1, 1, 2, 0, -1, -1]],
    ("E", "min7", None): [[0, 3, 0, 2, 2, 0]],
}


# ---------------------------------------------------------------------------
# External data integration
#
# At import time we try to load any JSON files placed next to the program in
# the data/ directory (currently chords-db and T-vK formats; see external_data.py).
# Each external entry is keyed under (root, quality, bass) like the built-in
# DB so db_lookup can serve them uniformly.
# ---------------------------------------------------------------------------

# Value tuples: (fret signature, source tag, fingers or None, barres or None).
_EXTERNAL_INDEX: dict[
    Tuple[str, str, Optional[str]],
    List[Tuple[Tuple[int, ...], str, Optional[Tuple[int, ...]], Optional[Tuple[Barre, ...]]]],
] = {}
_EXTERNAL_LOADED: bool = False

_QUALITY_INTERVALS: dict[str, List[int]] = dict(QUALITY_DEFS)
# A 13th chord legitimately contains the 11th; QUALITY_DEFS merely omits it.
_EXTRA_ALLOWED_INTERVALS: dict[str, List[int]] = {"13": [17], "maj13": [17]}


def _allowed_pitch_classes(root: str, quality: str, bass: Optional[str]) -> Optional[frozenset[int]]:
    """Chord tones plus slash bass for a key, or None when the parser cannot produce it."""
    intervals = _QUALITY_INTERVALS.get(quality)
    root_pc = NOTE_TO_SEMITONE.get(root)
    if intervals is None or root_pc is None:
        return None
    allowed = {(root_pc + iv) % 12 for iv in intervals + _EXTRA_ALLOWED_INTERVALS.get(quality, [])}
    if bass is not None and bass in NOTE_TO_SEMITONE:
        allowed.add(NOTE_TO_SEMITONE[bass])
    return frozenset(allowed)


def _fits_chord(frets: List[int], allowed: Optional[frozenset[int]]) -> bool:
    """Return True if every sounded pitch class is in `allowed` (from _allowed_pitch_classes).

    allowed=None (quality or root the parser cannot produce) always passes; those
    entries are unreachable anyway.
    """
    if allowed is None:
        return True
    return all((OPEN_STRING_SEMITONES[i] + f) % 12 in allowed for i, f in enumerate(frets) if f >= 0)


def _build_external_index() -> None:
    """Load external JSON datasets once and index them by (root, quality, bass).

    Failures (malformed or unreadable files) are logged; the program still works
    with just the built-in DB and the generator. Voicings that sound notes outside
    their chord (dataset errors) are dropped and counted in one log record.
    """
    global _EXTERNAL_LOADED
    if _EXTERNAL_LOADED:
        return
    _EXTERNAL_LOADED = True
    try:
        entries = load_all_external()
    except Exception:
        logger.exception("Failed to load external chord data; using built-in DB only.")
        return

    seen_per_key: dict[Tuple[str, str, Optional[str]], set] = {}
    allowed_per_key: dict[Tuple[str, str, Optional[str]], Optional[frozenset[int]]] = {}
    dropped = 0
    for e in entries:
        key = (e["root"], e["quality"], e["bass"])
        if key not in allowed_per_key:
            allowed_per_key[key] = _allowed_pitch_classes(*key)
        if not _fits_chord(e["frets"], allowed_per_key[key]):
            dropped += 1
            continue
        sig = tuple(e["frets"])
        seen = seen_per_key.setdefault(key, set())
        if sig in seen:
            continue
        seen.add(sig)
        _EXTERNAL_INDEX.setdefault(key, []).append(
            (sig, e["source"], e.get("fingers"), e.get("barres"))
        )
    if dropped:
        logger.warning("Dropped %d external voicings that sound notes outside their chord", dropped)


def db_lookup(parsed: ParsedChord) -> List[Voicing]:
    """Look up voicings by (root, quality, bass) across built-in DB + external JSON datasets.

    Built-in DB voicings come first (they're hand-curated and tend to be the most
    canonical), followed by external sources (chords-db, T-vK). Duplicates
    sharing the same fret signature are filtered.
    """
    _build_external_index()

    # Normalize root to sharp form for DB lookup (DBs are keyed in sharps).
    root_norm = semitone_to_name(parsed.root_semitone, use_flat=False)
    bass_norm = (
        semitone_to_name(parsed.bass_semitone, use_flat=False)
        if parsed.bass_semitone is not None else None
    )
    key = (root_norm, parsed.quality, bass_norm)

    voicings: List[Voicing] = []
    seen: set[Tuple[int, ...]] = set()
    external = _EXTERNAL_INDEX.get(key, [])
    # Built-in shapes have no fingering; borrow it from the identical external shape.
    external_fingering = {sig: (fingers, barres) for sig, _, fingers, barres in external}

    # 1. Built-in curated DB
    for frets in VOICING_DB.get(key, []):
        sig = tuple(frets)
        if sig in seen:
            continue
        seen.add(sig)
        fingers, barres = external_fingering.get(sig, (None, None))
        v = Voicing(frets=list(frets), source="db", fingers=fingers, barres=barres)
        v.base_fret = _compute_base_fret(v)
        voicings.append(v)

    # 2. External datasets
    for sig, source_tag, fingers, barres in external:
        if sig in seen:
            continue
        seen.add(sig)
        v = Voicing(frets=list(sig), source=source_tag, fingers=fingers, barres=barres)
        v.base_fret = _compute_base_fret(v)
        voicings.append(v)

    # Sort: lower position first, fewer mutes, fewer fingered frets
    voicings.sort(key=lambda v: (
        v.min_fingered_fret() if v.fingered_frets() else 0,
        sum(1 for f in v.frets if f == -1),
        len(v.fingered_frets()),
        v.max_fret(),
    ))
    return voicings


# ---------------------------------------------------------------------------
# Automatic voicing generator
#
# Approach:
#   1. Compute target pitch-class set from chord intervals.
#   2. For each string, enumerate candidate frets in window [start..start+span-1] plus optional open (0) and muted (-1).
#   3. Filter combinations whose played notes cover all required pitch classes
#      and contain no notes outside the set.
#   4. Apply playability constraints:
#        - At least 4 strings played
#        - Span of fingered frets <= 4
#        - Bass note (lowest pitch) is the root (or specified bass for slash chords)
#        - No "hole": a played string can't have muted strings above it that
#          would be hit by a strum unless we mark them muted; we allow that
#          but limit total muted strings to keep voicings clean.
# ---------------------------------------------------------------------------

def _compute_base_fret(v: Voicing) -> int:
    """Choose the starting fret for display.
    Default 1; if all fingered frets are >5 we shift up.
    """
    ff = v.fingered_frets()
    if not ff:
        return 1
    if max(ff) <= 5:
        return 1
    # Use min fingered fret as base
    return min(ff)


def generate_voicings(parsed: ParsedChord, max_results: int = 12) -> List[Voicing]:
    """Algorithmically generate playable voicings for the parsed chord."""
    # Required pitch classes
    root_pc = parsed.root_semitone
    chord_pcs: Set[int] = {(root_pc + iv) % 12 for iv in parsed.intervals}

    # Bass requirement
    if parsed.bass_semitone is not None:
        bass_pc = parsed.bass_semitone % 12
    else:
        bass_pc = root_pc

    results: List[Voicing] = []
    seen: Set[Tuple[int, ...]] = set()

    # Search across windows of 4 frets starting from 0 (open) to 12.
    for start in range(0, 13):
        span = 4  # fingerable span
        # For each string, choices are: -1 (mute), 0 (open if 0 is in window or always allowed),
        # or any fret in [start..start+span-1] (>=1).
        # We include open (0) always; bare open is musically natural even when window is high.
        per_string_choices: List[List[int]] = []
        for _ in range(6):
            choices = [-1, 0]
            for fret in range(max(1, start), start + span):
                choices.append(fret)
            per_string_choices.append(choices)

        _search(per_string_choices, chord_pcs, bass_pc, results, seen, max_results)
        if len(results) >= max_results:
            break

    # Mark source + compute base_fret
    for v in results:
        v.source = "auto"
        v.base_fret = _compute_base_fret(v)

    # Sort: prefer lower position, more strings played, fewer mutes
    results.sort(key=lambda v: (
        v.min_fingered_fret() if v.fingered_frets() else 0,
        -sum(1 for f in v.frets if f >= 0),       # more played strings first
        sum(1 for f in v.frets if f == -1),       # fewer mutes first
        v.max_fret(),
    ))
    return results[:max_results]


def _search(
    per_string_choices: List[List[int]],
    required_pcs: Set[int],
    bass_pc: int,
    results: List[Voicing],
    seen: Set[Tuple[int, ...]],
    max_results: int,
) -> None:
    """Backtracking search across the 6 strings."""
    current: List[int] = []

    def fret_to_pc(string_idx: int, fret: int) -> int:
        return (OPEN_STRING_SEMITONES[string_idx] + fret) % 12

    def backtrack(idx: int) -> None:
        if len(results) >= max_results:
            return
        if idx == 6:
            _evaluate(current[:], required_pcs, bass_pc, results, seen)
            return
        for choice in per_string_choices[idx]:
            if choice >= 0:
                pc = fret_to_pc(idx, choice)
                if pc not in required_pcs:
                    continue
            current.append(choice)
            backtrack(idx + 1)
            current.pop()

    backtrack(0)


def _evaluate(
    frets: List[int],
    required_pcs: Set[int],
    bass_pc: int,
    results: List[Voicing],
    seen: Set[Tuple[int, ...]],
) -> None:
    """Check if `frets` is a valid playable voicing."""
    played_indices = [i for i, f in enumerate(frets) if f >= 0]
    if len(played_indices) < 4:
        return

    # Pitch classes actually produced
    produced: Set[int] = set()
    for i in played_indices:
        produced.add((OPEN_STRING_SEMITONES[i] + frets[i]) % 12)
    if produced != required_pcs:
        return

    # Bass = lowest played string's pitch (string index 5 is lowest E)
    # played_indices in our model are 0..5 with 5 being the lowest (low E).
    lowest_idx = max(played_indices)
    bass_produced = (OPEN_STRING_SEMITONES[lowest_idx] + frets[lowest_idx]) % 12
    if bass_produced != bass_pc:
        return

    # Fingered fret span check
    fingered = [f for f in frets if f > 0]
    if fingered:
        span = max(fingered) - min(fingered) + 1
        if span > 4:
            return

    # No "muted hole" between played strings (looks awkward to strum).
    # Allow at most 1 muted string between two played strings.
    # played_indices sorted ascending
    played_sorted = sorted(played_indices)
    for a, b in zip(played_sorted, played_sorted[1:]):
        if b - a > 2:
            return

    sig = tuple(frets)
    if sig in seen:
        return
    seen.add(sig)
    results.append(Voicing(frets=list(frets)))


# ---------------------------------------------------------------------------
# Public API: produce voicings by source choice
# ---------------------------------------------------------------------------

def get_voicings(parsed: ParsedChord, source: str = "both", max_results: int = 60) -> List[Voicing]:
    """Return voicings for the chord.

    source:
      "db"   -> only DB (built-in + external JSON datasets)
      "auto" -> only generator
      "both" -> DB first, then auto (deduplicated by fret signature)

    max_results caps the total number returned; DB voicings are not truncated
    until this cap, so increase it if you want even more.
    """
    if source == "db":
        return db_lookup(parsed)[:max_results]
    if source == "auto":
        return generate_voicings(parsed, max_results=max_results)

    # both
    db_voicings = db_lookup(parsed)
    auto_voicings = generate_voicings(parsed, max_results=max_results)

    seen = {v.signature() for v in db_voicings}
    merged = list(db_voicings)
    for v in auto_voicings:
        if v.signature() in seen:
            continue
        seen.add(v.signature())
        merged.append(v)
    return merged[:max_results]


# ---------------------------------------------------------------------------
# Reverse lookup: fingering → chord
# ---------------------------------------------------------------------------

# Lower value = higher preference when multiple fallback candidates tie on other scores.
# Keeps common triads/7ths above extensions and altered chords.
_QUALITY_FALLBACK_PRIORITY: dict[str, int] = {
    "maj": 0, "min": 0,
    "7": 1, "maj7": 1, "min7": 1,
    "sus2": 2, "sus4": 2, "6": 2, "add9": 2, "9": 2,
    "maj9": 3, "min9": 3, "dim7": 3, "min7b5": 3,
    "dim": 4, "aug": 4, "aug7": 4,
    "11": 5, "13": 5, "5": 6,
}

# Maps internal quality token to user-facing suffix (e.g. "maj" → "" → "C").
_QUALITY_DISPLAY: dict[str, str] = {
    "maj": "",      "min": "m",     "7": "7",       "maj7": "M7",
    "min7": "m7",   "maj9": "M9",   "min9": "m9",   "sus2": "sus2",
    "sus4": "sus4", "add9": "add9", "dim7": "dim7",  "min7b5": "m7b5",
    "aug7": "aug7", "dim": "dim",   "aug": "aug",    "13": "13",
    "11": "11",     "9": "9",       "6": "6",        "5": "5",
    "min6": "m6",   "min11": "m11", "minadd9": "madd9", "min69": "m69",
    "maj7b5": "M7b5", "maj7#5": "M7#5", "maj7sus2": "M7sus2",
    "maj11": "M11", "maj13": "M13",
}

# Fallback (non-exact) matching only considers the qualities ranked above, in
# QUALITY_DEFS order; the qualities appended later take part in exact matches only.
_FALLBACK_QUALITIES: list[tuple[str, list[int]]] = [
    (token, intervals) for token, intervals in QUALITY_DEFS
    if token in _QUALITY_FALLBACK_PRIORITY
]


def _played_pitch_classes(fret_inputs: List[Optional[int]]) -> Tuple[Set[int], Optional[int]]:
    """Pitch classes sounded by fret_inputs (e..E, None = not played) and the bass pitch class."""
    played_pcs: Set[int] = set()
    bass_pc: Optional[int] = None
    for string_idx, fret in enumerate(fret_inputs):
        if fret is not None:
            pc = (OPEN_STRING_SEMITONES[string_idx] + fret) % 12
            played_pcs.add(pc)
            # Iterating 0→5 means the last assigned bass_pc is the lowest string played.
            bass_pc = pc
    return played_pcs, bass_pc


def _make_parsed(root_pc: int, token: str, intervals: List[int]) -> ParsedChord:
    """ParsedChord for a reverse-lookup result, named with the user-facing suffix."""
    root_name = NOTE_NAMES_SHARP[root_pc]
    return ParsedChord(
        raw=root_name + _QUALITY_DISPLAY.get(token, token),
        root=root_name,
        root_semitone=root_pc,
        quality=token,
        intervals=list(intervals),
    )


def _ranked_exact_matches(played_pcs: Set[int], bass_pc: Optional[int]) -> List[ParsedChord]:
    """Every chord (all QUALITY_DEFS) whose pitch classes equal played_pcs, best first."""
    exact_candidates = []
    for root_pc in range(12):
        for token, intervals in QUALITY_DEFS:
            chord_pcs = {(root_pc + iv) % 12 for iv in intervals}
            if played_pcs == chord_pcs:
                # Last element: on a full tie the original qualities keep winning,
                # so appended qualities only take over when strictly better.
                score = (
                    1 if bass_pc == root_pc else 0,
                    -len(intervals),
                    1 if token in _QUALITY_FALLBACK_PRIORITY else 0,
                )
                exact_candidates.append((score, _make_parsed(root_pc, token, intervals)))
    exact_candidates.sort(key=lambda x: x[0], reverse=True)   # stable: ties keep generation order
    return [parsed for _, parsed in exact_candidates]


def identify_exact_chords(fret_inputs: List[Optional[int]]) -> List[ParsedChord]:
    """List every exact-match chord name for the given frets, best first.

    Uses the same scoring as identify_chord_from_frets, whose exact result is
    always the first element. Fallback (subset) matches are not included.

    Args:
        fret_inputs: 6 elements ordered e, B, G, D, A, E (high → low string).
                     None = string not played. 0 = open string. n>0 = fret n.

    Returns:
        Ranked ParsedChords; empty when fewer than 2 distinct pitch classes
        are played or no chord matches exactly.
    """
    played_pcs, bass_pc = _played_pitch_classes(fret_inputs)
    if len(played_pcs) < 2:
        return []
    return _ranked_exact_matches(played_pcs, bass_pc)


def identify_chord_from_frets(
    fret_inputs: List[Optional[int]],
) -> Optional[Tuple[ParsedChord, bool]]:
    """Identify the chord whose pitch-class set best matches the given frets.

    Strategy:
      1. Exact match  – played pitch classes == chord pitch classes (all QUALITY_DEFS).
      2. Fallback     – played pitch classes ⊂ chord pitch classes (_FALLBACK_QUALITIES only)
         (used only when no exact match exists).
    Tie-breaking: root == bass note first, then simplest chord (fewest tones);
    exact ties go to the original (non-appended) qualities.

    Args:
        fret_inputs: 6 elements ordered e, B, G, D, A, E (high → low string).
                     None = string not entered. 0 = open string. n>0 = fret n.

    Returns:
        (ParsedChord, is_exact) where is_exact is True for an exact match,
        or None if fewer than 2 distinct pitch classes are provided.
    """
    played_pcs, bass_pc = _played_pitch_classes(fret_inputs)
    if len(played_pcs) < 2:
        return None

    # Phase 1: exact match (played_pcs == chord_pcs)
    exact = _ranked_exact_matches(played_pcs, bass_pc)
    if exact:
        return exact[0], True

    # Phase 2: fallback – played_pcs ⊆ chord_pcs
    fallback_candidates = []
    for root_pc in range(12):
        for token, intervals in _FALLBACK_QUALITIES:
            chord_pcs = {(root_pc + iv) % 12 for iv in intervals}
            if played_pcs.issubset(chord_pcs):
                score = (
                    1 if bass_pc == root_pc else 0,
                    len(played_pcs) / len(chord_pcs),       # higher = closer match
                    -len(intervals),
                    -_QUALITY_FALLBACK_PRIORITY.get(token, 10),  # maj/min before aug/dim
                )
                fallback_candidates.append((score, _make_parsed(root_pc, token, intervals)))

    if not fallback_candidates:
        return None

    fallback_candidates.sort(key=lambda x: x[0], reverse=True)
    return fallback_candidates[0][1], False


if __name__ == "__main__":
    from chord_parser import parse_chord

    # Self-check: every curated shape must sound only chord tones and include the root.
    _quality_intervals = dict(QUALITY_DEFS)
    for (_root, _quality, _bass), _shapes in VOICING_DB.items():
        _tones = {(NOTE_TO_SEMITONE[_root] + iv) % 12 for iv in _quality_intervals[_quality]}
        for _frets in _shapes:
            _pcs = {(OPEN_STRING_SEMITONES[i] + f) % 12 for i, f in enumerate(_frets) if f >= 0}
            assert _pcs <= _tones and NOTE_TO_SEMITONE[_root] in _pcs, (
                f"VOICING_DB{(_root, _quality, _bass)} shape {_frets} sounds "
                f"{sorted(semitone_to_name(p) for p in _pcs)}"
            )
    print(f"VOICING_DB self-check OK ({sum(len(v) for v in VOICING_DB.values())} shapes)")
    for s in ["C", "Cm", "G", "Am", "F", "CM7", "C/G"]:
        p = parse_chord(s)
        vs = get_voicings(p, source="both", max_results=6)
        print(f"\n=== {s} ({len(vs)} voicings) ===")
        for i, v in enumerate(vs):
            print(f"  [{i+1}] src={v.source:4s} frets(eBGDAE)={v.frets}  base_fret={v.base_fret}")
