"""
External data source loaders / adapters.

Both `chords-db` (tombatossals) and `chord-collection` (T-vK) store frets in
LOW-to-HIGH string order (E A D G B e). Our internal model stores them in
HIGH-to-LOW order (e B G D A E), so every imported voicing is reversed.

Returned voicings use these internal conventions:
    frets: List[int]  length 6, index 0 = high e
    -1 = muted, 0 = open, n>0 = absolute fret number
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple


# Maps used by both adapters
NOTE_NAME_NORMALIZE = {
    "Csharp": "C#", "Dsharp": "D#", "Fsharp": "F#", "Gsharp": "G#", "Asharp": "A#",
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",  # flats -> sharps (canonical)
}

SHARP_TO_FLAT = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
}


def _norm_root(s: str) -> str:
    """Normalize a root note string to the canonical sharp form (C, C#, D, ..., B).

    Args:
        s: Raw root string from an external dataset (e.g. "Csharp", "Db").

    Returns:
        Canonical sharp-form root string (e.g. "C#", "D").
    """
    s = s.strip()
    return NOTE_NAME_NORMALIZE.get(s, s)


# ---------------------------------------------------------------------------
# chords-db (tombatossals) loader
# ---------------------------------------------------------------------------
#
# Suffix mapping: chords-db uses "major"/"minor"/"maj7"/"m7"/...; our internal
# parser canonicalizes to "maj"/"min"/"maj7"/"min7"/... Build a map for the
# overlap; unknown suffixes are skipped.
#
# Map chords-db suffix body (after slash-split) -> our canonical quality token.
_CHORDSDB_QUALITY_MAP: dict[str, str] = {
    "":           "maj",
    "major":      "maj",
    "minor":      "min",
    "m":          "min",
    "maj":        "maj",
    "5":          "5",
    "6":          "6",
    "7":          "7",
    "9":          "9",
    "11":         "11",
    "13":         "13",
    "dim":        "dim",
    "dim7":       "dim7",
    "aug":        "aug",
    "aug7":       "aug7",
    "sus":        "sus4",
    "sus2":       "sus2",
    "sus4":       "sus4",
    "maj7":       "maj7",
    "maj9":       "maj9",
    "m6":         "min6",
    "m7":         "min7",
    "m9":         "min9",
    "m11":        "min11",
    "add9":       "add9",
    "madd9":      "minadd9",
    "add11":      "add11",
    # Extended/altered: accepted as opaque tokens matching our quality vocabulary.
    "7sus4":      "7sus4",
    "7b5":        "7b5",
    "9b5":        "9b5",
    "aug9":       "aug9",
    "7b9":        "7b9",
    "7#9":        "7#9",
    "9#11":       "9#11",
    "69":         "69",
    "m7b5":       "min7b5",
    "mmaj7":      "minmaj7",
    "maj7b5":     "maj7b5",
    "maj7#5":     "maj7#5",
    "maj7sus2":   "maj7sus2",
    "maj11":      "maj11",
    "maj13":      "maj13",
    "m69":        "min69",
    "mmaj7b5":    "minmaj7b5",
    "mmaj9":      "minmaj9",
    "mmaj11":     "minmaj11",
    "sus2sus4":   "sus2sus4",
    "alt":        "alt",
}


def _chordsdb_suffix_map(suffix: str) -> Optional[Tuple[str, Optional[str]]]:
    """Convert a chords-db suffix string to (quality, bass_root_or_None).

    Args:
        suffix: Raw suffix string from chords-db (e.g. "major", "m7", "m/B").

    Returns:
        Tuple of (canonical_quality, bass_note_or_None), or None if unsupported.
    """
    s = suffix.strip()

    bass: Optional[str] = None
    if "/" in s:
        body, _, bass_raw = s.partition("/")
        bass = _norm_root(bass_raw)
        s = body

    if s not in _CHORDSDB_QUALITY_MAP:
        return None
    return _CHORDSDB_QUALITY_MAP[s], bass


def _resolve_absolute_frets(frets_rel: List[int], base_fret: int) -> Optional[List[int]]:
    """Convert chords-db frets (relative to baseFret) into absolute fret numbers.

    Convention (verified against canonical voicings such as C major / open
    C-shape at baseFret=1 and F major / E-shape barre at baseFret=1):
      baseFret == 1: frets are already absolute. -1 mute, 0 open, 1.. pressed.
      baseFret >  1: a "1" means the baseFret itself; absolute = baseFret + n - 1
                     for n>0. n=0 still means open string (rare with high
                     baseFret, but we honor it). -1 still means mute.

    Args:
        frets_rel: Six-element list of relative fret values from chords-db.
        base_fret: The baseFret value from the chords-db position entry.

    Returns:
        Six-element list of absolute fret numbers, or None on malformed input.
    """
    if len(frets_rel) != 6:
        return None
    out: List[int] = []
    for n in frets_rel:
        if n == -1:
            out.append(-1)
        elif n == 0:
            out.append(0)
        elif n > 0:
            if base_fret <= 1:
                out.append(n)
            else:
                out.append(base_fret + n - 1)
        else:
            return None
    return out


def load_chords_db(path: str) -> List[dict]:
    """Load voicings from a chords-db lib/<instrument>.json file.

    Args:
        path: Filesystem path to the chords-db JSON file.

    Returns:
        List of dicts with keys {"root", "quality", "bass", "frets", "source"}.
        Frets are the high-e-first 6-element list of absolute fret numbers.
        Voicings that can't be normalized are skipped silently.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        db = json.load(f)

    out: List[dict] = []
    chords = db.get("chords", {})
    for root_raw, entries in chords.items():
        root = _norm_root(root_raw)  # "Csharp" -> "C#"
        for entry in entries:
            mapped = _chordsdb_suffix_map(entry.get("suffix", ""))
            if mapped is None:
                continue
            quality, bass = mapped
            for pos in entry.get("positions", []):
                frets_rel = pos.get("frets")
                base_fret = pos.get("baseFret", 1)
                if not isinstance(frets_rel, list):
                    continue
                abs_frets = _resolve_absolute_frets(list(frets_rel), int(base_fret))
                if abs_frets is None:
                    continue
                # Reverse: chords-db is low->high; we want high->low.
                abs_frets_hi_to_lo = list(reversed(abs_frets))
                out.append({
                    "root": root,
                    "quality": quality,
                    "bass": bass,
                    "frets": abs_frets_hi_to_lo,
                    "source": "chords-db",
                })
    return out


# ---------------------------------------------------------------------------
# T-vK chord-collection loader
# ---------------------------------------------------------------------------
#
# Key naming uses sharps only ("A#" not "Bb"). The chord name follows the
# pattern: <root>[#] [<quality>] [/<bass>[#]] where root is one of A-G.
#
# positions: list of 6 strings, low->high; "x" = mute, "0".."N" = fret number.
# Fret numbers in T-vK are ABSOLUTE (not relative to a baseFret).

# Compile once.
_TVK_NAME_RE = re.compile(
    r"^([A-G])(#)?(.*?)(?:/([A-G])(#)?)?$"
)

# Map T-vK quality token -> our canonical quality.
_TVK_QUALITY_MAP: dict[str, str] = {
    "":          "maj",
    "M":         "maj",
    "maj":       "maj",
    "m":         "min",
    "min":       "min",
    "5":         "5",
    "6":         "6",
    "7":         "7",
    "9":         "9",
    "11":        "11",
    "13":        "13",
    "dim":       "dim",
    "dim7":      "dim7",
    "aug":       "aug",
    "aug7":      "aug7",
    "sus":       "sus4",
    "sus2":      "sus2",
    "sus4":      "sus4",
    "maj7":      "maj7",
    "maj9":      "maj9",
    "m6":        "min6",
    "m7":        "min7",
    "m9":        "min9",
    "m11":       "min11",
    "add9":      "add9",
    "madd9":     "minadd9",
    # Altered / extended (kept as opaque tokens matching chords-db vocabulary).
    "7b5":       "7b5",
    "7#9":       "7#9",
    "7b9":       "7b9",
    "9b5":       "9b5",
    "9#11":      "9#11",
    "aug9":      "aug9",
    "m7b5":      "min7b5",
    "mmaj7":     "minmaj7",
    "maj7b5":    "maj7b5",
    "maj7#5":    "maj7#5",
    "6/9":       "69",
    "m6/9":      "min69",
    "7sus4":     "7sus4",
    "7sus2":     "7sus2",
    "11sus4":    "11sus4",
    "13sus4":    "13sus4",
    "sus2sus4":  "sus2sus4",
}


def _tvk_quality_map(token: str) -> Optional[str]:
    """Convert a T-vK quality token to our canonical quality string.

    Args:
        token: Raw quality token from a T-vK chord key (e.g. "m7", "maj9").

    Returns:
        Canonical quality string, or None if the token is not supported.
    """
    return _TVK_QUALITY_MAP.get(token.strip())


def _parse_tvk_name(name: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse a T-vK chord name into (root, quality, bass).

    Examples: 'C' -> ('C','maj',None), 'C#m7' -> ('C#','min7',None),
              'Cm/F#' -> ('C','min','F#').

    Args:
        name: Chord name string from the T-vK dataset.

    Returns:
        Tuple (root, canonical_quality, bass_or_None), or None if unsupported.
    """
    m = _TVK_NAME_RE.match(name.strip())
    if not m:
        return None
    root_letter, root_sharp, body, bass_letter, bass_sharp = m.groups()
    root = root_letter + (root_sharp or "")
    bass = (bass_letter + (bass_sharp or "")) if bass_letter else None

    quality = _tvk_quality_map(body or "")
    if quality is None:
        return None
    return root, quality, bass


def _tvk_position_to_frets(positions: List[str]) -> Optional[List[int]]:
    """Convert T-vK position strings (low->high) to high->low absolute fret ints.

    Args:
        positions: Six-element list of position strings (e.g. ["x","3","2","0","1","0"]).

    Returns:
        Six-element list of integers (high-e first), or None on malformed input.
    """
    if len(positions) != 6:
        return None
    nums: List[int] = []
    for p in positions:
        ps = str(p).strip().lower()
        if ps in ("x", "-1"):
            nums.append(-1)
        else:
            try:
                v = int(ps)
            except ValueError:
                return None
            if v < 0:
                return None
            nums.append(v)
    return list(reversed(nums))  # low->high  =>  high->low


def load_tvk(path: str) -> List[dict]:
    """Load voicings from T-vK chord-collection's chords.complete.json.

    Args:
        path: Filesystem path to the T-vK JSON file.

    Returns:
        List of dicts with keys {"root", "quality", "bass", "frets", "source"}.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[dict] = []
    for name, voicings in data.items():
        parsed = _parse_tvk_name(name)
        if parsed is None:
            continue
        root, quality, bass = parsed
        for v in voicings:
            positions = v.get("positions")
            if not isinstance(positions, list):
                continue
            frets = _tvk_position_to_frets(positions)
            if frets is None:
                continue
            out.append({
                "root": root,
                "quality": quality,
                "bass": bass,
                "frets": frets,
                "source": "T-vK",
            })
    return out


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_FILES = {
    "chords-db": "chords_db_guitar.json",
    "T-vK":      "tvk_chords_complete.json",
}


def load_all_external(data_dir: Optional[str] = None) -> List[dict]:
    """Load every recognized external dataset present in data_dir.

    Missing files are skipped silently.

    Args:
        data_dir: Directory containing the external JSON files.
                  Defaults to a 'data/' subdirectory next to this module.

    Returns:
        List of imported voicing dicts ({"root","quality","bass","frets","source"}).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    voicings: List[dict] = []
    voicings.extend(load_chords_db(os.path.join(data_dir, DEFAULT_FILES["chords-db"])))
    voicings.extend(load_tvk(os.path.join(data_dir, DEFAULT_FILES["T-vK"])))
    return voicings


if __name__ == "__main__":
    # Quick statistics
    from collections import Counter

    print("Loading chords-db ...")
    db = load_chords_db(os.path.join(DEFAULT_DATA_DIR, "chords_db_guitar.json"))
    print(f"  -> {len(db)} voicings")
    print("  qualities:", Counter(v["quality"] for v in db).most_common(10))

    print("\nLoading T-vK ...")
    tvk = load_tvk(os.path.join(DEFAULT_DATA_DIR, "tvk_chords_complete.json"))
    print(f"  -> {len(tvk)} voicings")
    print("  qualities:", Counter(v["quality"] for v in tvk).most_common(10))

    # Quick sanity check on some chords
    for target in [("C", "maj", None), ("C", "min", None), ("G", "maj", None),
                   ("A", "min", None), ("F", "maj", None), ("C", "maj", "G")]:
        n_db = sum(1 for v in db if (v["root"], v["quality"], v["bass"]) == target)
        n_tvk = sum(1 for v in tvk if (v["root"], v["quality"], v["bass"]) == target)
        print(f"  {target!s:30s}  chords-db={n_db:3d}  T-vK={n_tvk:3d}")
