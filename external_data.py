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
import logging
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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


Barre = Tuple[int, int, int]  # (absolute fret, first string idx, last string idx), high-e-first indices
_MISSING = object()


def _validate_fingers(fingers: object, cache: dict) -> Optional[Tuple[int, ...]]:
    """Validate a high-e-first finger list.

    Args:
        fingers: Candidate finger list (ints or digit strings), already in high-e-first order.
        cache: Per-load intern table so identical finger tuples share one object.

    Returns:
        Six ints (0 = no finger, 1-4 = finger), or None if malformed
        (wrong length, non-integer, or a value outside 0-4).
    """
    if not isinstance(fingers, list) or len(fingers) != 6:
        return None
    raw = tuple(fingers)
    try:
        hit = cache.get(raw, _MISSING)
    except TypeError:  # unhashable element (e.g. a nested list)
        return None
    if hit is not _MISSING:
        return hit
    # Only real ints or digit strings; int() would silently accept 1.9 or True.
    if all((isinstance(f, int) and not isinstance(f, bool)) or (isinstance(f, str) and f.strip().isdecimal())
           for f in raw):
        out: Optional[Tuple[int, ...]] = tuple(int(f) for f in raw)
    else:
        out = None
    if out is not None:
        out = cache.setdefault(out, out) if all(0 <= n <= 4 for n in out) else None
    cache[raw] = out  # only a few hundred distinct patterns, so most calls are one dict hit
    return out


def _fingers_match_frets(fingers: Tuple[int, ...], frets: List[int]) -> bool:
    """Return True if a finger tuple is physically consistent with its frets.

    Rejected (dataset misalignment, e.g. T-vK 'Bm' x24432 listed as 0 1 4 4 1 1):
    one finger (1-4) on two different frets, a finger on an open/muted string,
    a pressed string with no finger, or one finger on same-fret strings that an
    open or lower-fretted string in between keeps it from covering in one bar.

    Args:
        fingers: Validated high-e-first finger tuple.
        frets: The voicing's six absolute fret numbers (high-e first).

    Returns:
        True when the fingering can be shown as-is.
    """
    fret_of_finger: dict[int, int] = {}
    strings_of_finger: dict[int, List[int]] = {}
    for idx, (finger, fret) in enumerate(zip(fingers, frets)):
        if (finger > 0) != (fret > 0):
            return False
        if finger > 0:
            if fret_of_finger.setdefault(finger, fret) != fret:
                return False
            strings_of_finger.setdefault(finger, []).append(idx)
    for finger, idxs in strings_of_finger.items():
        fret = fret_of_finger[finger]
        if len(idxs) >= 2 and _barre_runs(fret, idxs, frets) != [(fret, idxs[0], idxs[-1])]:
            return False
    return True


def _barre_runs(fret: int, strings: List[int], frets: List[int]) -> List[Barre]:
    """Playable barre spans over the given strings (all pressed at `fret`).

    A bar cannot cross a string that is open or fretted below the barre fret, so the
    strings are split into runs at such strings (muted strings may lie under a bar);
    only runs covering two or more of the given strings are kept.

    Args:
        fret: Absolute barre fret.
        strings: High-e-first indices of the strings pressed at `fret` by the barre finger.
        frets: The voicing's six absolute fret numbers (high-e first).

    Returns:
        (fret, first idx, last idx) spans, in string order.
    """
    runs: List[List[int]] = []
    for idx in sorted(strings):
        if runs and not any(0 <= frets[i] < fret for i in range(runs[-1][-1] + 1, idx)):
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return [(fret, run[0], run[-1]) for run in runs if len(run) >= 2]


def _resolve_barres(
    barres_rel: object,
    base_fret: int,
    frets: List[int],
    fingers: Optional[Tuple[int, ...]],
    cache: dict,
) -> Optional[Tuple[Barre, ...]]:
    """Convert chords-db barres (relative to baseFret) into absolute, playable barre spans.

    Strings pressed at the barre fret are grouped by finger when finger data is
    valid (so another finger at that fret is not swallowed by the bar), then split
    by _barre_runs. A listed barre that leaves no playable run contributes nothing.

    Args:
        barres_rel: Raw "barres" value from a chords-db position.
        base_fret: The position's baseFret.
        frets: The voicing's six absolute fret numbers (high-e first).
        fingers: Validated high-e-first fingers, or None.
        cache: Per-load intern table.

    Returns:
        Barre spans (possibly empty), or None if malformed: not a list of
        positive ints, or a barre fret pressed on fewer than two strings.
    """
    if not isinstance(barres_rel, list):
        return None
    out: List[Barre] = []
    for b in barres_rel:
        if not isinstance(b, int) or isinstance(b, bool) or b <= 0:
            return None
        absolute = b if base_fret <= 1 else base_fret + b - 1
        strings = [i for i, f in enumerate(frets) if f == absolute]
        if len(strings) < 2:
            return None
        if fingers is None:
            out.extend(_barre_runs(absolute, strings, frets))
            continue
        by_finger: dict[int, List[int]] = {}
        for i in strings:
            if fingers[i] > 0:
                by_finger.setdefault(fingers[i], []).append(i)
        for idxs in by_finger.values():
            out.extend(_barre_runs(absolute, idxs, frets))
    result = tuple(sorted(out))
    return cache.setdefault(result, result)


def _derive_barres(fingers: Tuple[int, ...], frets: List[int], cache: dict) -> Tuple[Barre, ...]:
    """Barres implied by T-vK fingerings: one finger (1-4) pressing 2+ strings at the same fret.

    Precondition: _fingers_match_frets(fingers, frets) is True, so each finger sits on one fret.

    Args:
        fingers: Validated high-e-first finger tuple.
        frets: The voicing's six absolute fret numbers (high-e first).
        cache: Per-load intern table.

    Returns:
        Barre spans sorted by fret (empty when no finger covers two strings).
    """
    used = [f for f in fingers if f]
    if len(used) == len(set(used)):  # fast path: no finger presses two strings
        return ()
    strings_of: dict[int, List[int]] = {}
    for idx, finger in enumerate(fingers):
        if finger:
            strings_of.setdefault(finger, []).append(idx)
    result = tuple(sorted(
        span for idxs in strings_of.values() if len(idxs) >= 2
        for span in _barre_runs(frets[idxs[0]], idxs, frets)
    ))
    return cache.setdefault(result, result)


def load_chords_db(path: str) -> List[dict]:
    """Load voicings from a chords-db lib/<instrument>.json file.

    Args:
        path: Filesystem path to the chords-db JSON file.

    Returns:
        List of dicts with keys {"root", "quality", "bass", "frets", "fingers",
        "barres", "source"}. Frets are the high-e-first 6-element list of absolute
        fret numbers; fingers (high-e first, 0-4; the dataset's -1 is read as 0) and
        barres ((fret, first string, last string) tuples) are None when absent or
        malformed (fingers also when inconsistent with the frets, see
        _fingers_match_frets). Voicings that can't be normalized are skipped, and bad
        finger data is dropped; both are counted in a warning log.
    """
    if not os.path.exists(path):
        logger.warning("External data file not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        db = json.load(f)

    out: List[dict] = []
    skipped = 0
    bad_fingering = 0
    cache: dict = {}
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
                    skipped += 1
                    continue
                abs_frets = _resolve_absolute_frets(list(frets_rel), int(base_fret))
                if abs_frets is None:
                    skipped += 1
                    continue
                # Reverse: chords-db is low->high; we want high->low.
                abs_frets_hi_to_lo = list(reversed(abs_frets))
                fingers = barres = None
                if "fingers" in pos:
                    raw_fingers = pos["fingers"]
                    if isinstance(raw_fingers, list):
                        # Two entries mark an unfingered (muted) string with -1; same as 0.
                        raw_fingers = [0 if f == -1 else f for f in reversed(raw_fingers)]
                    fingers = _validate_fingers(raw_fingers, cache)
                    if fingers is not None and not _fingers_match_frets(fingers, abs_frets_hi_to_lo):
                        fingers = None
                    bad_fingering += fingers is None
                if "barres" in pos:
                    barres = _resolve_barres(
                        pos["barres"], int(base_fret), abs_frets_hi_to_lo, fingers, cache
                    )
                    bad_fingering += barres is None
                out.append({
                    "root": root,
                    "quality": quality,
                    "bass": bass,
                    "frets": abs_frets_hi_to_lo,
                    "fingers": fingers,
                    "barres": barres,
                    "source": "chords-db",
                })
    if skipped:
        logger.warning("Skipped %d malformed voicings in %s", skipped, path)
    if bad_fingering:
        logger.warning("Dropped %d malformed or inconsistent finger/barre entries in %s", bad_fingering, path)
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
        List of dicts with keys {"root", "quality", "bass", "frets", "fingers",
        "barres", "source"}. Fingers come from the first "fingerings" entry
        (high-e first, 0-4) or None if absent, malformed or inconsistent with the
        frets (see _fingers_match_frets). T-vK has no barre field,
        so barres are derived from the fingers (see _derive_barres); None without fingers.
    """
    if not os.path.exists(path):
        logger.warning("External data file not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[dict] = []
    skipped = 0
    bad_fingering = 0
    cache: dict = {}
    for name, voicings in data.items():
        parsed = _parse_tvk_name(name)
        if parsed is None:
            continue
        root, quality, bass = parsed
        for v in voicings:
            positions = v.get("positions")
            if not isinstance(positions, list):
                skipped += 1
                continue
            frets = _tvk_position_to_frets(positions)
            if frets is None:
                skipped += 1
                continue
            fingers = barres = None
            fingerings = v.get("fingerings")
            if isinstance(fingerings, list) and fingerings:
                raw_fingers = fingerings[0]
                if isinstance(raw_fingers, list):
                    raw_fingers = list(reversed(raw_fingers))  # low->high  =>  high->low
                fingers = _validate_fingers(raw_fingers, cache)
                if fingers is not None and not _fingers_match_frets(fingers, frets):
                    fingers = None
                bad_fingering += fingers is None
                if fingers is not None:
                    barres = _derive_barres(fingers, frets, cache)
            out.append({
                "root": root,
                "quality": quality,
                "bass": bass,
                "frets": frets,
                "fingers": fingers,
                "barres": barres,
                "source": "T-vK",
            })
    if skipped:
        logger.warning("Skipped %d malformed voicings in %s", skipped, path)
    if bad_fingering:
        logger.warning("Dropped %d malformed or inconsistent finger entries in %s", bad_fingering, path)
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

    Missing files are skipped (logged as a warning). A dataset that fails to
    load is logged and skipped without discarding the other one.

    Args:
        data_dir: Directory containing the external JSON files.
                  Defaults to a 'data/' subdirectory next to this module.

    Returns:
        List of imported voicing dicts ({"root","quality","bass","frets","fingers","barres","source"}).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    voicings: List[dict] = []
    for loader, key in ((load_chords_db, "chords-db"), (load_tvk, "T-vK")):
        path = os.path.join(data_dir, DEFAULT_FILES[key])
        try:
            voicings.extend(loader(path))
        except Exception:
            logger.exception("Failed to load %s data from %s; skipping it.", key, path)
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

    # Barre self-check (high-e-first): a bar must not cross a string fretted below it.
    c69 = [10, 10, 9, 10, 10, -1]  # chords-db C69: only finger 2 on D+A is a barre
    assert _resolve_barres([10], 1, c69, (4, 3, 1, 2, 2, 0), {}) == ((10, 3, 4),)
    assert _resolve_barres([10], 1, c69, None, {}) == ((10, 0, 1), (10, 3, 4))
    # T-vK C#m/F#: finger 1 at fret 2 spans B..E over a muted G; finger 4 at fret 4 is split by B/D at fret 2
    assert _derive_barres((4, 1, 1, 1, 4, 1), [4, 2, -1, 2, 4, 2], {}) == ((2, 1, 5),)
    assert _resolve_barres([1], 1, [1, 1, 2, 3, 3, 1], (1, 1, 2, 4, 3, 1), {}) == ((1, 0, 5),)  # F barre
    assert _validate_fingers([1.9, 0, 0, 0, 0, 0], {}) is None and _validate_fingers([True, 0, 0, 0, 0, 0], {}) is None
    assert _validate_fingers(["²", "0", "0", "0", "0", "0"], {}) is None  # isdigit-but-not-int must not raise
    # Impossible fingerings (T-vK data for the built-in Bm / B shapes) are rejected; real ones pass.
    assert not _fingers_match_frets((1, 1, 4, 4, 1, 0), [2, 3, 4, 4, 2, -1])   # finger 1 on frets 2 and 3
    assert not _fingers_match_frets((1, 1, 3, 4, 1, 0), [2, 4, 4, 4, 2, -1])   # finger 1 on frets 2 and 4
    assert not _fingers_match_frets((4, 1, 0, 0, 3, 2), [9, 7, 6, 6, -1, 9])   # fingers shifted onto muted/pressed strings
    assert not _fingers_match_frets((4, 3, 1, 1, 3, 0), [3, 3, 2, 2, 3, -1])   # chords-db C69: finger 3 on B+A over G/D at 2
    assert _fingers_match_frets((1, 2, 4, 3, 1, 0), [2, 3, 4, 4, 2, -1])       # Bm A-shape
    assert _fingers_match_frets((1, 1, 2, 4, 3, 1), [1, 1, 2, 3, 3, 1])        # F barre
    print("barre self-check OK")
