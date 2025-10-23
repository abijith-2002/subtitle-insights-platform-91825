# Inaccuracy Injector for SRT Files

This utility introduces small, human-like inaccuracies into `.srt` subtitle files and generates detailed change reports. It is designed to be idempotent, safe to re-run, and requires only the Python standard library.

## What It Does

For each `.srt` found in this `SUBTITLE` directory, the tool:
- Backs up the original SRT to `SUBTITLE/originals/<filename>.srt` (first run only).
- Introduces 1–3 small inaccuracies per file:
  - Remove a non-trivial word from a random subtitle line.
  - Replace a random word with a similar-looking word (small typo or homoglyph).
  - Shift a subtitle block’s start/end time by -150ms to +150ms (safely clamped, duration preserved).
- Writes the modified SRT back while preserving valid SRT structure (index, timing, text, blank line).
- Creates a text report next to each `.srt` (e.g., `file_inaccuracies.txt`) detailing each change:
  - Change type (WORD_DELETE, WORD_REPLACE, TIME_SHIFT)
  - Block index, line number (for text changes)
  - Before/after values (text or timestamps)
  - Applied offset in milliseconds (for time shifts)

Idempotence and safety:
- If the very first line of a file is `X-Kavia-Inaccuracies: applied`, the file is skipped to avoid compounding changes.
- Backups are created before modifications (first time only).
- Time shifts are clamped to avoid negative times and preserve `start <= end`.

## Usage

From the project root or this folder, run:

```
python3 SUBTITLE/tools/inaccuracy_injector.py
```

Optional: make changes reproducible by specifying a seed:

```
python3 SUBTITLE/tools/inaccuracy_injector.py --seed 42
```

The script will:
- Process every `.srt` in `SUBTITLE/`
- Write reports like `MySubtitles_inaccuracies.txt` next to the `.srt`
- Skip any `.srt` already marked as processed

## File Layout

- Originals backup: `SUBTITLE/originals/<original-filename>.srt`
- Modified file: `SUBTITLE/<original-filename>.srt` (prepended with the marker line)
- Change report: `SUBTITLE/<original-stem>_inaccuracies.txt`

## Re-running and Restoration

- Re-running the tool will skip already processed files (marker present).
- To re-apply with a fresh set of changes, restore the original from `SUBTITLE/originals/` and remove the modified file, or remove the marker and the report file manually.

## Notes

- The script prepends a marker line `X-Kavia-Inaccuracies: applied` to the top of modified `.srt` files. This is used to detect already-processed files and avoid compounding inaccuracies.
- Timestamp shifts are small and kept within safe bounds to preserve SRT validity.
- No external dependencies are required (Python standard library only).
