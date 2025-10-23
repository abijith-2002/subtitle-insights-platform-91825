#!/usr/bin/env python3
"""
Inaccuracy Injector for SRT files.

This script scans the SUBTITLE directory (parent of this script's folder) for .srt files,
backs up originals, introduces small randomized inaccuracies, and writes a per-file report.

Changes introduced (1-3 per file, randomly chosen):
- remove a random non-trivial word from a random subtitle line
- replace a random word with a similar-looking word (typo or homoglyph)
- shift subtitle block start/end by a small offset (-150ms..+150ms), clamped safely

Safety and idempotence:
- If a file already contains the top-line marker "X-Kavia-Inaccuracies: applied", the file is skipped.
- Before any modifications, the original file is copied to SUBTITLE/originals/<filename>.
- The modified file is prepended with the marker and remains SRT formatted (index, times, text, blank line).

Usage:
  python3 tools/inaccuracy_injector.py [--seed N]

No external dependencies beyond Python standard library.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

MARKER_LINE = "X-Kavia-Inaccuracies: applied"
REPORT_SUFFIX = "_inaccuracies.txt"
BACKUP_DIRNAME = "originals"

TIME_PATTERN = re.compile(
    r"^(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})$"
)

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SrtBlock:
    """A parsed SRT block."""
    index: int
    start_ms: int
    end_ms: int
    text_lines: List[str]

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def _time_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def _ms_to_time(ms: int) -> Tuple[int, int, int, int]:
    if ms < 0:
        ms = 0
    s, msec = divmod(ms, 1000)
    m, sec = divmod(s, 60)
    h, min_ = divmod(m, 60)
    return h, min_, sec, msec


def _format_ts(ms: int) -> str:
    h, m, s, ms_ = _ms_to_time(ms)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


def _format_times(start_ms: int, end_ms: int) -> str:
    return f"{_format_ts(start_ms)} --> {_format_ts(end_ms)}"


def _parse_times(line: str) -> Optional[Tuple[int, int]]:
    m = TIME_PATTERN.match(line.strip())
    if not m:
        return None
    sh, sm, ss, sms = int(m["sh"]), int(m["sm"]), int(m["ss"]), int(m["sms"])
    eh, em, es, ems = int(m["eh"]), int(m["em"]), int(m["es"]), int(m["ems"])
    start = _time_to_ms(sh, sm, ss, sms)
    end = _time_to_ms(eh, em, es, ems)
    return start, end


def _split_blocks(raw: str) -> List[str]:
    # Normalize newlines and split by double newlines
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Ensure trailing newline so last block is captured
    if not raw.endswith("\n"):
        raw += "\n"
    return re.split(r"\n{2,}", raw.strip(), flags=re.MULTILINE)


def _join_blocks(blocks: List[SrtBlock]) -> str:
    parts: List[str] = []
    for b in blocks:
        parts.append(str(b.index))
        parts.append(_format_times(b.start_ms, b.end_ms))
        # Text lines as-is (may include punctuation or be empty)
        parts.extend(b.text_lines if b.text_lines else [""])
        parts.append("")  # blank line between blocks
    return "\n".join(parts).rstrip() + "\n"


def _parse_srt(content: str) -> List[SrtBlock]:
    blocks: List[SrtBlock] = []
    for chunk in _split_blocks(content):
        lines = [ln for ln in chunk.split("\n") if ln is not None]
        if len(lines) < 2:
            continue
        # Try to parse index, times, and text
        try:
            index = int(lines[0].strip())
        except ValueError:
            # Not a valid block (could be header or stray) — skip
            continue
        times = _parse_times(lines[1])
        if not times:
            continue
        text_lines = lines[2:] if len(lines) > 2 else []
        blocks.append(SrtBlock(index=index, start_ms=times[0], end_ms=times[1], text_lines=text_lines))
    return blocks


def _find_candidate_words(line: str, min_len: int = 4) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, word) positions for candidate words."""
    candidates: List[Tuple[int, int, str]] = []
    for m in WORD_RE.finditer(line):
        wd = m.group(0)
        # Allow alphanum, but focus on non-trivial
        if len(wd) >= min_len:
            candidates.append((m.start(), m.end(), wd))
    return candidates


def _remove_word_in_line(line: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    candidates = _find_candidate_words(line)
    if not candidates:
        return None
    start, end, wd = rng.choice(candidates)
    new_line = (line[:start] + line[end:]).replace("  ", " ").strip()
    return wd, new_line


def _perturb_word(word: str, rng: random.Random) -> str:
    """Create a similar-looking word via a single small mutation."""
    if len(word) < 4:
        return word

    # 20%: homoglyph-ish substitution
    if rng.random() < 0.20:
        subs = {
            "o": "0", "O": "0",
            "l": "1", "I": "1",
            "e": "3", "E": "3",
            "s": "5", "S": "5",
        }
        chars = list(word)
        idxs = [i for i, ch in enumerate(chars) if ch in subs]
        if idxs:
            i = rng.choice(idxs)
            chars[i] = subs[chars[i]]
            return "".join(chars)

    # default: adjacent transposition in the middle of the word
    i = rng.randint(1, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _replace_word_in_line(line: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    candidates = _find_candidate_words(line)
    if not candidates:
        return None
    start, end, wd = rng.choice(candidates)
    replacement = _perturb_word(wd, rng)
    if replacement == wd:
        return None
    new_line = line[:start] + replacement + line[end:]
    return wd, new_line


def _choose_random_block_with_text(blocks: List[SrtBlock], rng: random.Random) -> Optional[Tuple[SrtBlock, int]]:
    candidates: List[Tuple[SrtBlock, int]] = []
    for b in blocks:
        for li, ln in enumerate(b.text_lines):
            if WORD_RE.search(ln or ""):
                candidates.append((b, li))
    if not candidates:
        return None
    return rng.choice(candidates)


def _choose_random_block(blocks: List[SrtBlock], rng: random.Random) -> Optional[SrtBlock]:
    if not blocks:
        return None
    return rng.choice(blocks)


# PUBLIC_INTERFACE
def process_srt_file(srt_path: Path, rng: random.Random) -> Tuple[bool, List[str]]:
    """
    Process a single SRT file: back up original (if first time), inject 1-3 inaccuracies,
    write the updated SRT with a marker, and produce a list of report lines.

    Returns:
        (modified: bool, report_lines: List[str])
    """
    report_lines: List[str] = []
    subtitle_dir = srt_path.parent
    backup_dir = subtitle_dir / BACKUP_DIRNAME

    raw = srt_path.read_text(encoding="utf-8", errors="ignore")
    # Idempotence marker check
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    if first_line.startswith(MARKER_LINE):
        report_lines.append(f"Skipped: {srt_path.name} already marked as processed.")
        return False, report_lines

    # Backup original
    backup_dir.mkdir(exist_ok=True, parents=True)
    backup_path = backup_dir / srt_path.name
    if not backup_path.exists():
        shutil.copy2(srt_path, backup_path)

    # Parse SRT blocks
    blocks = _parse_srt(raw)
    if not blocks:
        report_lines.append(f"No valid SRT blocks found in {srt_path.name}, skipping changes.")
        return False, report_lines

    changes_made = 0
    planned_changes = rng.randint(1, 3)
    available_actions = ["remove_word", "replace_word", "shift_time"]
    rng.shuffle(available_actions)

    # Keep track to not repeat the exact same change type if not needed
    actions_queue = available_actions.copy()

    # Helper to log
    def log_line(s: str) -> None:
        report_lines.append(s)

    log_line(f"File: {srt_path.name}")
    log_line(f"Processed at: {datetime.utcnow().isoformat()}Z")
    log_line("")

    while changes_made < planned_changes and actions_queue:
        action = actions_queue.pop(0)

        if action == "remove_word":
            chosen = _choose_random_block_with_text(blocks, rng)
            if not chosen:
                continue
            block, li = chosen
            before_line = block.text_lines[li]
            result = _remove_word_in_line(before_line, rng)
            if not result:
                continue
            removed_word, after_line = result
            block.text_lines[li] = after_line
            log_line(f"- Change #{changes_made + 1}: WORD_DELETE")
            log_line(f"  Block index: {block.index}, line: {li + 1}")
            log_line(f"  Removed word: '{removed_word}'")
            log_line(f"  Before: {before_line}")
            log_line(f"  After:  {after_line}")
            log_line("")
            changes_made += 1

        elif action == "replace_word":
            chosen = _choose_random_block_with_text(blocks, rng)
            if not chosen:
                continue
            block, li = chosen
            before_line = block.text_lines[li]
            result = _replace_word_in_line(before_line, rng)
            if not result:
                continue
            original_word, after_line = result
            # To find the actual replacement, we can diff a simple way
            block.text_lines[li] = after_line
            log_line(f"- Change #{changes_made + 1}: WORD_REPLACE")
            log_line(f"  Block index: {block.index}, line: {li + 1}")
            log_line(f"  Before: {before_line}")
            log_line(f"  After:  {after_line}")
            log_line("")
            changes_made += 1

        elif action == "shift_time":
            block = _choose_random_block(blocks, rng)
            if not block:
                continue
            # random offset in -150..+150 ms, excluding 0 to ensure a real change
            offset = 0
            attempts = 0
            while offset == 0 and attempts < 10:
                offset = rng.randint(-150, 150)
                attempts += 1
            if offset == 0:
                offset = 100  # fallback

            old_start, old_end = block.start_ms, block.end_ms
            dur = block.duration_ms
            new_start = max(0, old_start + offset)
            new_end = new_start + dur  # preserve duration
            # Ensure non-decreasing end
            if new_end < new_start:
                new_end = new_start
            block.start_ms, block.end_ms = new_start, new_end

            log_line(f"- Change #{changes_made + 1}: TIME_SHIFT")
            log_line(f"  Block index: {block.index}")
            log_line(f"  Offset (ms): {offset}")
            log_line(f"  Before: {_format_times(old_start, old_end)}")
            log_line(f"  After:  {_format_times(new_start, new_end)}")
            log_line("")
            changes_made += 1

    # Ensure at least one change: if none made, force a time shift on the first block
    if changes_made == 0:
        block = blocks[0]
        old_start, old_end = block.start_ms, block.end_ms
        dur = block.duration_ms
        offset = 100  # small deterministic nudge
        new_start = max(0, old_start + offset)
        new_end = new_start + dur
        block.start_ms, block.end_ms = new_start, new_end
        log_line(f"- Change #{changes_made + 1}: TIME_SHIFT (fallback)")
        log_line(f"  Block index: {block.index}")
        log_line(f"  Offset (ms): {offset}")
        log_line(f"  Before: {_format_times(old_start, old_end)}")
        log_line(f"  After:  {_format_times(new_start, new_end)}")
        log_line("")
        changes_made = 1

    # Recompose SRT content and prepend marker
    modified_body = _join_blocks(blocks)
    # Place marker on top with a blank line to keep readability
    stamped = f"{MARKER_LINE}\n{modified_body}"

    srt_path.write_text(stamped, encoding="utf-8")
    return True, report_lines


# PUBLIC_INTERFACE
def process_subtitle_directory(subtitle_dir: Path, seed: Optional[int] = None) -> None:
    """
    Scan the given subtitle directory for .srt files and process each file.

    Args:
        subtitle_dir: Path to the SUBTITLE directory that contains .srt files.
        seed: Optional seed for deterministic randomness across runs.
    """
    rng = random.Random(seed)
    srt_files = sorted([p for p in subtitle_dir.glob("*.srt") if p.is_file()])
    if not srt_files:
        print(f"No .srt files found in {subtitle_dir}")
        return

    for srt in srt_files:
        modified, report = process_srt_file(srt, rng)
        # Write report next to the srt file with suffix
        report_path = srt.with_name(srt.stem + REPORT_SUFFIX)
        header = [
            f"Report for {srt.name}",
            f"Directory: {subtitle_dir}",
            f"Seed: {seed if seed is not None else '(random)'}",
            "-" * 60,
            "",
        ]
        report_path.write_text("\n".join(header + report), encoding="utf-8")
        status = "modified" if modified else "skipped"
        print(f"{srt.name}: {status} -> {report_path.name}")


# PUBLIC_INTERFACE
def main() -> None:
    """CLI entrypoint for the inaccuracy injector."""
    parser = argparse.ArgumentParser(
        description="Introduce slight inaccuracies into .srt files and generate reports (idempotent with backups)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility."
    )
    args = parser.parse_args()

    # Determine SUBTITLE directory (script is at SUBTITLE/tools)
    script_path = Path(__file__).resolve()
    subtitle_dir = script_path.parent.parent

    process_subtitle_directory(subtitle_dir, seed=args.seed)


if __name__ == "__main__":
    main()
