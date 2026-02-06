#!/usr/bin/env python3
"""Compare two RTTM files and compute Diarization Error Rate (DER).

Uses frame-level comparison at 10ms resolution. No external dependencies.

RTTM format: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>

Usage:
    python scripts/compare_rttm.py ref_output/ref.rttm test_output/test.rttm
    python scripts/compare_rttm.py ref.rttm test.rttm --collar 0.5

Exit code: 0 always (informational tool)
"""

import argparse
import sys
from collections import defaultdict


FRAME_RATE = 100  # 10ms frames = 100 frames/sec


def parse_rttm(path: str) -> list:
    """Parse RTTM file, return list of (start_sec, dur_sec, speaker) tuples."""
    segments = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if len(parts) < 8:
                print(f"WARNING: Skipping malformed line {line_num} in {path}: {line}",
                      file=sys.stderr)
                continue
            if parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                segments.append((start, dur, spk))
            except (ValueError, IndexError) as e:
                print(f"WARNING: Parse error at line {line_num} in {path}: {e}",
                      file=sys.stderr)
    return segments


def segments_to_frames(segments: list, num_frames: int, collar: float = 0.0) -> dict:
    """Convert segment list to per-frame speaker sets.

    Returns dict mapping frame_idx -> set of active speakers.
    If collar > 0, exclude frames within collar of segment boundaries from scoring.
    """
    frame_labels = defaultdict(set)
    for start, dur, spk in segments:
        start_frame = int(round(start * FRAME_RATE))
        end_frame = int(round((start + dur) * FRAME_RATE))
        for f in range(max(0, start_frame), min(num_frames, end_frame)):
            frame_labels[f].add(spk)
    return frame_labels


def get_collar_frames(segments: list, num_frames: int, collar: float) -> set:
    """Get set of frame indices that fall within collar of any segment boundary."""
    if collar <= 0:
        return set()

    collar_frames_set = set()
    collar_f = int(round(collar * FRAME_RATE))

    for start, dur, _spk in segments:
        # Collar around segment start
        start_frame = int(round(start * FRAME_RATE))
        for f in range(max(0, start_frame - collar_f), min(num_frames, start_frame + collar_f)):
            collar_frames_set.add(f)
        # Collar around segment end
        end_frame = int(round((start + dur) * FRAME_RATE))
        for f in range(max(0, end_frame - collar_f), min(num_frames, end_frame + collar_f)):
            collar_frames_set.add(f)

    return collar_frames_set


def compute_der(ref_segments: list, hyp_segments: list, collar: float = 0.25):
    """Compute DER using frame-level comparison.

    Returns dict with: missed, false_alarm, confusion, total_ref, der
    """
    if not ref_segments and not hyp_segments:
        return {
            "missed": 0, "false_alarm": 0, "confusion": 0,
            "total_ref": 0, "der": 0.0
        }

    # Determine total duration from both files
    max_time = 0.0
    for start, dur, _ in ref_segments + hyp_segments:
        max_time = max(max_time, start + dur)

    if max_time == 0.0:
        return {
            "missed": 0, "false_alarm": 0, "confusion": 0,
            "total_ref": 0, "der": 0.0
        }

    num_frames = int(round(max_time * FRAME_RATE)) + 1

    ref_frames = segments_to_frames(ref_segments, num_frames)
    hyp_frames = segments_to_frames(hyp_segments, num_frames)

    # Get collar exclusion frames (based on reference boundaries)
    collar_exclusion = get_collar_frames(ref_segments, num_frames, collar)

    missed = 0
    false_alarm = 0
    confusion = 0
    total_ref = 0

    for f in range(num_frames):
        if f in collar_exclusion:
            continue

        ref_spks = ref_frames.get(f, set())
        hyp_spks = hyp_frames.get(f, set())

        n_ref = len(ref_spks)
        n_hyp = len(hyp_spks)

        total_ref += n_ref

        if n_ref == 0 and n_hyp == 0:
            continue

        # Count correct: intersection (optimal assignment approximation)
        n_correct = len(ref_spks & hyp_spks)

        # If speakers don't overlap by name, use min overlap as "correct"
        # This is a simplification — true DER uses optimal matching
        if n_correct == 0 and n_ref > 0 and n_hyp > 0:
            # No name overlap — treat as confusion
            n_matched = min(n_ref, n_hyp)
            confusion += n_matched
            if n_ref > n_hyp:
                missed += n_ref - n_hyp
            elif n_hyp > n_ref:
                false_alarm += n_hyp - n_ref
        else:
            # Some speakers matched by name
            missed += max(0, n_ref - n_hyp)
            false_alarm += max(0, n_hyp - n_ref)
            # Confusion: speakers that don't match
            n_mismatched_ref = n_ref - n_correct
            n_mismatched_hyp = n_hyp - n_correct
            confusion += min(n_mismatched_ref, n_mismatched_hyp)

    der = (missed + false_alarm + confusion) / total_ref if total_ref > 0 else 0.0

    return {
        "missed": missed,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "total_ref": total_ref,
        "der": der,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two RTTM files (DER)")
    parser.add_argument("reference", help="Reference RTTM file")
    parser.add_argument("hypothesis", help="Hypothesis RTTM file")
    parser.add_argument("--collar", type=float, default=0.25,
                        help="Collar in seconds around segment boundaries (default: 0.25)")
    args = parser.parse_args()

    # Parse RTTM files
    print(f"Reference: {args.reference}")
    ref_segs = parse_rttm(args.reference)
    print(f"  Segments: {len(ref_segs)}")

    print(f"Hypothesis: {args.hypothesis}")
    hyp_segs = parse_rttm(args.hypothesis)
    print(f"  Segments: {len(hyp_segs)}")

    # Handle empty case
    if not ref_segs and not hyp_segs:
        print(f"\nBoth files are empty — nothing to compare.")
        print(f"DER: 0.00%")
        return

    if not ref_segs:
        print(f"\nWARNING: Reference is empty. Cannot compute DER (no reference speech).")
        if hyp_segs:
            total_hyp_dur = sum(dur for _, dur, _ in hyp_segs)
            print(f"  Hypothesis has {len(hyp_segs)} segments, {total_hyp_dur:.2f}s total")
        return

    # Print segment details
    print(f"\nReference speakers:")
    ref_by_spk = defaultdict(float)
    for start, dur, spk in ref_segs:
        ref_by_spk[spk] += dur
    for spk, total in sorted(ref_by_spk.items()):
        print(f"  {spk}: {total:.2f}s")

    print(f"Hypothesis speakers:")
    hyp_by_spk = defaultdict(float)
    for start, dur, spk in hyp_segs:
        hyp_by_spk[spk] += dur
    for spk, total in sorted(hyp_by_spk.items()):
        print(f"  {spk}: {total:.2f}s")

    # Compute DER
    result = compute_der(ref_segs, hyp_segs, collar=args.collar)

    print(f"\n{'='*50}")
    print(f"  Collar:        {args.collar:.2f}s")
    print(f"  Total ref frames (scored): {result['total_ref']}")
    print(f"  Missed speech:    {result['missed']} frames")
    print(f"  False alarm:      {result['false_alarm']} frames")
    print(f"  Speaker confusion: {result['confusion']} frames")
    print(f"  DER:              {result['der']*100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
