"""
Prosody Intelligence — Layer 5.5: Calibration Loop
Rabid Raccoon Intelligence, LLC

Closes the feedback loop between reverse (generation) and forward (analysis)
pipelines. Runs forward prosody extraction on TTS-generated audio, compares
achieved acoustic features vs intended emotion targets, and logs deltas
for empirical tuning of EMOTION_PARAMS.

This is the piece that turns a rules engine into a calibrated controller.

Usage:
    # Calibrate a single generated audio file against its emotion map
    python3 calibration.py output/woodland_amendments_full.mp3 \\
                           output/woodland_amendments_emotion_map.json

    # Calibrate with ground truth comparison (human audio baseline)
    python3 calibration.py output/generated.mp3 \\
                           output/generated_emotion_map.json \\
                           --ground-truth output/human_annotated.json

    # View calibration history
    python3 calibration.py --report

    # View per-emotion summary
    python3 calibration.py --report --emotion sarcastic
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_LOG = CALIBRATION_DIR / "calibration_log.json"

# Import forward pipeline prosody extraction
from prosody_pipeline import extract_prosody, get_segment_prosody, ensure_wav

# Import reverse pipeline emotion params (the table we're calibrating)
from reverse_pipeline import EMOTION_PARAMS


# ──────────────────────────────────────────────────────────────
# Expected Prosodic Signatures per Emotion
#
# These are the acoustic signatures we EXPECT each emotion to
# produce. Derived from speech science literature + the intent
# behind each EMOTION_PARAMS entry. Ranges are (low, high).
#
# The calibration loop will measure whether TTS output actually
# lands in these ranges — if not, EMOTION_PARAMS needs tuning.
# ──────────────────────────────────────────────────────────────

EXPECTED_SIGNATURES = {
    # emotion: {feature: (low, high)}  — "what should this sound like?"
    "sarcastic": {
        "pitch_variance": (15, 60),     # moderate-high: the contempt needs movement
        "energy":         (0.3, 0.7),   # mid: not screaming, but present
        "speaking_rate":  (2.5, 5.0),   # slightly slow: deliberate
    },
    "comedic": {
        "pitch_variance": (20, 80),     # high: expressive, animated
        "energy":         (0.4, 0.9),   # mid-high: performing
        "speaking_rate":  (3.5, 7.0),   # normal-to-fast: comic timing
    },
    "dramatic": {
        "pitch_variance": (15, 70),     # high: theatrical range
        "energy":         (0.4, 0.9),   # mid-high: projecting
        "speaking_rate":  (1.5, 4.0),   # slow: weight and gravity
    },
    "excited": {
        "pitch_variance": (20, 80),     # high: bouncing around
        "energy":         (0.5, 1.0),   # high: full send
        "speaking_rate":  (4.0, 8.0),   # fast: can't contain it
    },
    "angry": {
        "pitch_variance": (10, 50),     # moderate: controlled rage vs explosion
        "energy":         (0.5, 1.0),   # high: intensity
        "speaking_rate":  (3.5, 7.0),   # fast: pressured
    },
    "urgent": {
        "pitch_variance": (10, 45),     # moderate: focused, not wild
        "energy":         (0.4, 0.9),   # mid-high: pressing
        "speaking_rate":  (4.5, 8.0),   # fast: clipped, hurried
    },
    "contempt": {
        "pitch_variance": (5, 30),      # low: controlled, measured disdain
        "energy":         (0.3, 0.7),   # mid: cold, not loud
        "speaking_rate":  (2.0, 4.5),   # slow: letting it drip
    },
    "disgusted": {
        "pitch_variance": (10, 50),     # moderate: visceral reaction
        "energy":         (0.3, 0.8),   # mid: recoiling
        "speaking_rate":  (2.5, 5.0),   # moderate: the words taste bad
    },
    "confident": {
        "pitch_variance": (5, 30),      # low: steady, unwavering
        "energy":         (0.4, 0.8),   # mid-high: present
        "speaking_rate":  (3.0, 5.5),   # normal: unhurried authority
    },
    "analytical": {
        "pitch_variance": (3, 20),      # very low: monotone precision
        "energy":         (0.2, 0.6),   # low-mid: clinical
        "speaking_rate":  (3.0, 5.5),   # normal: measured
    },
    "neutral": {
        "pitch_variance": (5, 35),      # moderate: baseline conversational
        "energy":         (0.3, 0.7),   # mid: unremarkable
        "speaking_rate":  (3.0, 6.0),   # normal: conversational
    },
    "hesitant": {
        "pitch_variance": (5, 40),      # variable: uncertainty wobbles
        "energy":         (0.1, 0.5),   # low: shrinking
        "speaking_rate":  (1.5, 4.0),   # slow: trailing off
    },
    "tender": {
        "pitch_variance": (5, 25),      # low: gentle, controlled
        "energy":         (0.1, 0.4),   # low: hushed
        "speaking_rate":  (2.0, 4.0),   # slow: intimate
    },
    "resigned": {
        "pitch_variance": (2, 15),      # very low: flat affect
        "energy":         (0.1, 0.4),   # low: drained
        "speaking_rate":  (1.5, 3.5),   # slow: no energy to rush
    },
}


# ──────────────────────────────────────────────────────────────
# Calibration Runner
# ──────────────────────────────────────────────────────────────

def extract_segment_audio_prosody(audio_path: str, start: float, end: float) -> dict:
    """
    Extract prosodic features for a specific time range from an audio file.
    Thin wrapper around the forward pipeline's extraction functions.
    """
    prosody_data = extract_prosody(audio_path)
    return get_segment_prosody(prosody_data, start, end)


def calibrate_from_segments(audio_path: str, emotion_map_path: str,
                            prosody_data: dict = None) -> list:
    """
    Run forward prosody extraction on each segment of generated audio,
    compare against intended emotion targets.

    Args:
        audio_path: Path to the combined generated audio (MP3)
        emotion_map_path: Path to the emotion map JSON from reverse pipeline
        prosody_data: Pre-extracted prosody data (optional, avoids re-extraction)

    Returns:
        List of calibration records, one per segment.
    """
    from pydub import AudioSegment
    import io

    audio_path = str(Path(audio_path).resolve())

    # Load emotion map
    with open(emotion_map_path) as f:
        emotion_map = json.load(f)

    print("=" * 60)
    print("PROSODY INTELLIGENCE — Calibration Runner")
    print(f"Audio:      {Path(audio_path).name}")
    print(f"Emotion map: {Path(emotion_map_path).name}")
    print(f"Segments:   {len(emotion_map)}")
    print("=" * 60)

    # Extract prosody from the full audio once
    if prosody_data is None:
        print("\n[Cal] Extracting prosody from generated audio...")
        prosody_data = extract_prosody(audio_path)

    total_duration = prosody_data["duration"]

    # We need segment boundaries. The emotion map doesn't store timing
    # from the combined audio — we need to detect them from the
    # individual segment files, or estimate from the full audio.
    # Strategy: look for individual segment MP3s first, fall back to
    # even split.
    stem = Path(emotion_map_path).stem.replace("_emotion_map", "")
    seg_dir = OUTPUT_DIR

    # Try to find individual segment files for precise timing
    segment_files = sorted(seg_dir.glob(f"{stem}_seg*.mp3"))

    if segment_files:
        print(f"[Cal] Found {len(segment_files)} individual segment files")
        # Calculate cumulative timing from segment durations
        boundaries = []
        cumulative = 0.0
        for sf in segment_files:
            seg_audio = AudioSegment.from_mp3(str(sf))
            seg_dur = len(seg_audio) / 1000.0
            boundaries.append((cumulative, cumulative + seg_dur))
            cumulative += seg_dur  # approximate — crossfade makes this slightly off
    else:
        print(f"[Cal] No segment files found — estimating even split")
        seg_dur = total_duration / len(emotion_map)
        boundaries = [(i * seg_dur, (i + 1) * seg_dur) for i in range(len(emotion_map))]

    # Run prosody extraction per segment
    print(f"\n[Cal] Analyzing {len(emotion_map)} segments...")
    records = []

    for i, (item, (start, end)) in enumerate(zip(emotion_map, boundaries)):
        emotion = item.get("emotion", "neutral")
        intended_params = item.get("tts_params", {})
        text = item.get("text", "")

        # Get actual prosody from the generated audio
        achieved = get_segment_prosody(prosody_data, start, end)

        # Look up expected signature
        expected = EXPECTED_SIGNATURES.get(emotion, EXPECTED_SIGNATURES["neutral"])

        # Calculate deltas: is the achieved value within expected range?
        deltas = {}
        in_range_count = 0
        total_features = 0

        for feature in ["pitch_variance", "energy", "speaking_rate"]:
            actual_val = achieved.get(feature, 0.0)
            exp_low, exp_high = expected[feature]
            total_features += 1

            if exp_low <= actual_val <= exp_high:
                delta_status = "in_range"
                in_range_count += 1
                delta_pct = 0.0
            elif actual_val < exp_low:
                delta_status = "below"
                delta_pct = round((exp_low - actual_val) / max(exp_low, 0.01) * 100, 1)
            else:
                delta_status = "above"
                delta_pct = round((actual_val - exp_high) / max(exp_high, 0.01) * 100, 1)

            deltas[feature] = {
                "actual": round(actual_val, 2),
                "expected_range": [exp_low, exp_high],
                "status": delta_status,
                "delta_pct": delta_pct,
            }

        accuracy = round(in_range_count / total_features * 100, 1) if total_features > 0 else 0.0

        # Status icon
        if accuracy == 100:
            icon = "✓"
        elif accuracy >= 66:
            icon = "~"
        else:
            icon = "✗"

        record = {
            "segment": i,
            "emotion": emotion,
            "text": text[:60],
            "intended_tts_params": intended_params,
            "achieved_prosody": {
                "avg_pitch": achieved["avg_pitch"],
                "pitch_direction": achieved["pitch_direction"],
                "pitch_variance": achieved["pitch_variance"],
                "energy": achieved["energy"],
                "speaking_rate": achieved["speaking_rate"],
            },
            "deltas": deltas,
            "accuracy": accuracy,
            "timestamp": {"start": round(start, 2), "end": round(end, 2)},
        }
        records.append(record)

        # Print compact status
        d_pv = deltas["pitch_variance"]
        d_en = deltas["energy"]
        d_sr = deltas["speaking_rate"]
        print(
            f"  {icon} Seg {i:2d} | {emotion:12s} | "
            f"PV:{d_pv['actual']:5.1f} [{d_pv['status']:8s}] | "
            f"E:{d_en['actual']:4.2f} [{d_en['status']:8s}] | "
            f"SR:{d_sr['actual']:4.1f} [{d_sr['status']:8s}] | "
            f"acc:{accuracy:5.1f}%"
        )

    return records


# ──────────────────────────────────────────────────────────────
# Delta Logger
# ──────────────────────────────────────────────────────────────

def load_calibration_log() -> list:
    """Load existing calibration log or create empty."""
    if CALIBRATION_LOG.exists():
        with open(CALIBRATION_LOG) as f:
            return json.load(f)
    return []


def save_calibration_log(log: list):
    """Persist calibration log to disk."""
    with open(CALIBRATION_LOG, "w") as f:
        json.dump(log, f, indent=2)


def log_calibration_run(records: list, audio_name: str, source: str = "generated"):
    """
    Append a calibration run to the persistent log.

    Args:
        records: List of per-segment calibration records
        audio_name: Name of the audio file tested
        source: "generated" (TTS output) or "human" (ground truth)
    """
    log = load_calibration_log()

    run_entry = {
        "run_id": len(log) + 1,
        "timestamp": datetime.now().isoformat(),
        "audio": audio_name,
        "source": source,
        "segment_count": len(records),
        "per_emotion_summary": _summarize_by_emotion(records),
        "overall_accuracy": round(
            np.mean([r["accuracy"] for r in records]), 1
        ) if records else 0.0,
        "segments": records,  # full per-segment data
    }

    log.append(run_entry)
    save_calibration_log(log)

    print(f"\n[Cal] Run #{run_entry['run_id']} logged to {CALIBRATION_LOG}")
    print(f"      Overall accuracy: {run_entry['overall_accuracy']}%")

    return run_entry


def _summarize_by_emotion(records: list) -> dict:
    """
    Aggregate calibration records by emotion.
    Returns per-emotion mean achieved values and accuracy.
    """
    from collections import defaultdict

    by_emotion = defaultdict(list)
    for r in records:
        by_emotion[r["emotion"]].append(r)

    summary = {}
    for emotion, items in sorted(by_emotion.items()):
        accs = [r["accuracy"] for r in items]
        pvs = [r["achieved_prosody"]["pitch_variance"] for r in items]
        ens = [r["achieved_prosody"]["energy"] for r in items]
        srs = [r["achieved_prosody"]["speaking_rate"] for r in items]

        summary[emotion] = {
            "count": len(items),
            "mean_accuracy": round(np.mean(accs), 1),
            "achieved_means": {
                "pitch_variance": round(np.mean(pvs), 2),
                "energy": round(np.mean(ens), 3),
                "speaking_rate": round(np.mean(srs), 2),
            },
            "achieved_std": {
                "pitch_variance": round(np.std(pvs), 2),
                "energy": round(np.std(ens), 3),
                "speaking_rate": round(np.std(srs), 2),
            },
            "expected_ranges": EXPECTED_SIGNATURES.get(emotion, EXPECTED_SIGNATURES["neutral"]),
        }

    return summary


# ──────────────────────────────────────────────────────────────
# Calibration Report
# ──────────────────────────────────────────────────────────────

def print_calibration_report(emotion_filter: str = None):
    """
    Print a summary of all calibration runs from the log.
    Shows per-emotion accuracy trends over time.
    """
    log = load_calibration_log()
    if not log:
        print("[Cal] No calibration data yet. Run a calibration first.")
        return

    print("=" * 70)
    print("PROSODY INTELLIGENCE — Calibration Report")
    print(f"Total runs: {len(log)}")
    print("=" * 70)

    # Aggregate across all runs
    from collections import defaultdict
    all_by_emotion = defaultdict(list)

    for run in log:
        for emotion, data in run.get("per_emotion_summary", {}).items():
            if emotion_filter and emotion != emotion_filter:
                continue
            all_by_emotion[emotion].append({
                "run_id": run["run_id"],
                "source": run["source"],
                "timestamp": run["timestamp"],
                "accuracy": data["mean_accuracy"],
                "means": data["achieved_means"],
            })

    if not all_by_emotion:
        filter_msg = f" for emotion '{emotion_filter}'" if emotion_filter else ""
        print(f"[Cal] No calibration data found{filter_msg}.")
        return

    print(f"\n{'Emotion':12s} | {'Runs':>4s} | {'Accuracy':>8s} | "
          f"{'PV mean':>8s} | {'E mean':>7s} | {'SR mean':>8s} | "
          f"{'Expected PV':>14s} | {'Expected E':>12s} | {'Expected SR':>14s}")
    print("─" * 115)

    for emotion in sorted(all_by_emotion.keys()):
        entries = all_by_emotion[emotion]
        n = len(entries)
        mean_acc = np.mean([e["accuracy"] for e in entries])
        mean_pv = np.mean([e["means"]["pitch_variance"] for e in entries])
        mean_en = np.mean([e["means"]["energy"] for e in entries])
        mean_sr = np.mean([e["means"]["speaking_rate"] for e in entries])

        exp = EXPECTED_SIGNATURES.get(emotion, EXPECTED_SIGNATURES["neutral"])

        # Color-code accuracy
        if mean_acc >= 80:
            acc_icon = "✓"
        elif mean_acc >= 50:
            acc_icon = "~"
        else:
            acc_icon = "✗"

        print(
            f"{emotion:12s} | {n:4d} | {acc_icon} {mean_acc:5.1f}% | "
            f"{mean_pv:8.2f} | {mean_en:7.3f} | {mean_sr:8.2f} | "
            f"[{exp['pitch_variance'][0]:4.0f}-{exp['pitch_variance'][1]:4.0f}]     | "
            f"[{exp['energy'][0]:4.2f}-{exp['energy'][1]:4.2f}] | "
            f"[{exp['speaking_rate'][0]:4.1f}-{exp['speaking_rate'][1]:4.1f}]     "
        )

    # Print TTS param recommendations for low-accuracy emotions
    low_acc = [(e, entries) for e, entries in all_by_emotion.items()
               if np.mean([x["accuracy"] for x in entries]) < 66]

    if low_acc:
        print(f"\n{'─' * 70}")
        print("TUNING RECOMMENDATIONS (accuracy < 66%)")
        print(f"{'─' * 70}")

        for emotion, entries in sorted(low_acc, key=lambda x: np.mean([e["accuracy"] for e in x[1]])):
            mean_pv = np.mean([e["means"]["pitch_variance"] for e in entries])
            mean_en = np.mean([e["means"]["energy"] for e in entries])
            mean_sr = np.mean([e["means"]["speaking_rate"] for e in entries])

            exp = EXPECTED_SIGNATURES.get(emotion, EXPECTED_SIGNATURES["neutral"])
            params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])

            recs = []

            # Pitch variance recommendations
            if mean_pv < exp["pitch_variance"][0]:
                recs.append(f"PV too low ({mean_pv:.1f} vs [{exp['pitch_variance'][0]}-{exp['pitch_variance'][1]}]) → decrease stability (currently {params['stability']})")
            elif mean_pv > exp["pitch_variance"][1]:
                recs.append(f"PV too high ({mean_pv:.1f} vs [{exp['pitch_variance'][0]}-{exp['pitch_variance'][1]}]) → increase stability (currently {params['stability']})")

            # Energy recommendations
            if mean_en < exp["energy"][0]:
                recs.append(f"Energy too low ({mean_en:.2f} vs [{exp['energy'][0]}-{exp['energy'][1]}]) → increase style (currently {params['style']})")
            elif mean_en > exp["energy"][1]:
                recs.append(f"Energy too high ({mean_en:.2f} vs [{exp['energy'][0]}-{exp['energy'][1]}]) → decrease style (currently {params['style']})")

            # Speaking rate recommendations
            if mean_sr < exp["speaking_rate"][0]:
                recs.append(f"Rate too slow ({mean_sr:.1f} vs [{exp['speaking_rate'][0]}-{exp['speaking_rate'][1]}]) → increase speed (currently {params['speed']})")
            elif mean_sr > exp["speaking_rate"][1]:
                recs.append(f"Rate too fast ({mean_sr:.1f} vs [{exp['speaking_rate'][0]}-{exp['speaking_rate'][1]}]) → decrease speed (currently {params['speed']})")

            print(f"\n  {emotion} (accuracy: {np.mean([e['accuracy'] for e in entries]):.1f}%):")
            for rec in recs:
                print(f"    → {rec}")

    print()


# ──────────────────────────────────────────────────────────────
# Ground Truth Comparison (Human Audio Baseline)
# ──────────────────────────────────────────────────────────────

def compare_to_ground_truth(generated_records: list,
                            ground_truth_path: str) -> dict:
    """
    Compare generated audio prosody against human audio analysis.
    The ground truth is a forward pipeline annotated JSON from real audio.

    This is the ultimate calibration: does our TTS sound like a human
    expressing the same emotion?
    """
    with open(ground_truth_path) as f:
        human_segments = json.load(f)

    print(f"\n[Cal] Comparing against human ground truth: {Path(ground_truth_path).name}")
    print(f"      Human segments: {len(human_segments)}")
    print(f"      Generated segments: {len(generated_records)}")

    # Build human prosody profile by computing overall stats
    human_prosody = {
        "avg_pitch": np.mean([s["prosody"]["avg_pitch"] for s in human_segments if s["prosody"]["avg_pitch"] > 0]),
        "pitch_variance": np.mean([s["prosody"]["pitch_variance"] for s in human_segments]),
        "energy": np.mean([s["prosody"]["energy"] for s in human_segments]),
        "speaking_rate": np.mean([s["prosody"]["speaking_rate"] for s in human_segments if s["prosody"]["speaking_rate"] < 50]),
    }

    gen_prosody = {
        "avg_pitch": np.mean([r["achieved_prosody"]["avg_pitch"] for r in generated_records if r["achieved_prosody"]["avg_pitch"] > 0]),
        "pitch_variance": np.mean([r["achieved_prosody"]["pitch_variance"] for r in generated_records]),
        "energy": np.mean([r["achieved_prosody"]["energy"] for r in generated_records]),
        "speaking_rate": np.mean([r["achieved_prosody"]["speaking_rate"] for r in generated_records if r["achieved_prosody"]["speaking_rate"] < 50]),
    }

    print(f"\n  {'Feature':18s} | {'Human':>10s} | {'Generated':>10s} | {'Delta':>10s}")
    print(f"  {'─' * 55}")

    comparison = {}
    for feature in ["avg_pitch", "pitch_variance", "energy", "speaking_rate"]:
        h = human_prosody[feature]
        g = gen_prosody[feature]
        delta = g - h
        pct = (delta / max(h, 0.01)) * 100

        comparison[feature] = {
            "human": round(float(h), 2),
            "generated": round(float(g), 2),
            "delta": round(float(delta), 2),
            "delta_pct": round(float(pct), 1),
        }

        print(f"  {feature:18s} | {h:10.2f} | {g:10.2f} | {delta:+10.2f} ({pct:+.1f}%)")

    return comparison


# ──────────────────────────────────────────────────────────────
# Full Calibration Pipeline
# ──────────────────────────────────────────────────────────────

def run_calibration(audio_path: str, emotion_map_path: str,
                    ground_truth_path: str = None,
                    source: str = "generated") -> dict:
    """
    Full calibration pipeline:
    1. Extract prosody from generated audio
    2. Compare per-segment achieved vs intended
    3. Log deltas to persistent calibration log
    4. Optionally compare against human ground truth
    5. Print per-emotion summary with tuning recommendations
    """
    audio_name = Path(audio_path).stem

    # Step 1-2: Run calibration
    records = calibrate_from_segments(audio_path, emotion_map_path)

    # Step 3: Log to persistent store
    run_entry = log_calibration_run(records, audio_name, source=source)

    # Step 4: Ground truth comparison (optional)
    gt_comparison = None
    if ground_truth_path:
        gt_comparison = compare_to_ground_truth(records, ground_truth_path)

    # Step 5: Print summary
    summary = run_entry["per_emotion_summary"]
    print(f"\n{'─' * 60}")
    print("PER-EMOTION CALIBRATION SUMMARY")
    print(f"{'─' * 60}")
    print(f"{'Emotion':12s} | {'N':>3s} | {'Accuracy':>8s} | "
          f"{'PV':>6s} | {'Energy':>6s} | {'Rate':>6s}")
    print(f"{'─' * 55}")

    for emotion in sorted(summary.keys()):
        s = summary[emotion]
        m = s["achieved_means"]
        acc = s["mean_accuracy"]
        icon = "✓" if acc >= 80 else ("~" if acc >= 50 else "✗")
        print(
            f"{emotion:12s} | {s['count']:3d} | {icon} {acc:5.1f}% | "
            f"{m['pitch_variance']:6.1f} | {m['energy']:6.3f} | {m['speaking_rate']:6.1f}"
        )

    result = {
        "records": records,
        "run_entry": run_entry,
        "ground_truth_comparison": gt_comparison,
    }

    # Save detailed report
    report_path = CALIBRATION_DIR / f"{audio_name}_calibration.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[Cal] Detailed report saved: {report_path}")

    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prosody Intelligence — Calibration Loop"
    )
    parser.add_argument("audio", nargs="?", help="Path to generated audio (MP3)")
    parser.add_argument("emotion_map", nargs="?", help="Path to emotion map JSON")
    parser.add_argument("--ground-truth", help="Path to human annotated JSON for comparison")
    parser.add_argument("--source", default="generated",
                        choices=["generated", "human"],
                        help="Source type: 'generated' (TTS) or 'human' (real audio)")
    parser.add_argument("--report", action="store_true",
                        help="Print calibration report from log")
    parser.add_argument("--emotion", help="Filter report to specific emotion")

    args = parser.parse_args()

    if args.report:
        print_calibration_report(emotion_filter=args.emotion)
    elif args.audio and args.emotion_map:
        run_calibration(
            args.audio,
            args.emotion_map,
            ground_truth_path=args.ground_truth,
            source=args.source,
        )
    else:
        parser.print_help()
