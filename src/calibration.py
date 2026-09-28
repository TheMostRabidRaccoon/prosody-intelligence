"""Manifest-backed acoustic target checks. These scores do not measure emotion accuracy."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from prosody_pipeline import OUTPUT_DIR, extract_prosody, get_segment_prosody
from generation_manifest import load_generation_manifest, manifest_file
from recordings import write_json

CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_LOG = CALIBRATION_DIR / "calibration_log.json"

# Legacy, manually selected target ranges. Retained for descriptive comparisons,
# not validated psychological labels. The syllable-rate target cannot be scored
# by the current intensity-crossing proxy.
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


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def calibrate_from_segments(audio_path: str, emotion_map_path: str | None = None,
                            prosody_data: dict | None = None) -> list:
    if prosody_data is not None:
        raise ValueError("Calibration measures individual manifest clips, not an assembled-track query")
    manifest = load_generation_manifest(audio_path, emotion_map_path)
    directory = Path(audio_path).resolve().parent
    records = []
    for segment in manifest["segments"]:
        path = manifest_file(directory, segment["file"])
        data = extract_prosody(str(path))
        observed = get_segment_prosody(data, 0, data["duration"])
        aliases = observed["prosody"]
        expected = EXPECTED_SIGNATURES.get(segment["emotion"], EXPECTED_SIGNATURES["neutral"])
        deltas = {}
        assessed = in_range = 0
        for feature in ("pitch_variance", "energy", "speaking_rate"):
            value = aliases[feature]
            low, high = expected[feature]
            if value is None:
                deltas[feature] = {"actual": None, "expected_range": [low, high],
                                   "status": "unavailable", "reason": "no_valid_measurement"}
                if feature == "speaking_rate":
                    deltas[feature]["reason"] = "intensity_crossings_are_not_syllables"
                continue
            assessed += 1
            status = "in_range" if low <= value <= high else "below" if value < low else "above"
            in_range += status == "in_range"
            deltas[feature] = {"actual": value, "expected_range": [low, high], "status": status}
        records.append({"schema_version": 2, "segment": segment["source_index"],
            "segment_id": segment["segment_id"], "source_line": segment["source_line"],
            "source_sha256": segment["sha256"], "voice_id": segment["voice_id"],
            "model_id": segment["model_id"], "emotion": segment["emotion"], "text": segment["text"],
            "intended_tts_params": segment["voice_settings"], "measurement": observed["measurement"],
            "validity": observed["validity"], "units": observed["units"],
            "quality": observed["quality"], "extractor": observed["extractor"],
            "achieved_prosody": aliases, "deltas": deltas,
            "target_range_coverage_pct": 100 * in_range / assessed if assessed else None,
            "assessed_feature_count": assessed, "target_feature_count": 3,
            "scoring_status": "partial" if assessed < 3 else "complete",
            "target_definition": "legacy_manual_ranges_v1",
            "measurement_scope": "individual_clip",
            "timestamp": {"start": segment["start_ms"] / 1000, "end": segment["end_ms"] / 1000}})
    return records


def _summarize_by_emotion(records: list, score_key="target_range_coverage_pct") -> dict:
    summary = {}
    for emotion in sorted({r["emotion"] for r in records}):
        rows = [r for r in records if r["emotion"] == emotion]
        summary[emotion] = {"count": len(rows),
            f"mean_{score_key}": _mean([r.get(score_key) for r in rows]),
            "scored_segment_count": sum(r.get(score_key) is not None for r in rows),
            "achieved_means": {feature: _mean([r["achieved_prosody"].get(feature) for r in rows])
                                for feature in ("pitch_variance", "energy", "speaking_rate")}}
    return summary


def load_calibration_log() -> list:
    # Keep historical observations unchanged. New runs are separate atomic files,
    # avoiding read/modify/write races and run-ID reuse across processes.
    legacy = json.loads(CALIBRATION_LOG.read_text()) if CALIBRATION_LOG.exists() else []
    current = [json.loads(path.read_text()) for path in sorted((CALIBRATION_DIR / "runs").glob("*.json"))]
    return legacy + current


def log_calibration_run(records: list, audio_name: str, source: str = "generated"):
    entry = {"run_id": uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
             "audio": audio_name, "source": source, "segment_count": len(records),
             "per_emotion_summary": _summarize_by_emotion(records), "segments": records,
             "overall_target_range_coverage_pct": _mean([r.get("target_range_coverage_pct") for r in records]),
             "score_definition": "legacy target-range coverage over available features; not emotion accuracy"}
    write_json(CALIBRATION_DIR / "runs" / f"{entry['run_id']}.json", entry)
    return entry


def print_calibration_report(emotion_filter: str | None = None):
    rows = [row for run in load_calibration_log() for row in run.get("segments", [])
            if not emotion_filter or row["emotion"] == emotion_filter]
    print("Acoustic target-range coverage (legacy heuristic targets; not emotion accuracy)")
    current = [r for r in rows if r.get("schema_version") == 2]
    historical = [r for r in rows if r.get("schema_version") != 2]
    print(json.dumps({"schema_2": _summarize_by_emotion(current),
                      "historical": {
                          "score_definition": "historical three-feature heuristic; not comparable with schema 2",
                          "by_emotion": _summarize_by_emotion(historical, score_key="accuracy")}},
                     indent=2, allow_nan=False))


def compare_to_ground_truth(generated_records: list, ground_truth_path: str) -> dict:
    """Descriptive, unmatched reference means; no inference of equivalent emotion."""
    human = json.loads(Path(ground_truth_path).read_text())
    if not human or not generated_records or any(row.get("schema_version") != 2 for row in human):
        raise ValueError("Reference comparison requires nonempty schema-v2 measurement records")
    all_rows = human + generated_records
    reference = all_rows[0]["extractor"]
    for row in all_rows:
        if any(row["extractor"][key] != reference[key] for key in ("name", "version", "config", "conversion")):
            raise ValueError("Reference and generated measurements use incompatible extractors")
    comparison = {}
    for feature in ("f0_mean_hz", "f0_std_hz", "intensity_mean_db", "rms_dbfs"):
        left = [r["measurement"][feature] for r in human if r["validity"][feature] == "ok"]
        right = [r["measurement"][feature] for r in generated_records if r["validity"][feature] == "ok"]
        h, g = _mean(left), _mean(right)
        comparison[feature] = {"reference": h, "generated": g, "reference_n": len(left),
                               "generated_n": len(right), "delta": None if h is None or g is None else g - h}
    return {"comparison_type": "unmatched_descriptive_means", "is_emotion_validation": False,
            "features": comparison}


def run_calibration(audio_path: str, emotion_map_path: str | None = None,
                    ground_truth_path: str | None = None, source: str = "generated") -> dict:
    records = calibrate_from_segments(audio_path, emotion_map_path)
    comparison = compare_to_ground_truth(records, ground_truth_path) if ground_truth_path else None
    entry = log_calibration_run(records, str(Path(audio_path).resolve()), source)
    result = {"records": records, "run_entry": entry, "reference_comparison": comparison}
    path = CALIBRATION_DIR / "reports" / f"{entry['run_id']}.json"
    write_json(path, result)
    print(f"Calibration report: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifest-backed acoustic target checks")
    parser.add_argument("audio", nargs="?")
    parser.add_argument("emotion_map", nargs="?", help="Optional consistency check against original targets")
    parser.add_argument("--reference", "--ground-truth", dest="reference")
    parser.add_argument("--source", choices=["generated", "human"], default="generated")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--emotion")
    args = parser.parse_args()
    if args.report:
        print_calibration_report(args.emotion)
    elif args.audio:
        run_calibration(args.audio, args.emotion_map, args.reference, args.source)
    else:
        parser.print_help()
