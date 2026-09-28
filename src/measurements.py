"""Versioned acoustic observations. No emotion or speaker-identity inference."""

import math
import numpy as np
import parselmouth


SCHEMA_VERSION = 2
EXTRACTOR_CONFIG = {"pitch_floor_hz": 75.0, "pitch_ceiling_hz": 600.0,
                    "pitch_method": "praat_autocorrelation", "time_step_s": 0.0,
                    "intensity_subtract_mean": True, "direction_threshold_hz": 5.0}
UNITS = {"f0_mean_hz": "Hz", "f0_std_hz": "Hz", "f0_delta_hz": "Hz",
         "intensity_mean_db": "dB re Praat reference (uncalibrated)",
         "rms_amplitude": "digital full scale", "rms_dbfs": "dBFS",
         "intensity_crossings_per_second": "crossings/s"}


def measure_segment(data: dict, start: float, end: float) -> dict:
    values = dict.fromkeys(UNITS)
    validity = dict.fromkeys(UNITS, "invalid_interval")
    quality = {"interval_status": "invalid_interval", "sample_count": 0,
               "pitch_frame_count": 0, "voiced_frame_count": 0,
               "voiced_fraction": None, "intensity_frame_count": 0,
               "clipped_sample_fraction": None,
               "extraction_errors": dict(data.get("feature_errors", {}))}
    result = {"schema_version": SCHEMA_VERSION, "measurement": values,
              "validity": validity, "units": dict(UNITS), "quality": quality,
              "extractor": data.get("extractor", {
                  "name": "praat-parselmouth", "version": parselmouth.__version__,
                  "config": dict(EXTRACTOR_CONFIG)})}
    if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end) or end <= start):
        return result
    duration = data["duration"]
    if start < 0 or end > duration + 1e-9 or start >= duration:
        quality["interval_status"] = "out_of_bounds"
        validity.update(dict.fromkeys(UNITS, "out_of_bounds"))
        return result
    end = min(end, duration)
    quality["interval_status"] = "ok"
    validity.update(dict.fromkeys(UNITS, "insufficient_frames"))

    sound = data["sound"]
    first = max(0, math.ceil((start - sound.x1) / sound.dx))
    last = min(sound.n_samples, math.ceil((end - sound.x1) / sound.dx))
    samples = sound.values[:, first:last]
    quality["sample_count"] = int(samples.size)
    if samples.size and np.isfinite(samples).all():
        rms = float(np.sqrt(np.mean(np.square(samples))))
        values["rms_amplitude"] = rms
        validity["rms_amplitude"] = "ok"
        quality["clipped_sample_fraction"] = float(np.mean(np.abs(samples) >= 32767 / 32768))
        if rms > 0:
            values["rms_dbfs"] = 20 * math.log10(rms)
            validity["rms_dbfs"] = "ok"
        else:
            validity["rms_dbfs"] = "silent"

    pitch = data["pitch"]
    times = pitch.xs() if pitch else np.array([])
    all_f0 = pitch.selected_array["frequency"][(times >= start) & (times < end)] if pitch else np.array([])
    if pitch is None:
        for feature in ("f0_mean_hz", "f0_std_hz", "f0_delta_hz"):
            validity[feature] = "extraction_failed"
    voiced = all_f0[np.isfinite(all_f0) & (all_f0 > 0)]
    quality["pitch_frame_count"] = int(all_f0.size)
    quality["voiced_frame_count"] = int(voiced.size)
    if all_f0.size:
        quality["voiced_fraction"] = float(voiced.size / all_f0.size)
    if voiced.size:
        values["f0_mean_hz"] = float(np.mean(voiced))
        validity["f0_mean_hz"] = "ok"
    elif all_f0.size:
        for feature in ("f0_mean_hz", "f0_std_hz", "f0_delta_hz"):
            validity[feature] = "unvoiced"
    if voiced.size >= 2:
        values["f0_std_hz"] = float(np.std(voiced))
        validity["f0_std_hz"] = "ok"
    if voiced.size >= 6:
        third = voiced.size // 3
        values["f0_delta_hz"] = float(np.mean(voiced[-third:]) - np.mean(voiced[:third]))
        validity["f0_delta_hz"] = "ok"

    intensity = data["intensity"]
    times = intensity.xs() if intensity else np.array([])
    db = intensity.values[0][(times >= start) & (times < end)] if intensity else np.array([])
    if intensity is None:
        validity["intensity_mean_db"] = "extraction_failed"
        validity["intensity_crossings_per_second"] = "extraction_failed"
    db = db[np.isfinite(db)]
    quality["intensity_frame_count"] = int(db.size)
    if values["rms_amplitude"] == 0:
        validity["intensity_mean_db"] = "silent"
        validity["intensity_crossings_per_second"] = "silent"
    elif db.size and values["rms_amplitude"] is not None:
        values["intensity_mean_db"] = float(np.mean(db))
        validity["intensity_mean_db"] = "ok"
        if db.size >= 2:
            crossings = np.count_nonzero(np.diff(np.sign(db - np.mean(db))) < 0)
            values["intensity_crossings_per_second"] = float(crossings / (end - start))
            validity["intensity_crossings_per_second"] = "proxy_unvalidated"
    return result


def legacy_prosody(record: dict) -> dict:
    """Deprecated display aliases. Unavailable observations stay null.

    The old intensity crossing estimator never measured syllables. Do not
    populate its former speaking_rate field, including for legacy consumers.
    """
    values = record["measurement"]
    db = values["intensity_mean_db"]
    delta = values["f0_delta_hz"]
    direction = "unknown" if delta is None else (
        "rising" if delta > 5 else "falling" if delta < -5 else "flat")
    return {"avg_pitch": values["f0_mean_hz"], "pitch_variance": values["f0_std_hz"],
            "pitch_direction": direction,
            "energy": None if db is None else max(0.0, min(1.0, (db - 40) / 40)),
            "speaking_rate": None, "deprecated": True}
