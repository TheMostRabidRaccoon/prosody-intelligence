import json
from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment

from prosody_pipeline import extract_prosody, get_segment_prosody, run_pipeline
from recordings import save_recording


def test_known_tone_has_hz_units_full_precision_and_no_syllable_claim(tone):
    record = get_segment_prosody(extract_prosody(str(tone(hz=217.3))), 0, 1)
    assert record["measurement"]["f0_mean_hz"] == pytest.approx(217.3, abs=0.5)
    assert record["measurement"]["f0_std_hz"] < 0.5
    assert record["units"]["f0_std_hz"] == "Hz"
    assert record["validity"]["intensity_crossings_per_second"] == "proxy_unvalidated"
    assert record["prosody"]["speaking_rate"] is None
    assert record["quality"]["voiced_fraction"] == 1


def test_silence_distinguishes_valid_zero_amplitude_from_missing_pitch(tone):
    record = get_segment_prosody(extract_prosody(str(tone(amplitude=0))), 0, 1)
    assert record["measurement"]["rms_amplitude"] == 0
    assert record["validity"]["rms_amplitude"] == "ok"
    assert record["measurement"]["f0_mean_hz"] is None
    assert record["validity"]["f0_mean_hz"] == "unvoiced"
    assert record["measurement"]["intensity_mean_db"] is None
    assert record["validity"]["intensity_mean_db"] == "silent"
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("start,end,status", [(0.5, 0.4, "invalid_interval"),
    (0, 0, "invalid_interval"), (float("nan"), 1, "invalid_interval"),
    (0, float("inf"), "invalid_interval"), (-0.1, 1, "out_of_bounds"),
    (0, 1.1, "out_of_bounds"), (2, 3, "out_of_bounds")])
def test_invalid_intervals_do_not_manufacture_measurements(tone, start, end, status):
    record = get_segment_prosody(extract_prosody(str(tone())), start, end)
    assert record["quality"]["interval_status"] == status
    assert all(value is None for value in record["measurement"].values())
    json.dumps(record, allow_nan=False)


def test_same_basename_different_content_and_overwrite_are_not_reused(tone):
    def mp3(name, hz):
        wav = tone(name, hz=hz)
        path = wav.with_suffix(".mp3")
        AudioSegment.from_wav(str(wav)).export(str(path), format="mp3").close()
        return path
    first = mp3("a/recording.wav", hz=200)
    second = mp3("b/recording.wav", hz=300)
    def pitch(path):
        return get_segment_prosody(extract_prosody(str(path)), 0, 1)["measurement"]["f0_mean_hz"]
    assert pitch(first) == pytest.approx(200, abs=0.5)
    assert pitch(second) == pytest.approx(300, abs=0.5)
    mp3("a/recording.wav", hz=400)
    assert pitch(first) == pytest.approx(400, abs=0.5)


def test_measure_only_never_calls_transcription_or_interpretation(tone, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Measurement-only mode called a provider")
    monkeypatch.setattr("prosody_pipeline.transcribe_audio", forbidden)
    monkeypatch.setattr("prosody_pipeline.analyze_with_llm", forbidden)
    result = run_pipeline(str(tone()), measure_only=True, visualize=True, speaker_id="test-speaker")
    assert "analysis" not in result
    record = result["annotated_segments"][0]
    assert record["speaker_identity_source"] == "declared_single_speaker"
    assert record["measurement"]["transcript_gap_before_s"] is None
    assert len(record["source"]["source_sha256"]) == 64
    assert json.loads(Path(result["output_path"]).read_text()) == result["annotated_segments"]
    assert Path(result["viz_path"]).stat().st_size > 0


def test_filename_is_metadata_only(tone, tmp_path):
    path = tone()
    root = tmp_path / "uploads"
    with path.open("rb") as stream:
        first = save_recording(stream, "../../outside.wav", root)
    with path.open("rb") as stream:
        second = save_recording(stream, "/absolute/outside.wav", root)
    assert first.path.is_relative_to(root)
    assert second.path.is_relative_to(root)
    assert first.recording_id != second.recording_id
    assert first.sha256 == second.sha256
    assert not (tmp_path / "outside.wav").exists()


def test_too_short_for_praat_still_has_explicit_partial_measurements(tone):
    result = run_pipeline(str(tone(duration=0.005)), measure_only=True, visualize=True)
    observation = result["annotated_segments"][0]
    assert observation["measurement"]["rms_amplitude"] > 0
    assert observation["measurement"]["f0_mean_hz"] is None
    assert observation["validity"]["f0_mean_hz"] == "extraction_failed"
    json.dumps(result, allow_nan=False)
