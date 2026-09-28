import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from pydub import AudioSegment
from pydub.generators import Sine
import pytest

import reverse_pipeline as reverse
from generation_manifest import IncompleteGenerationError, load_generation_manifest
from calibration import calibrate_from_segments


@pytest.fixture
def render(monkeypatch, isolated_output):
    monkeypatch.setattr(reverse, "OUTPUT_DIR", isolated_output)
    def run(fail_at=None, durations=(1000, 500, 1500), crossfade=100):
        calls = []
        def convert(**kwargs):
            index = len(calls)
            calls.append(kwargs)
            if index == fail_at:
                raise RuntimeError("Simulated provider failure")
            stream = io.BytesIO()
            Sine((200, 300, 400)[index]).to_audio_segment(duration=durations[index]).export(stream, format="mp3")
            return iter([stream.getvalue()])
        monkeypatch.setattr("elevenlabs.ElevenLabs", lambda: SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert)))
        targets = reverse.map_emotions_to_params([
            {"line": 10, "text": "CLAUDE: first", "emotion": "neutral"},
            {"line": 20, "text": "GROK: middle", "emotion": "angry"},
            {"line": 30, "text": "KYRA: last", "emotion": "tender"}])
        return reverse.generate_audio(targets, multi_voice=True, crossfade_ms=crossfade), targets, calls
    return run


def test_manifest_records_actual_overlaps_voice_and_original_identity(render):
    paths, targets, calls = render()
    manifest = load_generation_manifest(paths[0])
    assert manifest["duration_ms"] == 2800
    assert len(AudioSegment.from_mp3(paths[0])) == 2800
    assert [s["start_ms"] for s in manifest["segments"]] == [0, 900, 1300]
    assert [s["end_ms"] for s in manifest["segments"]] == [1000, 1400, 2800]
    assert [s["source_line"] for s in manifest["segments"]] == [10, 20, 30]
    assert [s["voice_id"] for s in manifest["segments"]] == [c["voice_id"] for c in calls]
    assert [c["text"] for c in calls] == ["first", "middle", "last"]
    assert all(s["voice_settings"] == c["voice_settings"] for s, c in zip(manifest["segments"], calls))


def test_short_segments_use_actual_safe_fade(render):
    paths, _, _ = render(durations=(20, 100, 100), crossfade=100)
    manifest = load_generation_manifest(paths[0])
    assert [s["crossfade_ms"] for s in manifest["segments"]] == [0, 0, 50]
    assert manifest["duration_ms"] == 170


def test_calibration_measures_unblended_clips_with_correct_targets(render):
    paths, _, _ = render()
    records = calibrate_from_segments(paths[0], str(Path(paths[0]).parent / "emotion_map.json"))
    assert [r["emotion"] for r in records] == ["neutral", "angry", "tender"]
    assert [r["measurement"]["f0_mean_hz"] for r in records] == pytest.approx([200, 300, 400], abs=1)
    assert records[-1]["timestamp"] == {"start": 1.3, "end": 2.8}
    assert all(r["measurement_scope"] == "individual_clip" for r in records)
    assert all(r["deltas"]["speaking_rate"]["status"] == "unavailable" for r in records)
    assert all(r["assessed_feature_count"] == 2 and "accuracy" not in r for r in records)


def test_missing_middle_segment_cannot_become_a_full_production(render):
    with pytest.raises(IncompleteGenerationError) as failure:
        render(fail_at=1)
    path = failure.value.manifest_path
    manifest = json.loads(path.read_text())
    assert not manifest["complete"]
    assert [s["status"] for s in manifest["segments"]] == ["ok", "failed", "ok"]
    assert manifest["segments"][-1]["emotion"] == "tender"
    assert not (path.parent / "combined.mp3").exists()
    with pytest.raises(ValueError, match="complete"):
        calibrate_from_segments(str(path.parent / "combined.mp3"))


def test_missing_manifest_never_falls_back_to_even_split(tmp_path):
    with pytest.raises(ValueError, match="will not be guessed"):
        calibrate_from_segments(str(tmp_path / "legacy.mp3"))


def test_modified_clip_is_rejected_before_scoring(render):
    paths, _, _ = render()
    Path(paths[-1]).write_bytes(b"replaced clip")
    with pytest.raises(ValueError, match="Segment content"):
        calibrate_from_segments(paths[0])


def test_mismatched_map_rejected(render, tmp_path):
    paths, targets, _ = render()
    targets.reverse()
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(targets))
    with pytest.raises(ValueError, match="does not match"):
        calibrate_from_segments(paths[0], str(wrong))


def test_repeated_output_name_creates_disjoint_runs(render):
    first, _, _ = render()
    second, _, _ = render()
    assert Path(first[0]).parent != Path(second[0]).parent
    assert load_generation_manifest(first[0])["complete"]


def test_emotion_tagger_preserves_original_text_order_and_missing_lines(monkeypatch):
    raw = '[{"line": 2, "text": "rewritten", "emotion": "tender"}]'
    completion = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]))
    monkeypatch.setattr(reverse, "OpenAI", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completion)))
    rows = reverse.detect_emotions("ORIGINAL A\nORIGINAL B")
    assert [r["text"] for r in rows] == ["ORIGINAL A", "ORIGINAL B"]
    assert [r["emotion"] for r in rows] == ["neutral", "tender"]
    assert rows[0]["annotation_status"] == "fallback"


@pytest.mark.parametrize("crossfade", [-1, 2.5, True])
def test_bad_crossfade_rejected_before_provider(render, crossfade):
    with pytest.raises(ValueError, match="nonnegative integer"):
        render(crossfade=crossfade)


def test_video_uses_crossfaded_duration(render, tmp_path):
    from compositor import composite_animated_short
    paths, _, _ = render(durations=(200, 300, 250), crossfade=100)
    stills = tmp_path / "stills"
    stills.mkdir()
    output = tmp_path / "video.mp4"
    composite_animated_short(str(stills), paths[0],
        str(Path(paths[0]).parent / "emotion_map.json"), str(output),
        resolution=(160, 90), fps=20, subtitle_style="none")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                           capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert float(video["duration"]) == pytest.approx(0.55, abs=0.05)
    assert any(stream["codec_type"] == "audio" for stream in streams)


def test_calibration_runs_persist_separately_without_rewriting_history(render, tmp_path, monkeypatch):
    import calibration
    directory = tmp_path / "calibration"
    directory.mkdir()
    legacy = directory / "calibration_log.json"
    legacy.write_text('[{"run_id": 1, "segments": []}]')
    original = legacy.read_bytes()
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", directory)
    monkeypatch.setattr(calibration, "CALIBRATION_LOG", legacy)
    paths, _, _ = render()
    first = calibration.run_calibration(paths[0])
    second = calibration.run_calibration(paths[0])
    assert first["run_entry"]["run_id"] != second["run_entry"]["run_id"]
    assert len(calibration.load_calibration_log()) == 3
    assert legacy.read_bytes() == original


def test_report_does_not_average_incompatible_historical_scores(monkeypatch, capsys):
    import calibration
    historic = {"emotion": "neutral", "accuracy": 100, "achieved_prosody": {}}
    current = {"schema_version": 2, "emotion": "neutral", "target_range_coverage_pct": 0,
               "achieved_prosody": {}}
    monkeypatch.setattr(calibration, "load_calibration_log", lambda: [
        {"segments": [historic]}, {"segments": [current]}])
    calibration.print_calibration_report()
    report = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert report["schema_2"]["neutral"]["mean_target_range_coverage_pct"] == 0
    assert report["schema_2"]["neutral"]["count"] == 1
    assert report["historical"]["by_emotion"]["neutral"]["mean_accuracy"] == 100
