import io
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def client(monkeypatch, isolated_output):
    import app
    monkeypatch.setattr(app, "OUTPUT_DIR", isolated_output)
    monkeypatch.setattr(app, "UPLOAD_DIR", isolated_output.parent / "uploads")
    app.app.config.update(TESTING=True, MAX_CONTENT_LENGTH=500 * 1024 * 1024)
    return app.app.test_client()


def upload(client, path, filename="recording.wav", route="/api/measure", **fields):
    return client.post(route, data={"audio": (io.BytesIO(path.read_bytes()), filename), **fields},
                       content_type="multipart/form-data")


def test_measure_endpoint_has_no_provider_calls_and_safe_paths(client, tone, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Measurement endpoint used a provider")
    monkeypatch.setattr("prosody_pipeline.transcribe_audio", forbidden)
    monkeypatch.setattr("prosody_pipeline.analyze_with_llm", forbidden)
    response = upload(client, tone(), filename="../../outside.wav", speaker_id="test")
    assert response.status_code == 200
    result = response.get_json()
    assert result["analysis"] is None and result["measurement_only"]
    assert result["speaker_threshold"] is None
    assert result["segments"][0]["speaker_id"] == "test"
    assert not (tmp_path / "outside.wav").exists()
    assert client.get(result["measurements_url"]).status_code == 200
    assert client.get(result["visualization"]).status_code == 200


def test_repeated_upload_name_preserves_independent_recordings(client, tone):
    one = upload(client, tone(hz=200)).get_json()
    two = upload(client, tone(hz=300)).get_json()
    assert one["recording"]["recording_id"] != two["recording"]["recording_id"]
    assert one["recording"]["source_sha256"] != two["recording"]["source_sha256"]
    assert two["segments"][0]["measurement"]["f0_mean_hz"] == pytest.approx(300, abs=1)
    assert client.get(one["measurements_url"]).get_json()[0]["measurement"]["f0_mean_hz"] == pytest.approx(200, abs=1)


def test_transcript_mode_is_measurements_without_interpretation_by_default(client, tone, monkeypatch):
    monkeypatch.setattr("prosody_pipeline.transcribe_audio", lambda path: {
        "text": "Hello", "words": [], "segments": [{"start": 0, "end": 1, "text": "Hello"}]})
    monkeypatch.setattr("prosody_pipeline.analyze_with_llm", lambda *args, **kwargs: pytest.fail("Implicit interpretation"))
    response = upload(client, tone(), route="/api/analyze")
    assert response.status_code == 200
    result = response.get_json()
    assert result["transcript"] == "Hello" and result["analysis"] is None
    assert result["segments"][0]["validity"]["transcript_gap_before_s"] == "asr_boundary_gap"


@pytest.mark.parametrize("route", ["/api/measure", "/api/analyze", "/api/proof-test"])
def test_upload_routes_reject_missing_audio(client, route):
    assert client.post(route).status_code == 400


def test_empty_and_undecodable_inputs_fail_without_provider_calls(client, tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    assert upload(client, empty).status_code == 400
    empty.write_text("not audio")
    assert upload(client, empty).status_code == 400


def test_upload_limit_returns_json(client, tone, monkeypatch):
    import app
    monkeypatch.setitem(app.app.config, "MAX_CONTENT_LENGTH", 100)
    response = upload(client, tone())
    assert response.status_code == 413 and "error" in response.get_json()


def test_reverse_rejects_invalid_parameters_before_tagging(client, monkeypatch):
    monkeypatch.setattr("app.detect_emotions", lambda *args: pytest.fail("Should validate before a provider call"))
    for payload in ({"text": "hello", "crossfade_ms": -1}, {"text": "hello", "voice": []},
                    {"text": "hello", "generate_audio": "false"}, {"text": []}):
        assert client.post("/api/reverse", json=payload).status_code == 400


def test_incomplete_render_returns_inspectable_failure_without_audio_url(client, monkeypatch, isolated_output):
    from generation_manifest import IncompleteGenerationError
    from recordings import write_json
    manifest = isolated_output / "renders" / "failed-run" / "generation_manifest.json"
    write_json(manifest, {"complete": False, "segments": [{"status": "failed"}]})
    monkeypatch.setattr("app.detect_emotions", lambda text: [
        {"text": text, "emotion": "neutral"}])
    def incomplete(*args, **kwargs):
        raise IncompleteGenerationError(manifest)
    monkeypatch.setattr("app.generate_audio", incomplete)
    response = client.post("/api/reverse", json={"text": "hello", "generate_audio": True})
    data = response.get_json()
    assert response.status_code == 422 and not data["complete"]
    assert "audio_url" not in data
    assert client.get(data["manifest_url"]).get_json()["segments"][0]["status"] == "failed"


def test_comparison_instructions_equal_and_text_arm_has_no_timing(monkeypatch):
    import prosody_pipeline
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="stub"))])
    monkeypatch.setattr(prosody_pipeline, "OpenAI", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    rows = [{"text": "Example", "start": 17.9, "end": 20.5,
             "measurement": {"f0_mean_hz": 217.3}, "segment_id": "source-hash:000001"}]
    prosody_pipeline.analyze_with_llm(rows, text_only=True)
    prosody_pipeline.analyze_with_llm(rows, text_only=False)
    assert calls[0]["messages"][0] == calls[1]["messages"][0]
    text = calls[0]["messages"][1]["content"]
    assert all(part not in text for part in ("17.9", "20.5", "217.3", "source-hash"))
    assert "217.3" in calls[1]["messages"][1]["content"]


def test_home_page_exposes_measurement_first_modes(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b'Acoustic measurements (no API calls)' in page.data
    assert b'escapeHtml(seg.text' in page.data


def test_silence_can_be_measured_and_visualized_in_browser_api(client, tone):
    response = upload(client, tone(amplitude=0))
    assert response.status_code == 200
    record = response.get_json()["segments"][0]
    assert record["measurement"]["f0_mean_hz"] is None
    assert record["measurement"]["rms_amplitude"] == 0
    assert record["prosody"]["energy"] is None
