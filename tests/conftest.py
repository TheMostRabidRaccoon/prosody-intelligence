import socket

import numpy as np
import parselmouth
import pytest


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("PROSODY_OUTPUT_DIR", str(output))
    import prosody_pipeline
    monkeypatch.setattr(prosody_pipeline, "OUTPUT_DIR", output)

    def no_network(*args, **kwargs):
        raise AssertionError("Tests must not contact external services")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    return output


@pytest.fixture
def tone(tmp_path):
    def make(name="tone.wav", hz=200, duration=1.0, amplitude=0.1):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(round(duration * 16000)) / 16000
        parselmouth.Sound(amplitude * np.sin(2 * np.pi * hz * t), sampling_frequency=16000).save(str(path), "WAV")
        return path
    return make
