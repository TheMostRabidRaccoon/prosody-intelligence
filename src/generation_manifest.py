"""Shared, verified render provenance for calibration and video assembly."""

import json
from pathlib import Path

from recordings import file_sha256


class IncompleteGenerationError(RuntimeError):
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        super().__init__("Audio generation is incomplete; inspect the generation manifest")


def manifest_file(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Manifest references a missing file or a path outside its run directory")
    return path


def load_generation_manifest(audio_path: str, emotion_map_path: str | None = None) -> dict:
    audio = Path(audio_path).resolve()
    manifest_path = audio.parent / "generation_manifest.json"
    if not manifest_path.exists():
        raise ValueError("No generation manifest. Regenerate with the current reverse pipeline; "
                         "timing and target alignment will not be guessed from filenames.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("complete"):
        raise ValueError("Calibration and video require a complete version-1 generation manifest")
    if manifest_file(audio.parent, manifest["audio_file"]) != audio:
        raise ValueError("Audio does not match the manifest")
    if file_sha256(audio) != manifest["audio_sha256"]:
        raise ValueError("Combined audio content does not match the manifest")
    segments = manifest["segments"]
    if not segments or len({s["segment_id"] for s in segments}) != len(segments):
        raise ValueError("Manifest segment IDs must be nonempty and unique")
    end_ms = 0
    for index, segment in enumerate(segments):
        if segment["status"] != "ok" or segment["source_index"] != index:
            raise ValueError("Manifest contains missing or reordered source segments")
        duration = segment["duration_ms"]
        overlap = segment["crossfade_ms"]
        if (not isinstance(duration, int) or duration <= 0 or not isinstance(overlap, int)
                or overlap < 0 or overlap > min(end_ms, duration) or (index == 0 and overlap)):
            raise ValueError("Invalid manifest duration or crossfade")
        start_ms = end_ms - overlap
        end_ms = start_ms + duration
        if segment["start_ms"] != start_ms or segment["end_ms"] != end_ms:
            raise ValueError("Manifest timing is inconsistent")
        path = manifest_file(audio.parent, segment["file"])
        if file_sha256(path) != segment["sha256"]:
            raise ValueError("Segment content does not match the manifest")
    if manifest["duration_ms"] != end_ms:
        raise ValueError("Manifest total duration is inconsistent")
    if emotion_map_path:
        targets = json.loads(Path(emotion_map_path).read_text())
        if len(targets) != len(segments):
            raise ValueError("Emotion map does not match manifest source lines")
        for target, segment in zip(targets, segments):
            if any(target.get(k) != segment[k] for k in ("text", "emotion", "tts_params")):
                raise ValueError("Emotion map does not match manifest targets")
    return manifest
