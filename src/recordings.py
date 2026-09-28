"""Recording identity and atomic artifact storage. No model/provider dependencies."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO
from uuid import uuid4


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
CONVERSION = {"version": 1, "sample_rate_hz": 16000, "channels": 1, "sample_format": "pcm_s16le"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    """Publish complete, standards-compliant JSON, never a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".json-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Recording:
    recording_id: str
    path: Path
    original_filename: str
    sha256: str
    byte_count: int

    def metadata(self) -> dict:
        return {"recording_id": self.recording_id, "original_filename": self.original_filename,
                "source_sha256": self.sha256, "byte_count": self.byte_count}


def save_recording(stream: BinaryIO, original_filename: str, root: str | Path) -> Recording:
    """Snapshot an input under a generated ID; client filenames are metadata only."""
    suffix = Path(original_filename).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise ValueError("Unsupported audio extension")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    recording_id = uuid4().hex
    directory = root / recording_id
    directory.mkdir()
    destination = directory / f"source{suffix}"
    try:
        with destination.open("xb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
        byte_count = destination.stat().st_size
        if byte_count == 0:
            raise ValueError("The uploaded audio file is empty")
        recording = Recording(recording_id, destination, original_filename,
                              file_sha256(destination), byte_count)
        write_json(directory / "recording.json", recording.metadata())
        return recording
    except Exception:
        shutil.rmtree(directory)
        raise


def snapshot_recording(path: str | Path, root: str | Path) -> Recording:
    path = Path(path)
    with path.open("rb") as stream:
        return save_recording(stream, path.name, root)
