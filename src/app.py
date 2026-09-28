"""Browser/API access to versioned acoustic records and optional interpretation."""

import os
from pathlib import Path
import subprocess

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import parselmouth

from prosody_pipeline import OUTPUT_DIR, run_pipeline, run_proof_test
from reverse_pipeline import detect_emotions, map_emotions_to_params, generate_audio, VOICE_MAP, EMOTION_PARAMS
from recordings import save_recording
from generation_manifest import IncompleteGenerationError

load_dotenv(Path.home() / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("PROSODY_MAX_UPLOAD_MB", "500")) * 1024 * 1024
CORS(app)
UPLOAD_DIR = OUTPUT_DIR.parent / "uploads"


def _url(path):
    return "/output/" + Path(path).resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()


def _upload():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        raise ValueError("An audio file with a filename is required")
    return save_recording(audio.stream, audio.filename, UPLOAD_DIR)


def _context():
    result = {}
    for name in ("speaker_id", "channel", "register"):
        value = request.form.get(name, "").strip()
        if len(value) > 128:
            raise ValueError(f"{name} must be at most 128 characters")
        result[name] = value or None
    return result


def _response(result):
    segments = result["annotated_segments"]
    response = {"success": True, "schema_version": result["schema_version"],
                "recording": result["recording"], "transcript": result["transcript"]["text"],
                "segments": segments, "duration": result["duration"], "segment_count": len(segments),
                "measurement_only": result["measurement_only"],
                "measurements_url": _url(result["output_path"]),
                "visualization": _url(result["viz_path"]) if "viz_path" in result else None,
                "analysis": result.get("analysis"), "speaker_threshold": None}
    for key in ("text_analysis", "prosody_analysis", "evaluation_status"):
        if key in result:
            response[key] = result[key]
    return jsonify(response)


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="Audio exceeds the configured upload limit"), 413


@app.errorhandler(ValueError)
def invalid_request(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(subprocess.CalledProcessError)
@app.errorhandler(parselmouth.PraatError)
def invalid_audio(error):
    return jsonify(error="Audio could not be decoded or measured"), 400


@app.route("/")
def index():
    return render_template("index.html", voices=VOICE_MAP, emotions=EMOTION_PARAMS)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/measure", methods=["POST"])
def api_measure():
    context = _context()
    recording = _upload()
    return _response(run_pipeline(str(recording.path), recording=recording,
                                  measure_only=True, visualize=True, **context))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    context = _context()
    skip_llm = request.form.get("skip_llm", "true").lower()
    if skip_llm not in ("true", "false"):
        raise ValueError("skip_llm must be true or false")
    recording = _upload()
    return _response(run_pipeline(str(recording.path), recording=recording,
                                  skip_llm=skip_llm == "true", visualize=True, **context))


@app.route("/api/proof-test", methods=["POST"])
def api_proof_test():
    recording = _upload()
    return _response(run_proof_test(str(recording.path), recording=recording))


@app.route("/api/reverse", methods=["POST"])
def api_reverse():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str) or not data["text"].strip():
        raise ValueError("Nonempty text is required")
    voice = data.get("voice", "claude")
    if not isinstance(voice, str) or voice not in VOICE_MAP:
        raise ValueError("Unknown voice")
    crossfade = data.get("crossfade_ms", 100)
    if type(crossfade) is not int or crossfade < 0:
        raise ValueError("crossfade_ms must be a nonnegative integer")
    for flag in ("generate_audio", "multi_voice"):
        if flag in data and not isinstance(data[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    tagged = detect_emotions(data["text"])
    mapped = map_emotions_to_params(tagged)
    result = {"success": True, "emotion_map": mapped}
    if data.get("generate_audio", False):
        try:
            paths = generate_audio(mapped, voice_key=voice, multi_voice=data.get("multi_voice", False),
                                   crossfade_ms=crossfade)
        except IncompleteGenerationError as error:
            return jsonify(success=False, error=str(error), complete=False,
                           manifest_url=_url(error.manifest_path)), 422
        result.update(audio_url=_url(paths[0]), complete=True,
                      manifest_url=_url(Path(paths[0]).parent / "generation_manifest.json"))
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PROSODY_PORT", "5050"))
    debug = os.getenv("PROSODY_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
