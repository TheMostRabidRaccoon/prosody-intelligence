"""
Prosody Intelligence — Web UI
Rabid Raccoon Intelligence, LLC

Flask app providing a browser interface for the forward and reverse pipelines.

Usage:
    python app.py
    # Opens at http://localhost:5050
"""

import json
import os
import time
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Import our pipelines
import numpy as np

from prosody_pipeline import (
    transcribe_audio,
    extract_prosody,
    align_transcript_with_prosody,
    analyze_with_llm,
    visualize_prosody,
    OUTPUT_DIR,
)
from reverse_pipeline import (
    detect_emotions,
    map_emotions_to_params,
    generate_audio,
    VOICE_MAP,
    EMOTION_PARAMS,
)

load_dotenv(Path.home() / ".env")

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB — allow long drive-in recordings
CORS(app)

UPLOAD_DIR = OUTPUT_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _compute_speaker_threshold(annotated: list) -> float:
    """Compute adaptive speaker split threshold from annotated segments."""
    voiced = [
        seg["prosody"]["avg_pitch"]
        for seg in annotated
        if seg["prosody"]["avg_pitch"] > 0
        and seg["prosody"]["pitch_direction"] != "unknown"
        and seg["prosody"]["speaking_rate"] < 50
    ]
    return float(np.median(voiced)) if voiced else 170.0


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", voices=VOICE_MAP, emotions=EMOTION_PARAMS)


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve generated output files (audio, images, JSON)."""
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Forward pipeline: upload audio → get annotated transcript + analysis + visualization.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    upload_path = UPLOAD_DIR / audio_file.filename
    audio_file.save(str(upload_path))

    try:
        steps = {}

        # Layer 1A: Transcribe
        t0 = time.time()
        transcript = transcribe_audio(str(upload_path))
        steps["transcribe"] = round(time.time() - t0, 1)

        # Layer 1B: Prosody extraction
        t0 = time.time()
        prosody_data = extract_prosody(str(upload_path))
        steps["prosody"] = round(time.time() - t0, 1)

        # Layer 2: Alignment
        t0 = time.time()
        annotated = align_transcript_with_prosody(transcript, prosody_data)
        steps["align"] = round(time.time() - t0, 1)

        # Save annotated JSON
        audio_name = Path(audio_file.filename).stem
        json_path = OUTPUT_DIR / f"{audio_name}_annotated.json"
        with open(json_path, "w") as f:
            json.dump(annotated, f, indent=2)

        # Visualization
        t0 = time.time()
        viz_path = visualize_prosody(annotated, prosody_data, audio_name)
        viz_filename = Path(viz_path).name
        steps["visualize"] = round(time.time() - t0, 1)

        # Layer 3: LLM Analysis (if requested)
        analysis = None
        skip_llm = request.form.get("skip_llm") == "true"
        if not skip_llm:
            t0 = time.time()
            analysis = analyze_with_llm(annotated, text_only=False)
            steps["llm_analysis"] = round(time.time() - t0, 1)

        speaker_threshold = _compute_speaker_threshold(annotated)

        return jsonify({
            "success": True,
            "transcript": transcript["text"],
            "segments": annotated,
            "visualization": f"/output/{viz_filename}",
            "analysis": analysis,
            "duration": prosody_data["duration"],
            "segment_count": len(annotated),
            "speaker_threshold": speaker_threshold,
            "timing": steps,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/proof-test", methods=["POST"])
def api_proof_test():
    """
    Run A/B proof test: same transcript, text-only vs text+prosody.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    upload_path = UPLOAD_DIR / audio_file.filename
    audio_file.save(str(upload_path))

    try:
        # Layers 1+2
        transcript = transcribe_audio(str(upload_path))
        prosody_data = extract_prosody(str(upload_path))
        annotated = align_transcript_with_prosody(transcript, prosody_data)

        audio_name = Path(audio_file.filename).stem
        viz_path = visualize_prosody(annotated, prosody_data, audio_name)
        viz_filename = Path(viz_path).name

        # Pass A: Text only
        text_analysis = analyze_with_llm(annotated, text_only=True)

        # Pass B: Text + Prosody
        prosody_analysis = analyze_with_llm(annotated, text_only=False)

        speaker_threshold = _compute_speaker_threshold(annotated)

        return jsonify({
            "success": True,
            "transcript": transcript["text"],
            "segments": annotated,
            "visualization": f"/output/{viz_filename}",
            "text_analysis": text_analysis,
            "prosody_analysis": prosody_analysis,
            "duration": prosody_data["duration"],
            "segment_count": len(annotated),
            "speaker_threshold": speaker_threshold,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reverse", methods=["POST"])
def api_reverse():
    """
    Reverse pipeline: text → emotion detection → TTS audio.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    voice = data.get("voice", "claude")
    generate_tts = data.get("generate_audio", False)
    multi_voice = data.get("multi_voice", False)
    crossfade_ms = data.get("crossfade_ms", 100)

    try:
        # Step 1: Emotion detection
        tagged = detect_emotions(text)

        # Step 2: Parameter mapping
        mapped = map_emotions_to_params(tagged)

        result = {
            "success": True,
            "emotion_map": mapped,
        }

        # Step 3: TTS (if requested)
        if generate_tts:
            output_name = f"reverse_{voice}_{int(time.time())}"
            paths = generate_audio(
                mapped,
                voice_key=voice,
                output_name=output_name,
                multi_voice=multi_voice,
                crossfade_ms=crossfade_ms,
            )
            if paths:
                result["audio_url"] = f"/output/{Path(paths[0]).name}"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("PROSODY INTELLIGENCE — Web UI")
    print("http://localhost:5050")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=True)
