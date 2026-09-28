"""
Prosody Intelligence — Reverse Pipeline (Layer 4)
Rabid Raccoon Intelligence, LLC

Text → Emotion Detection → Prosody Parameter Mapping → ElevenLabs TTS → Audio

Text annotations choose intended delivery from a manual parameter table.
Each render records the actual voice, settings, clip identity and timing.
Neither these annotations nor target-range checks establish emotion accuracy.

Usage:
    python reverse_pipeline.py input.txt --voice claude
    python reverse_pipeline.py input.txt --voice grok --emotion-map   # show emotion map without TTS
    python reverse_pipeline.py input.txt --voice all                  # multi-voice performance

Voice assignments (from spec):
    claude  → Onyx-style (George - Warm Storyteller)
    gemini  → Adam (Dominant, Firm)
    grok    → Callum (Husky Trickster)
    gpt     → Marcus-style (Brian - Deep, Resonant)
    kyra    → Kyra (cloned voice)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from recordings import file_sha256, write_json
from generation_manifest import IncompleteGenerationError

from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.getenv("PROSODY_OUTPUT_DIR", str(PROJECT_ROOT / "output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(Path.home() / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Voice assignments — mapping character to ElevenLabs voice ID
VOICE_MAP = {
    "claude": {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",  # George - Warm, Captivating Storyteller
        "name": "George (Claude)",
        "description": "Dry snobbery, librarian energy, radioactive spider",
    },
    "gemini": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam - Dominant, Firm
        "name": "Adam (Gemini)",
        "description": "Theatrical gravitas, dramatic weight",
    },
    "grok": {
        "voice_id": "N2lVS1w4EtoT3dr4eOWO",  # Callum - Husky Trickster
        "name": "Callum (Grok)",
        "description": "Unhinged confidence, chaos energy",
    },
    "gpt": {
        "voice_id": "nPczCjzI2devNBz1zQrb",  # Brian - Deep, Resonant and Comforting
        "name": "Brian (GPT)",
        "description": "Muffled desperation from the penalty box",
    },
    "kyra": {
        "voice_id": "DLm68MJvI3f80aZijHAn",  # Kyra (cloned)
        "name": "Kyra",
        "description": "The raccoon pulling the strings",
    },
    "narrator": {
        "voice_id": "Aqqzjc8no56A9UgQcOnP",  # Narrator — observant, fourth-wall
        "name": "The Narrator",
        "description": "Observant stage directions, holds the fourth wall",
    },
}

# ──────────────────────────────────────────────────────────────
# Emotion → ElevenLabs Parameter Mapping (from spec Section 6.2)
# ──────────────────────────────────────────────────────────────

EMOTION_PARAMS = {
    # ── HIGH ENERGY: Push style hard, let the voice rip ──
    "sarcastic": {
        "stability": 0.25,
        "style": 0.9,
        "speed": 0.9,
        "description": "Deliberate, dry delivery — low stability lets the contempt land",
    },
    "comedic": {
        "stability": 0.15,
        "style": 1.0,
        "speed": 1.05,
        "description": "Maximum expressiveness — timing and pitch variation are everything",
    },
    "dramatic": {
        "stability": 0.3,
        "style": 1.0,
        "speed": 0.75,
        "description": "Theatrical weight — slow + max style = money moments",
    },
    "excited": {
        "stability": 0.2,
        "style": 0.95,
        "speed": 1.25,
        "description": "High energy, enthusiastic — fast and wild",
    },
    "angry": {
        "stability": 0.35,
        "style": 0.9,
        "speed": 1.15,
        "description": "Intense, forceful — controlled instability",
    },
    "urgent": {
        "stability": 0.4,
        "style": 0.8,
        "speed": 1.25,
        "description": "Pressured, clipped — speed drives the urgency",
    },
    "contempt": {
        "stability": 0.7,
        "style": 0.85,
        "speed": 0.8,
        "description": "Cold superiority — stable, slow, dripping with disdain",
    },
    "disgusted": {
        "stability": 0.4,
        "style": 0.9,
        "speed": 0.85,
        "description": "Visceral revulsion — unstable delivery, like the words taste bad",
    },
    # ── MID ENERGY: Balanced delivery ──
    "confident": {
        "stability": 0.6,
        "style": 0.6,
        "speed": 1.0,
        "description": "Steady, authoritative — stability is the power move",
    },
    "analytical": {
        "stability": 0.75,
        "style": 0.2,
        "speed": 0.95,
        "description": "Clinical, precise — high stability, minimal style",
    },
    "neutral": {
        "stability": 0.5,
        "style": 0.5,
        "speed": 1.0,
        "description": "Baseline conversational",
    },
    # ── LOW ENERGY: Pull back hard — teach the AI to get quiet ──
    "hesitant": {
        "stability": 0.25,
        "style": 0.3,
        "speed": 0.75,
        "description": "Uncertain, trailing off — slow + low style = shrinking",
    },
    "tender": {
        "stability": 0.7,
        "style": 0.35,
        "speed": 0.75,
        "description": "Warm, gentle — stable but hushed, no broadcast energy",
    },
    "resigned": {
        "stability": 0.8,
        "style": 0.15,
        "speed": 0.7,
        "description": "Flat, drained, defeated — near-monotone, slowest delivery",
    },
}


# ──────────────────────────────────────────────────────────────
# STEP 1: Emotion Detection (LLM-native)
# ──────────────────────────────────────────────────────────────

EMOTION_DETECTION_PROMPT = """You are an emotion tagger for a text-to-speech system. For each line of text, identify the emotional register that should drive the vocal performance.

Available emotions: sarcastic, urgent, confident, comedic, dramatic, analytical, hesitant, angry, tender, resigned, excited, neutral, contempt, disgusted

Rules:
1. Tag EVERY line with exactly ONE emotion from the list above.
2. Consider context — the same words can carry different emotions depending on what comes before/after.
3. Think about how a skilled voice actor would perform each line.
4. If a speaker tag is present (e.g., "CLAUDE:", "GROK:"), factor the character's personality into the emotion choice.

Output format — return valid JSON array:
[
  {"line": 1, "text": "the original text", "emotion": "sarcastic", "note": "brief reason"},
  ...
]

Return ONLY the JSON array. No other text."""


def detect_emotions(text: str) -> list:
    """
    Use LLM to tag each line with emotional register.
    Returns list of dicts with line, text, emotion, and note.
    """
    client = OpenAI()
    print("[Reverse L1] Detecting emotions per line...")

    # Split into lines, skip empties
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # Build numbered text for the LLM
    numbered = "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(lines))

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EMOTION_DETECTION_PROMPT},
            {"role": "user", "content": numbered},
        ],
        temperature=0.2,
        max_tokens=4000,
    )
    elapsed = time.time() - start
    print(f"[Reverse L1] Emotion detection complete in {elapsed:.1f}s")

    # Parse JSON response
    raw = response.choices[0].message.content.strip()
    # Handle markdown code blocks
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        tagged = json.loads(raw)
    except json.JSONDecodeError:
        tagged = []
    # Models supply annotations, never replacement source text or ordering.
    by_line = {}
    if isinstance(tagged, list):
        for item in tagged:
            if isinstance(item, dict) and type(item.get("line")) is int:
                line = item["line"]
                if line in by_line:
                    raise ValueError("Emotion response contains duplicate line IDs")
                if not 1 <= line <= len(lines):
                    raise ValueError("Emotion response contains an unknown line ID")
                by_line[line] = item
    result = []
    for number, original in enumerate(lines, 1):
        item = by_line.get(number, {})
        emotion = item.get("emotion")
        valid = isinstance(emotion, str) and emotion.lower() in EMOTION_PARAMS
        result.append({"line": number, "text": original,
                       "emotion": emotion.lower() if valid else "neutral",
                       "note": str(item.get("note", "")) if valid else "annotation unavailable; neutral fallback",
                       "annotation_status": "ok" if valid else "fallback"})
    return result


# ──────────────────────────────────────────────────────────────
# STEP 2: Parameter Mapping
# ──────────────────────────────────────────────────────────────

def map_emotions_to_params(tagged_lines: list) -> list:
    """
    Map detected emotions to ElevenLabs TTS parameters.
    Each line gets stability, style, and speed values.
    """
    print("[Reverse L2] Mapping emotions to TTS parameters...")
    mapped = []

    for index, item in enumerate(tagged_lines):
        emotion = item.get("emotion", "neutral").lower()
        if emotion not in EMOTION_PARAMS:
            emotion = "neutral"

        params = EMOTION_PARAMS[emotion]
        mapped.append({
            "line": item.get("line", index + 1),
            "annotation_status": item.get("annotation_status", "provided"),
            "text": item["text"],
            "emotion": emotion,
            "note": item.get("note", ""),
            "tts_params": {
                "stability": params["stability"],
                "style": params["style"],
                "speed": params["speed"],
            },
            "delivery": params["description"],
        })

    return mapped


# ──────────────────────────────────────────────────────────────
# STEP 3: ElevenLabs TTS Generation
# ──────────────────────────────────────────────────────────────

def _detect_speaker(text: str) -> str | None:
    """
    Detect speaker tag at start of line (e.g., 'CLAUDE:', 'GPT:', 'Grok:').
    Returns lowercase voice key if found, None otherwise.
    """
    match = re.match(r"^(CLAUDE|GPT|GROK|GEMINI|KYRA|NARRATOR)\s*:", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def _strip_speaker_tag(text: str) -> str:
    """Remove speaker tag prefix from text for cleaner TTS."""
    return re.sub(r"^(CLAUDE|GPT|GROK|GEMINI|KYRA|NARRATOR)\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def generate_audio(mapped_lines: list, voice_key: str = "claude",
                   output_name: str = "reverse_output", multi_voice: bool = False,
                   crossfade_ms: int = 100) -> list:
    """Render into an isolated run and persist identity, failures and exact timing.

    Returns combined audio first, then clips. An incomplete run raises with its
    manifest path, and never publishes a partial production as a complete file.
    """
    import io
    from elevenlabs import ElevenLabs
    from pydub import AudioSegment
    if not mapped_lines:
        raise ValueError("No source lines to generate")
    if type(crossfade_ms) is not int or crossfade_ms < 0:
        raise ValueError("crossfade_ms must be a nonnegative integer")
    if voice_key not in VOICE_MAP:
        raise ValueError("Unknown voice")
    run_id = uuid4().hex
    directory = OUTPUT_DIR / "renders" / run_id
    directory.mkdir(parents=True)
    manifest_path = directory / "generation_manifest.json"
    model_id = "eleven_multilingual_v2"
    segments = []
    for index, item in enumerate(mapped_lines):
        speaker = _detect_speaker(item["text"]) if multi_voice else None
        config = VOICE_MAP.get(speaker, VOICE_MAP[voice_key])
        tts_text = _strip_speaker_tag(item["text"]) if speaker else item["text"]
        segments.append({"segment_id": f"{run_id}:{index:06d}", "source_index": index,
                         "source_line": item.get("line", index + 1), "text": item["text"],
                         "tts_text": tts_text, "emotion": item["emotion"],
                         "annotation_status": item.get("annotation_status", "provided"),
                         "tts_params": dict(item["tts_params"]),
                         "voice_settings": {**item["tts_params"], "similarity_boost": 0.75,
                                            "use_speaker_boost": True},
                         "voice_id": config["voice_id"], "model_id": model_id,
                         "status": "pending"})
    manifest = {"schema_version": 1, "run_id": run_id, "source_name": output_name,
                "complete": False, "segments": segments, "requested_crossfade_ms": crossfade_ms}
    write_json(directory / "emotion_map.json", mapped_lines)
    write_json(manifest_path, manifest)
    client = ElevenLabs()
    clips = []
    for index, segment in enumerate(segments):
        try:
            if not segment["tts_text"].strip():
                raise ValueError("Empty spoken line")
            chunks = client.text_to_speech.convert(voice_id=segment["voice_id"],
                text=segment["tts_text"], model_id=model_id, voice_settings=segment["voice_settings"])
            audio_bytes = b"".join(chunks)
            clip = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            if len(clip) == 0:
                raise ValueError("Empty generated audio")
            path = directory / f"segment_{index:06d}.mp3"
            path.write_bytes(audio_bytes)
            segment.update(status="ok", file=path.name, sha256=file_sha256(path), duration_ms=len(clip))
            clips.append(clip)
        except Exception as error:
            # Provider exception bodies may contain sensitive request details.
            segment.update(status="failed", error_type=type(error).__name__)
        write_json(manifest_path, manifest)
    if any(s["status"] != "ok" for s in segments):
        raise IncompleteGenerationError(manifest_path)
    combined = clips[0]
    segments[0].update(start_ms=0, end_ms=len(combined), crossfade_ms=0)
    for segment, clip in zip(segments[1:], clips[1:]):
        overlap = min(crossfade_ms, len(combined) // 2, len(clip) // 2)
        overlap = overlap if overlap > 10 else 0
        start_ms = len(combined) - overlap
        combined = combined.append(clip, crossfade=overlap)
        segment.update(start_ms=start_ms, end_ms=len(combined), crossfade_ms=overlap)
    audio_path = directory / "combined.mp3"
    combined.export(str(audio_path), format="mp3", bitrate="192k").close()
    manifest.update(complete=True, audio_file=audio_path.name,
                    audio_sha256=file_sha256(audio_path), duration_ms=len(combined))
    write_json(manifest_path, manifest)
    return [str(audio_path), *[str(directory / s["file"]) for s in segments]]


# ──────────────────────────────────────────────────────────────
# Full Reverse Pipeline
# ──────────────────────────────────────────────────────────────

def run_reverse_pipeline(text_path: str, voice: str = "claude",
                         emotion_map_only: bool = False, multi_voice: bool = False,
                         crossfade_ms: int = 100) -> dict:
    text_path = Path(text_path)
    text = text_path.read_text()
    tagged = detect_emotions(text)
    mapped = map_emotions_to_params(tagged)
    result = {"tagged_lines": tagged, "mapped_lines": mapped}
    if emotion_map_only:
        directory = OUTPUT_DIR / "annotations" / uuid4().hex
        directory.mkdir(parents=True)
        write_json(directory / "emotion_map.json", mapped)
    else:
        paths = generate_audio(mapped, voice_key=voice, output_name=text_path.name,
                               multi_voice=multi_voice, crossfade_ms=crossfade_ms)
        directory = Path(paths[0]).parent
        result.update(audio_paths=paths, manifest_path=str(directory / "generation_manifest.json"))
    (directory / "input.txt").write_text(text)
    result["emotion_map_path"] = str(directory / "emotion_map.json")
    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prosody Intelligence — Reverse Pipeline")
    parser.add_argument("text_file", help="Path to text file")
    parser.add_argument("--voice", default="claude",
                        choices=list(VOICE_MAP.keys()),
                        help="Voice to use for TTS")
    parser.add_argument("--emotion-map", action="store_true",
                        help="Only generate emotion map (no TTS)")
    parser.add_argument("--multi-voice", action="store_true",
                        help="Detect speaker tags and route to assigned voices")
    parser.add_argument("--crossfade", type=int, default=100,
                        help="Crossfade overlap in ms between segments (default: 100)")

    args = parser.parse_args()
    result = run_reverse_pipeline(
        args.text_file,
        voice=args.voice,
        emotion_map_only=args.emotion_map,
        multi_voice=args.multi_voice,
        crossfade_ms=args.crossfade,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
