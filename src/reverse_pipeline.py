"""
Prosody Intelligence — Reverse Pipeline (Layer 4)
Rabid Raccoon Intelligence, LLC

Text → Emotion Detection → Prosody Parameter Mapping → ElevenLabs TTS → Audio

This is both the demo layer and the proof that the system works.
Feed it text, it detects emotional register per line, maps to
ElevenLabs voice parameters, and generates emotionally accurate audio.

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

from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(Path.home() / ".env")

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
}

# ──────────────────────────────────────────────────────────────
# Emotion → ElevenLabs Parameter Mapping (from spec Section 6.2)
# ──────────────────────────────────────────────────────────────

EMOTION_PARAMS = {
    "sarcastic": {
        "stability": 0.3,
        "style": 0.8,
        "speed": 0.9,
        "description": "Deliberate, dry delivery",
    },
    "urgent": {
        "stability": 0.5,
        "style": 0.7,
        "speed": 1.2,
        "description": "Pressured, clipped",
    },
    "confident": {
        "stability": 0.6,
        "style": 0.6,
        "speed": 1.0,
        "description": "Steady, authoritative",
    },
    "comedic": {
        "stability": 0.2,
        "style": 0.9,
        "speed": 1.05,
        "description": "Expressive, timing-driven",
    },
    "dramatic": {
        "stability": 0.4,
        "style": 0.9,
        "speed": 0.8,
        "description": "Theatrical, weighted",
    },
    "analytical": {
        "stability": 0.7,
        "style": 0.3,
        "speed": 1.0,
        "description": "Clinical, precise",
    },
    "hesitant": {
        "stability": 0.3,
        "style": 0.5,
        "speed": 0.85,
        "description": "Uncertain, trailing off",
    },
    "angry": {
        "stability": 0.4,
        "style": 0.8,
        "speed": 1.15,
        "description": "Intense, forceful",
    },
    "tender": {
        "stability": 0.6,
        "style": 0.7,
        "speed": 0.85,
        "description": "Warm, gentle, vulnerable",
    },
    "resigned": {
        "stability": 0.7,
        "style": 0.3,
        "speed": 0.8,
        "description": "Flat, drained, defeated",
    },
    "excited": {
        "stability": 0.3,
        "style": 0.8,
        "speed": 1.2,
        "description": "High energy, enthusiastic",
    },
    "neutral": {
        "stability": 0.5,
        "style": 0.5,
        "speed": 1.0,
        "description": "Baseline conversational",
    },
}


# ──────────────────────────────────────────────────────────────
# STEP 1: Emotion Detection (LLM-native)
# ──────────────────────────────────────────────────────────────

EMOTION_DETECTION_PROMPT = """You are an emotion tagger for a text-to-speech system. For each line of text, identify the emotional register that should drive the vocal performance.

Available emotions: sarcastic, urgent, confident, comedic, dramatic, analytical, hesitant, angry, tender, resigned, excited, neutral

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
        print(f"[Reverse L1] Warning: Failed to parse LLM JSON. Raw:\n{raw[:200]}")
        # Fallback: tag everything as neutral
        tagged = [{"line": i+1, "text": l, "emotion": "neutral", "note": "parse failure fallback"}
                  for i, l in enumerate(lines)]

    return tagged


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

    for item in tagged_lines:
        emotion = item.get("emotion", "neutral").lower()
        if emotion not in EMOTION_PARAMS:
            emotion = "neutral"

        params = EMOTION_PARAMS[emotion]
        mapped.append({
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

def generate_audio(mapped_lines: list, voice_key: str = "claude",
                   output_name: str = "reverse_output") -> list:
    """
    Generate audio for each line using ElevenLabs with emotion-mapped parameters.
    Returns list of output file paths.
    """
    from elevenlabs import ElevenLabs

    client = ElevenLabs()
    voice_config = VOICE_MAP.get(voice_key, VOICE_MAP["claude"])
    voice_id = voice_config["voice_id"]

    print(f"[Reverse L3] Generating TTS with voice: {voice_config['name']}")

    audio_parts = []
    output_paths = []

    for i, item in enumerate(mapped_lines):
        text = item["text"]
        params = item["tts_params"]

        if not text.strip():
            continue

        print(f"  [{i+1}/{len(mapped_lines)}] {item['emotion']:12s} | {text[:50]}...")

        try:
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings={
                    "stability": params["stability"],
                    "similarity_boost": 0.75,
                    "style": params["style"],
                    "use_speaker_boost": True,
                    "speed": params["speed"],
                },
            )

            # Collect audio bytes
            audio_bytes = b""
            for chunk in audio:
                audio_bytes += chunk

            # Save individual segment
            seg_path = OUTPUT_DIR / f"{output_name}_seg{i:03d}.mp3"
            with open(seg_path, "wb") as f:
                f.write(audio_bytes)
            output_paths.append(str(seg_path))
            audio_parts.append(audio_bytes)

        except Exception as e:
            print(f"  [ERROR] Segment {i}: {e}")
            continue

    # Concatenate all segments into one file
    if audio_parts:
        combined_path = OUTPUT_DIR / f"{output_name}_full.mp3"
        with open(combined_path, "wb") as f:
            for part in audio_parts:
                f.write(part)
        output_paths.insert(0, str(combined_path))
        print(f"\n[Reverse L3] Combined audio: {combined_path}")

    return output_paths


# ──────────────────────────────────────────────────────────────
# Full Reverse Pipeline
# ──────────────────────────────────────────────────────────────

def run_reverse_pipeline(text_path: str, voice: str = "claude",
                         emotion_map_only: bool = False) -> dict:
    """
    Full reverse pipeline: text → emotion → params → audio.
    """
    text_path = Path(text_path)
    text = text_path.read_text()
    output_name = text_path.stem

    print("=" * 60)
    print("PROSODY INTELLIGENCE — Reverse Pipeline")
    print(f"Input: {text_path.name}")
    print(f"Voice: {VOICE_MAP.get(voice, {}).get('name', voice)}")
    print("=" * 60)

    # Step 1: Detect emotions
    tagged = detect_emotions(text)

    # Step 2: Map to TTS params
    mapped = map_emotions_to_params(tagged)

    # Save emotion map
    map_path = OUTPUT_DIR / f"{output_name}_emotion_map.json"
    with open(map_path, "w") as f:
        json.dump(mapped, f, indent=2)
    print(f"\n[Output] Emotion map saved: {map_path}")

    # Print emotion map
    print("\n" + "─" * 50)
    print("EMOTION MAP")
    print("─" * 50)
    for item in mapped:
        emotion = item["emotion"]
        delivery = item["delivery"]
        text_preview = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
        print(f"  {emotion:12s} [{delivery:25s}] {text_preview}")
    print("─" * 50)

    result = {
        "tagged_lines": tagged,
        "mapped_lines": mapped,
        "emotion_map_path": str(map_path),
    }

    # Step 3: Generate audio (unless map-only)
    if not emotion_map_only:
        paths = generate_audio(mapped, voice_key=voice, output_name=output_name)
        result["audio_paths"] = paths

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

    args = parser.parse_args()
    run_reverse_pipeline(args.text_file, voice=args.voice, emotion_map_only=args.emotion_map)
