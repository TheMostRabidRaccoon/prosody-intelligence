"""
Prosody Intelligence — Reverse Pipeline (Layer 4)
Rabid Raccoon Intelligence, LLC

Text → Emotion Detection → Prosody Parameter Mapping → ElevenLabs TTS → Audio

This is both the demo layer and the proof that the system works.
Feed it text, it detects emotional register per line, maps to
ElevenLabs voice parameters, and generates emotionally accurate audio.

Usage:
    python3 reverse_pipeline.py input.txt --voice claude
    python3 reverse_pipeline.py input.txt --voice grok --emotion-map   # show emotion map without TTS
    python3 reverse_pipeline.py input.txt --voice all                  # multi-voice performance

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
                   output_name: str = "reverse_output",
                   multi_voice: bool = False,
                   crossfade_ms: int = 100) -> list:
    """
    Generate audio for each line using ElevenLabs with emotion-mapped parameters.

    Improvements over v1:
      - multi_voice=True: detect speaker tags and route to correct voice
      - pydub crossfade stitching (default 100ms) eliminates choppy concatenation
      - Individual segments still saved for debugging

    Returns list of output file paths (combined first, then segments).
    """
    from elevenlabs import ElevenLabs
    from pydub import AudioSegment
    import io

    client = ElevenLabs()

    # Default voice for non-tagged lines
    default_config = VOICE_MAP.get(voice_key, VOICE_MAP["claude"])
    default_voice_id = default_config["voice_id"]

    if multi_voice:
        print(f"[Reverse L3] Multi-voice mode — routing by speaker tags")
        print(f"  Default voice: {default_config['name']}")
        for k, v in VOICE_MAP.items():
            print(f"  {k:8s} → {v['name']}")
    else:
        print(f"[Reverse L3] Single voice: {default_config['name']}")

    audio_segments = []  # pydub AudioSegment objects
    output_paths = []

    for i, item in enumerate(mapped_lines):
        text = item["text"]
        params = item["tts_params"]

        if not text.strip():
            continue

        # Determine voice for this line
        if multi_voice:
            speaker = _detect_speaker(text)
            if speaker and speaker in VOICE_MAP:
                voice_id = VOICE_MAP[speaker]["voice_id"]
                voice_label = VOICE_MAP[speaker]["name"]
                # Strip the tag so TTS doesn't say "CLAUDE:"
                tts_text = _strip_speaker_tag(text)
            else:
                voice_id = default_voice_id
                voice_label = default_config["name"]
                tts_text = text
        else:
            voice_id = default_voice_id
            voice_label = default_config["name"]
            tts_text = text

        if not tts_text.strip():
            continue

        emotion_tag = item["emotion"]
        print(f"  [{i+1}/{len(mapped_lines)}] {emotion_tag:12s} | {voice_label:20s} | {tts_text[:45]}...")

        try:
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=tts_text,
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

            # Convert to pydub AudioSegment for crossfade stitching
            seg_audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            audio_segments.append(seg_audio)

        except Exception as e:
            print(f"  [ERROR] Segment {i}: {e}")
            continue

    # ── Crossfade stitch all segments ──
    if audio_segments:
        print(f"\n[Reverse L3] Stitching {len(audio_segments)} segments with {crossfade_ms}ms crossfade...")

        combined = audio_segments[0]
        for seg in audio_segments[1:]:
            # Crossfade overlap — eliminates the choppy inter-segment gaps
            # Use min of crossfade_ms and half the shorter segment to avoid artifacts
            safe_fade = min(crossfade_ms, len(combined) // 2, len(seg) // 2)
            if safe_fade > 10:  # only crossfade if segments are long enough
                combined = combined.append(seg, crossfade=safe_fade)
            else:
                combined = combined + seg  # fallback to simple concat for tiny segments

        combined_path = OUTPUT_DIR / f"{output_name}_full.mp3"
        combined.export(str(combined_path), format="mp3", bitrate="192k")
        output_paths.insert(0, str(combined_path))

        duration_s = len(combined) / 1000.0
        print(f"[Reverse L3] Combined audio: {combined_path} ({duration_s:.1f}s)")

    return output_paths


# ──────────────────────────────────────────────────────────────
# Full Reverse Pipeline
# ──────────────────────────────────────────────────────────────

def run_reverse_pipeline(text_path: str, voice: str = "claude",
                         emotion_map_only: bool = False,
                         multi_voice: bool = False,
                         crossfade_ms: int = 100) -> dict:
    """
    Full reverse pipeline: text → emotion → params → audio.

    Args:
        multi_voice: If True, detect speaker tags (CLAUDE:, GPT:, etc.) and
                     route each line to the assigned ElevenLabs voice.
        crossfade_ms: Overlap in ms between segments for smooth stitching.
    """
    text_path = Path(text_path)
    text = text_path.read_text()
    output_name = text_path.stem

    # Auto-save input transcript for future editing
    saved_input = OUTPUT_DIR / f"{output_name}_input.txt"
    if not saved_input.exists():
        saved_input.write_text(text)
        print(f"[Reverse] Input transcript saved: {saved_input}")

    mode = "MULTI-VOICE" if multi_voice else f"SINGLE ({VOICE_MAP.get(voice, {}).get('name', voice)})"
    print("=" * 60)
    print("PROSODY INTELLIGENCE — Reverse Pipeline")
    print(f"Input: {text_path.name}")
    print(f"Mode:  {mode}")
    print(f"Crossfade: {crossfade_ms}ms")
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
        print(f"  {emotion:12s} [{delivery:30s}] {text_preview}")
    print("─" * 50)

    result = {
        "tagged_lines": tagged,
        "mapped_lines": mapped,
        "emotion_map_path": str(map_path),
    }

    # Step 3: Generate audio (unless map-only)
    if not emotion_map_only:
        paths = generate_audio(
            mapped,
            voice_key=voice,
            output_name=output_name,
            multi_voice=multi_voice,
            crossfade_ms=crossfade_ms,
        )
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
    parser.add_argument("--multi-voice", action="store_true",
                        help="Detect speaker tags and route to assigned voices")
    parser.add_argument("--crossfade", type=int, default=100,
                        help="Crossfade overlap in ms between segments (default: 100)")

    args = parser.parse_args()
    run_reverse_pipeline(
        args.text_file,
        voice=args.voice,
        emotion_map_only=args.emotion_map,
        multi_voice=args.multi_voice,
        crossfade_ms=args.crossfade,
    )
