"""
Prosody Intelligence — Forward Pipeline
Rabid Raccoon Intelligence, LLC

Audio → Whisper Transcription → Parselmouth Prosody Extraction →
Alignment → Annotated Transcript → LLM Analysis → Deep Insight

Usage:
    python prosody_pipeline.py /path/to/audio.m4a
    python prosody_pipeline.py /path/to/audio.m4a --no-llm       # skip LLM analysis
    python prosody_pipeline.py /path/to/audio.m4a --text-only     # run LLM without prosody (for A/B comparison)
    python prosody_pipeline.py /path/to/audio.m4a --visualize     # generate prosody visualization PNG
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Flask/threads

import numpy as np
import parselmouth
from parselmouth.praat import call
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(Path.home() / ".env")


# ──────────────────────────────────────────────────────────────
# LAYER 1A: Whisper Transcription
# ──────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.
    Returns word-level timestamps and segment data.
    """
    client = OpenAI()
    audio_path = Path(audio_path)

    print(f"[Layer 1A] Transcribing: {audio_path.name}")
    start = time.time()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    elapsed = time.time() - start
    print(f"[Layer 1A] Transcription complete in {elapsed:.1f}s")
    print(f"[Layer 1A] Text: {response.text[:120]}...")

    result = {
        "text": response.text,
        "segments": [],
        "words": [],
    }

    if response.segments:
        for seg in response.segments:
            result["segments"].append({
                "id": getattr(seg, "id", 0),
                "start": getattr(seg, "start", 0),
                "end": getattr(seg, "end", 0),
                "text": getattr(seg, "text", "").strip(),
            })

    if response.words:
        for w in response.words:
            result["words"].append({
                "word": getattr(w, "word", "").strip(),
                "start": getattr(w, "start", 0),
                "end": getattr(w, "end", 0),
            })

    return result


# ──────────────────────────────────────────────────────────────
# LAYER 1B: Parselmouth Prosody Extraction
# ──────────────────────────────────────────────────────────────

def ensure_wav(audio_path: str) -> str:
    """
    Parselmouth/Praat needs WAV format. If the input isn't WAV,
    convert it via ffmpeg. Returns path to a WAV file.
    """
    p = Path(audio_path)
    if p.suffix.lower() == ".wav":
        return str(p)

    wav_path = OUTPUT_DIR / f"{p.stem}_converted.wav"
    if wav_path.exists():
        return str(wav_path)

    print(f"[Layer 1B] Converting {p.suffix} → WAV via ffmpeg...")
    subprocess.run(
        ["ffmpeg", "-i", str(p), "-ar", "16000", "-ac", "1", "-y", str(wav_path)],
        capture_output=True, check=True,
    )
    return str(wav_path)


def extract_prosody(audio_path: str) -> dict:
    """
    Extract full-file prosody data using Parselmouth (Praat).
    Returns the Sound object, Pitch object, and Intensity object
    for segment-level querying in the alignment layer.
    """
    print(f"[Layer 1B] Extracting prosody from: {Path(audio_path).name}")
    start = time.time()

    wav_path = ensure_wav(audio_path)
    sound = parselmouth.Sound(wav_path)
    duration = sound.get_total_duration()

    # Pitch extraction — Praat's autocorrelation method, tuned for speech
    pitch = call(sound, "To Pitch", 0.0, 75, 600)

    # Intensity (energy/volume)
    intensity = call(sound, "To Intensity", 75, 0.0, "yes")

    elapsed = time.time() - start
    print(f"[Layer 1B] Prosody extraction complete in {elapsed:.1f}s")
    print(f"[Layer 1B] Audio duration: {duration:.1f}s")

    return {
        "sound": sound,
        "pitch": pitch,
        "intensity": intensity,
        "duration": duration,
    }


def get_segment_prosody(prosody_data: dict, start: float, end: float) -> dict:
    """
    Query prosody features for a specific time segment.
    This is where Parselmouth earns its keep over librosa —
    Praat's pitch tracking is built for human speech.
    """
    pitch = prosody_data["pitch"]
    intensity = prosody_data["intensity"]
    sound = prosody_data["sound"]

    # Protect against zero-length segments
    if end <= start:
        end = start + 0.01

    # --- Pitch (F0) ---
    pitch_values = []
    time_step = 0.01  # 10ms steps
    t = start
    while t <= end:
        f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
        if f0 and not np.isnan(f0):
            pitch_values.append(f0)
        t += time_step

    avg_pitch = float(np.mean(pitch_values)) if pitch_values else 0.0
    pitch_variance = float(np.std(pitch_values)) if len(pitch_values) > 1 else 0.0

    # Pitch direction: compare first third vs last third of segment
    if len(pitch_values) >= 6:
        third = len(pitch_values) // 3
        first_third = np.mean(pitch_values[:third])
        last_third = np.mean(pitch_values[-third:])
        diff = last_third - first_third
        if diff > 5:
            pitch_direction = "rising"
        elif diff < -5:
            pitch_direction = "falling"
        else:
            pitch_direction = "flat"
    elif len(pitch_values) >= 2:
        diff = pitch_values[-1] - pitch_values[0]
        pitch_direction = "rising" if diff > 5 else ("falling" if diff < -5 else "flat")
    else:
        pitch_direction = "unknown"

    # --- Energy (RMS/Intensity) ---
    energy_values = []
    t = start
    while t <= end:
        e = call(intensity, "Get value at time", t, "Cubic")
        if e and not np.isnan(e):
            energy_values.append(e)
        t += time_step

    avg_energy = float(np.mean(energy_values)) if energy_values else 0.0
    # Normalize energy to 0-1 range (typical speech: 40-80 dB)
    energy_normalized = max(0.0, min(1.0, (avg_energy - 40) / 40))

    # --- Speaking Rate (syllables per second estimate) ---
    # Approximate: use intensity dips as syllable boundaries
    # More accurate than word count / duration for natural speech
    segment_duration = end - start
    if segment_duration > 0 and energy_values:
        # Count energy peaks as rough syllable proxy
        arr = np.array(energy_values)
        mean_e = np.mean(arr)
        crossings = np.diff(np.sign(arr - mean_e))
        peaks = np.sum(crossings < 0)  # downward crossings ~ syllable boundaries
        speaking_rate = round(peaks / segment_duration, 1)
    else:
        speaking_rate = 0.0

    return {
        "avg_pitch": round(avg_pitch, 1),
        "pitch_direction": pitch_direction,
        "pitch_variance": round(pitch_variance, 1),
        "energy": round(energy_normalized, 2),
        "speaking_rate": speaking_rate,
    }


# ──────────────────────────────────────────────────────────────
# LAYER 2: Alignment & Annotation
# ──────────────────────────────────────────────────────────────

def align_transcript_with_prosody(transcript: dict, prosody_data: dict) -> list:
    """
    Merge Whisper timestamps with Parselmouth prosody data.
    Calculates pause_before and pause_after for each segment.
    This is the glue layer — every segment gets both its text
    and its acoustic signature.
    """
    print("[Layer 2] Aligning transcript with prosody data...")
    start_time = time.time()

    segments = transcript["segments"]
    annotated = []

    for i, seg in enumerate(segments):
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Pause detection — the analytical gold
        if i == 0:
            pause_before = seg_start  # silence before first utterance
        else:
            prev_end = segments[i - 1]["end"]
            pause_before = max(0, seg_start - prev_end)

        if i < len(segments) - 1:
            next_start = segments[i + 1]["start"]
            pause_after = max(0, next_start - seg_end)
        else:
            pause_after = 0.0

        # Get prosody features for this segment
        prosody = get_segment_prosody(prosody_data, seg_start, seg_end)

        # Attach pause data
        prosody["pause_before"] = round(pause_before, 2)
        prosody["pause_after"] = round(pause_after, 2)

        annotated.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
            "prosody": prosody,
        })

    elapsed = time.time() - start_time
    print(f"[Layer 2] Alignment complete in {elapsed:.1f}s — {len(annotated)} segments annotated")

    return annotated


# ──────────────────────────────────────────────────────────────
# LAYER 3: LLM Analysis
# ──────────────────────────────────────────────────────────────

PROSODY_SYSTEM_PROMPT = """You are a prosody-aware communication analyst. You receive transcripts annotated with acoustic prosody data extracted from the original audio. Your job is to analyze not just WHAT was said, but HOW it was said — and what that reveals about the speaker's true emotional state, intent, and meaning.

## Prosody Feature Guide

Each segment includes these acoustic measurements:

- **avg_pitch** (Hz): Fundamental frequency. Higher = excitement, stress, questions. Lower = certainty, calm, authority. Typical ranges: male 85-180 Hz, female 165-255 Hz.
- **pitch_direction**: Rising, falling, or flat contour over the segment.
  - Rising on statements → uncertainty, seeking validation, turning statement into question
  - Falling → certainty, finality, declarative confidence
  - Flat → controlled, guarded, rehearsed, or monotone delivery
- **pitch_variance**: How much pitch moves within the segment.
  - High variance → emotional, expressive, engaged
  - Low variance → controlled, guarded, rehearsed, flat affect
- **energy** (0-1 normalized): Volume/intensity.
  - High → emphasis, engagement, arousal, conviction
  - Low → fatigue, disengagement, resignation, or deliberate quiet for effect
- **speaking_rate** (syllables/sec): Typical conversational speech is 3-5 syl/sec.
  - Fast (>5) → excitement, anxiety, rushing through uncomfortable material
  - Slow (<3) → deliberation, fatigue, emphasis, or emotional weight
- **pause_before** (seconds): Silence before this segment.
  - Long pause (>0.8s) before response → hesitation, careful thought, discomfort, internal conflict
  - No pause → immediate/automatic response, rehearsed, or interruption
- **pause_after** (seconds): Silence after this segment.
  - Long pause → expecting response, dramatic weight, or topic shift
  - Short/no pause → continuing thought, or other speaker jumped in

## Analysis Instructions

1. Read both the text AND the prosody data for each segment.
2. Flag moments where prosody contradicts or adds nuance to the text (e.g., "I'm fine" said with falling energy and long preceding pause).
3. Identify emotional shifts, hesitation patterns, and moments of emphasis.
4. Note power dynamics reflected in speaking patterns (who speaks louder, faster, who pauses more).
5. Provide a prosodic summary that captures the emotional arc of the conversation.
6. Be specific — cite the actual numbers when they tell a story.

Your analysis should reveal what a text-only reading would miss."""


def build_annotated_prompt(annotated_segments: list) -> str:
    """Format annotated segments for LLM consumption."""
    lines = ["# Annotated Transcript (Text + Prosody)\n"]

    for seg in annotated_segments:
        p = seg["prosody"]
        lines.append(f'[{seg["start"]:.1f}s - {seg["end"]:.1f}s]')
        lines.append(f'Text: "{seg["text"]}"')
        lines.append(
            f'Prosody: pitch={p["avg_pitch"]}Hz ({p["pitch_direction"]}), '
            f'variance={p["pitch_variance"]}, energy={p["energy"]}, '
            f'rate={p["speaking_rate"]} syl/s, '
            f'pause_before={p["pause_before"]}s, pause_after={p["pause_after"]}s'
        )
        lines.append("")

    return "\n".join(lines)


def analyze_with_llm(annotated_segments: list, text_only: bool = False) -> str:
    """
    Run LLM analysis on the transcript.
    If text_only=True, strips prosody data for the A/B comparison test.
    """
    client = OpenAI()
    mode = "TEXT ONLY" if text_only else "TEXT + PROSODY"
    print(f"[Layer 3] Running LLM analysis ({mode})...")

    if text_only:
        # A/B test: text-only version
        prompt = "# Transcript (Text Only)\n\n"
        for seg in annotated_segments:
            prompt += f'[{seg["start"]:.1f}s - {seg["end"]:.1f}s] "{seg["text"]}"\n'
        prompt += "\nAnalyze this conversation. What emotions, dynamics, and subtext do you detect?"
        system = "You are an expert communication analyst. Analyze the transcript for emotional dynamics, subtext, and interpersonal patterns."
    else:
        # Full prosody-annotated version
        prompt = build_annotated_prompt(annotated_segments)
        prompt += "\nProvide a deep prosodic analysis. What does the voice reveal that the words alone don't?"
        system = PROSODY_SYSTEM_PROMPT

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    elapsed = time.time() - start
    print(f"[Layer 3] Analysis complete in {elapsed:.1f}s")

    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────

def visualize_prosody(annotated_segments: list, prosody_data: dict, audio_name: str) -> str:
    """
    Generate a multi-panel prosody visualization.

    Panel 1: Waveform with segment boundaries and silence craters
    Panel 2: Pitch (F0) contour — continuous, color-coded by segment
    Panel 3: Energy + speaking rate per segment (bar chart)

    Returns path to saved PNG.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection

    print("[Viz] Generating prosody visualization...")

    sound = prosody_data["sound"]
    pitch_obj = prosody_data["pitch"]
    intensity_obj = prosody_data["intensity"]
    duration = prosody_data["duration"]

    # --- Extract continuous data ---
    # Waveform
    waveform = sound.values[0]
    wave_times = np.linspace(0, duration, len(waveform))

    # Continuous pitch
    pitch_times = np.arange(0, duration, 0.01)
    pitch_vals = []
    for t in pitch_times:
        f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        pitch_vals.append(f0 if f0 and not np.isnan(f0) else np.nan)
    pitch_vals = np.array(pitch_vals)

    # Continuous intensity
    int_vals = []
    for t in pitch_times:
        e = call(intensity_obj, "Get value at time", t, "Cubic")
        int_vals.append(e if e and not np.isnan(e) else np.nan)
    int_vals = np.array(int_vals)

    # --- Color palette ---
    # Alternating segment colors — warm vs cool to distinguish speakers
    seg_colors = []
    for i, seg in enumerate(annotated_segments):
        p = seg["prosody"]["avg_pitch"]
        if p > 170:  # likely higher-pitched speaker
            seg_colors.append("#E8594F")  # RRI red
        else:
            seg_colors.append("#4A90D9")  # cool blue
    silence_color = "#2D2D2D"

    # --- Figure setup ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.patch.set_facecolor("#1A1A1A")
    for ax in axes:
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="#AAAAAA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # ── Panel 1: Waveform ──
    ax1 = axes[0]
    ax1.plot(wave_times, waveform, color="#666666", linewidth=0.3, alpha=0.6)
    ax1.set_ylabel("Amplitude", color="#AAAAAA", fontsize=9)
    ax1.set_title(f"PROSODY INTELLIGENCE  —  {audio_name}",
                  color="#E8594F", fontsize=14, fontweight="bold", pad=12)

    # Color segments on waveform
    for i, seg in enumerate(annotated_segments):
        mask = (wave_times >= seg["start"]) & (wave_times <= seg["end"])
        ax1.fill_between(wave_times, waveform, where=mask,
                         color=seg_colors[i], alpha=0.35)

    # Mark silence craters
    for i, seg in enumerate(annotated_segments):
        pb = seg["prosody"]["pause_before"]
        if pb > 1.0:
            crater_start = seg["start"] - pb
            crater_end = seg["start"]
            ax1.axvspan(crater_start, crater_end, color=silence_color, alpha=0.7)
            ax1.text((crater_start + crater_end) / 2, ax1.get_ylim()[1] * 0.85,
                     f"{pb:.1f}s\nsilence",
                     ha="center", va="top", color="#FF6B6B", fontsize=7,
                     fontweight="bold", style="italic")

    # Segment text labels (truncated)
    for i, seg in enumerate(annotated_segments):
        text = seg["text"][:40] + ("..." if len(seg["text"]) > 40 else "")
        mid = (seg["start"] + seg["end"]) / 2
        y_pos = ax1.get_ylim()[0] * 0.8  # below center
        ax1.text(mid, y_pos, text, ha="center", va="center",
                 color="#DDDDDD", fontsize=5.5, rotation=0, alpha=0.9,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333",
                          edgecolor="none", alpha=0.7))

    # ── Panel 2: Pitch Contour ──
    ax2 = axes[1]
    ax2.set_ylabel("Pitch (Hz)", color="#AAAAAA", fontsize=9)

    # Plot pitch as colored segments
    for i, seg in enumerate(annotated_segments):
        mask = (pitch_times >= seg["start"]) & (pitch_times <= seg["end"])
        seg_times = pitch_times[mask]
        seg_pitch = pitch_vals[mask]
        valid = ~np.isnan(seg_pitch)
        if np.any(valid):
            ax2.plot(seg_times[valid], seg_pitch[valid],
                     color=seg_colors[i], linewidth=1.8, alpha=0.9)
            # Mark avg pitch as horizontal line
            avg_p = seg["prosody"]["avg_pitch"]
            if avg_p > 0:
                ax2.hlines(avg_p, seg["start"], seg["end"],
                          color=seg_colors[i], linewidth=0.8, linestyle="--", alpha=0.4)

    # Pitch direction arrows
    for i, seg in enumerate(annotated_segments):
        direction = seg["prosody"]["pitch_direction"]
        mid_t = (seg["start"] + seg["end"]) / 2
        avg_p = seg["prosody"]["avg_pitch"]
        if avg_p > 0 and direction != "unknown":
            arrow = "↗" if direction == "rising" else ("↘" if direction == "falling" else "→")
            ax2.text(mid_t, avg_p + 15, arrow, ha="center", va="bottom",
                     color=seg_colors[i], fontsize=12, fontweight="bold")

    # Mark silence craters on pitch panel too
    for i, seg in enumerate(annotated_segments):
        pb = seg["prosody"]["pause_before"]
        if pb > 1.0:
            ax2.axvspan(seg["start"] - pb, seg["start"],
                        color=silence_color, alpha=0.5)

    ax2.set_ylim(50, 350)

    # ── Panel 3: Energy & Rate Bars ──
    ax3 = axes[2]
    ax3.set_ylabel("Energy / Rate", color="#AAAAAA", fontsize=9)
    ax3.set_xlabel("Time (seconds)", color="#AAAAAA", fontsize=9)

    bar_width_factor = 0.9
    for i, seg in enumerate(annotated_segments):
        seg_dur = seg["end"] - seg["start"]
        energy = seg["prosody"]["energy"]
        rate = seg["prosody"]["speaking_rate"]

        # Energy bar
        ax3.barh(0.6, seg_dur * bar_width_factor, left=seg["start"],
                 height=0.35, color=seg_colors[i],
                 alpha=max(0.3, energy), edgecolor="none")
        ax3.text(seg["start"] + seg_dur / 2, 0.6,
                 f"E:{energy:.2f}", ha="center", va="center",
                 color="white", fontsize=6, fontweight="bold")

        # Rate bar
        rate_norm = min(1.0, rate / 8.0)  # normalize to 0-1
        ax3.barh(0.15, seg_dur * bar_width_factor, left=seg["start"],
                 height=0.35, color=seg_colors[i],
                 alpha=max(0.2, rate_norm * 0.8), edgecolor="none")
        ax3.text(seg["start"] + seg_dur / 2, 0.15,
                 f"R:{rate:.1f}", ha="center", va="center",
                 color="white", fontsize=6, fontweight="bold")

    ax3.set_ylim(-0.1, 1.0)
    ax3.set_yticks([0.15, 0.6])
    ax3.set_yticklabels(["Rate\n(syl/s)", "Energy\n(0-1)"], fontsize=7, color="#AAAAAA")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#E8594F", alpha=0.6, label="Speaker 1 (higher pitch)"),
        mpatches.Patch(facecolor="#4A90D9", alpha=0.6, label="Speaker 2 (lower pitch)"),
        mpatches.Patch(facecolor=silence_color, alpha=0.7, label="Silence (>1s)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right",
               fontsize=7, facecolor="#2A2A2A", edgecolor="#444444",
               labelcolor="#CCCCCC")

    # ── Finalize ──
    plt.xlim(0, duration)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{audio_name}_prosody_viz.png"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)

    print(f"[Viz] Saved: {out_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────
# Full Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────

def run_pipeline(audio_path: str, skip_llm: bool = False, text_only: bool = False, visualize: bool = False) -> dict:
    """
    Run the full forward pipeline on an audio file.

    Returns dict with all intermediate outputs for inspection.
    """
    audio_path = str(Path(audio_path).resolve())
    audio_name = Path(audio_path).stem

    print("=" * 60)
    print(f"PROSODY INTELLIGENCE — Forward Pipeline")
    print(f"Input: {Path(audio_path).name}")
    print("=" * 60)

    # Layer 1A: Transcribe
    transcript = transcribe_audio(audio_path)

    # Layer 1B: Extract prosody
    prosody_data = extract_prosody(audio_path)

    # Layer 2: Align
    annotated = align_transcript_with_prosody(transcript, prosody_data)

    # Save annotated transcript
    output_path = OUTPUT_DIR / f"{audio_name}_annotated.json"
    with open(output_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"\n[Output] Annotated transcript saved: {output_path}")

    result = {
        "audio_path": audio_path,
        "transcript": transcript,
        "annotated_segments": annotated,
        "output_path": str(output_path),
    }

    # Visualization
    if visualize:
        viz_path = visualize_prosody(annotated, prosody_data, audio_name)
        result["viz_path"] = viz_path

    # Layer 3: LLM Analysis
    if not skip_llm:
        analysis = analyze_with_llm(annotated, text_only=text_only)
        result["analysis"] = analysis

        # Save analysis
        suffix = "_text_only" if text_only else "_prosody"
        analysis_path = OUTPUT_DIR / f"{audio_name}_analysis{suffix}.txt"
        with open(analysis_path, "w") as f:
            f.write(analysis)
        print(f"[Output] Analysis saved: {analysis_path}")

        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        print(analysis)

    return result


def run_proof_test(audio_path: str) -> dict:
    """
    The Proof Test (Section 5.1 of the spec):
    Run the same transcript through LLM twice —
    once text-only, once with prosody — and show the delta.
    """
    audio_path = str(Path(audio_path).resolve())
    audio_name = Path(audio_path).stem

    print("=" * 60)
    print("PROSODY INTELLIGENCE — PROOF TEST (A/B Comparison)")
    print(f"Input: {Path(audio_path).name}")
    print("=" * 60)

    # Layer 1 + 2: Get annotated transcript
    transcript = transcribe_audio(audio_path)
    prosody_data = extract_prosody(audio_path)
    annotated = align_transcript_with_prosody(transcript, prosody_data)

    # A: Text-only analysis
    print("\n" + "—" * 40)
    print("PASS A: Text Only")
    print("—" * 40)
    text_analysis = analyze_with_llm(annotated, text_only=True)

    # B: Prosody-annotated analysis
    print("\n" + "—" * 40)
    print("PASS B: Text + Prosody")
    print("—" * 40)
    prosody_analysis = analyze_with_llm(annotated, text_only=False)

    # Save both
    for suffix, content in [("_A_text_only", text_analysis), ("_B_prosody", prosody_analysis)]:
        path = OUTPUT_DIR / f"{audio_name}_proof{suffix}.txt"
        with open(path, "w") as f:
            f.write(content)

    # Print comparison
    print("\n" + "=" * 60)
    print("PROOF TEST RESULTS")
    print("=" * 60)
    print("\n--- PASS A: TEXT ONLY ---\n")
    print(text_analysis)
    print("\n--- PASS B: TEXT + PROSODY ---\n")
    print(prosody_analysis)

    return {
        "text_analysis": text_analysis,
        "prosody_analysis": prosody_analysis,
        "annotated_segments": annotated,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prosody Intelligence — Forward Pipeline")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis (just extract and align)")
    parser.add_argument("--text-only", action="store_true", help="Run LLM without prosody data")
    parser.add_argument("--proof-test", action="store_true", help="Run A/B proof test (text vs prosody)")
    parser.add_argument("--visualize", action="store_true", help="Generate prosody visualization PNG")

    args = parser.parse_args()

    if args.proof_test:
        run_proof_test(args.audio)
    else:
        run_pipeline(args.audio, skip_llm=args.no_llm, text_only=args.text_only, visualize=args.visualize)
