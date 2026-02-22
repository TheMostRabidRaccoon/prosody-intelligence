"""
Prosody Intelligence — Layer 6: Animated Short Compositor
Rabid Raccoon Intelligence, LLC

Stills + Audio + Emotion Map → Animated Short (.mp4)

Takes a folder of sequenced stills (scene_00.png - scene_NN.png),
an emotion map JSON, and a combined audio track, then composites
them into an animated short with:
  - Per-scene image display timed to audio segments
  - Ken Burns pan/zoom effects per emotion type
  - Emotion-colored subtitle overlay
  - Crossfade transitions between scenes
  - Audio track composited on final render

Usage:
    python compositor.py --stills input/stills/ \\
                         --audio output/gemini_wifi_petition_full.mp3 \\
                         --emotion-map output/gemini_wifi_petition_emotion_map.json \\
                         --output output/gemini_wifi_petition_animated.mp4

    python compositor.py --stills input/stills/ \\
                         --audio output/gemini_wifi_petition_full.mp3 \\
                         --emotion-map output/gemini_wifi_petition_emotion_map.json \\
                         --output output/animated.mp4 \\
                         --resolution 1920x1080 \\
                         --transition-ms 500 \\
                         --subtitle-style full
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Emotion → visual style mapping
EMOTION_VISUALS = {
    # emotion: (hex_color, ken_burns_type, subtitle_bg_alpha)
    "dramatic":   ("#FF6B6B", "slow_zoom_in",   200),  # Red — theatrical
    "contempt":   ("#A78BFA", "slow_pan_right",  180),  # Purple — cold
    "sarcastic":  ("#FBBF24", "slight_zoom_out", 180),  # Gold — dry
    "comedic":    ("#34D399", "quick_zoom_in",   160),  # Green — playful
    "confident":  ("#60A5FA", "hold_steady",     180),  # Blue — authoritative
    "analytical": ("#94A3B8", "slow_pan_right",  160),  # Steel — clinical (L→R scan)
    "angry":      ("#EF4444", "shake",           220),  # Red hot
    "tender":     ("#F9A8D4", "slow_zoom_in",    140),  # Pink — gentle
    "resigned":   ("#6B7280", "slow_zoom_out",   140),  # Gray — fading
    "excited":    ("#FDE68A", "quick_zoom_in",   180),  # Yellow — energetic
    "hesitant":   ("#D1D5DB", "slight_drift",    140),  # Light gray — uncertain
    "urgent":     ("#F97316", "quick_pan_right",  200),  # Orange — pressured
    "disgusted":  ("#84CC16", "slow_zoom_out",   180),  # Lime — revulsion
    "neutral":    ("#9CA3AF", "hold_steady",     140),  # Gray — baseline
}

# Default for unknown emotions
DEFAULT_VISUAL = ("#FFFFFF", "hold_steady", 160)

FPS = 24
FONT_SIZE = 32
FONT_SIZE_EMOTION = 18
SUBTITLE_PADDING = 20
SUBTITLE_MARGIN_BOTTOM = 60


# ──────────────────────────────────────────────────────────────
# Ken Burns effects
# ──────────────────────────────────────────────────────────────

def apply_ken_burns(img_array: np.ndarray, effect: str, progress: float,
                    target_w: int, target_h: int) -> np.ndarray:
    """
    Apply Ken Burns pan/zoom effect to an image frame.

    Args:
        img_array: Source image as numpy array (H, W, 3)
        effect: Effect type from EMOTION_VISUALS
        progress: 0.0 to 1.0 through the scene
        target_w: Output width
        target_h: Output height

    Returns:
        Cropped/transformed numpy array at target resolution
    """
    h, w = img_array.shape[:2]

    # All effects work by defining a crop box that moves/zooms over time
    if effect == "slow_zoom_in":
        # Start at 100%, end at 115%
        scale = 1.0 + 0.15 * progress
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif effect == "slow_zoom_out":
        # Start at 115%, end at 100%
        scale = 1.15 - 0.15 * progress
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif effect == "quick_zoom_in":
        # Start at 100%, end at 125%
        scale = 1.0 + 0.25 * progress
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif effect == "slow_pan_right":
        # Pan from left to right
        scale = 1.1
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        max_x = w - crop_w
        x = int(max_x * progress)
        y = (h - crop_h) // 2

    elif effect == "slow_pan_left":
        # Pan from right to left
        scale = 1.1
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        max_x = w - crop_w
        x = int(max_x * (1.0 - progress))
        y = (h - crop_h) // 2

    elif effect == "quick_pan_right":
        scale = 1.1
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        max_x = w - crop_w
        x = int(max_x * progress)
        y = (h - crop_h) // 2

    elif effect == "slight_zoom_out":
        scale = 1.08 - 0.08 * progress
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    elif effect == "slight_drift":
        # Subtle diagonal drift
        scale = 1.05
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        max_x = w - crop_w
        max_y = h - crop_h
        x = int(max_x * 0.3 + max_x * 0.4 * progress)
        y = int(max_y * 0.3 + max_y * 0.4 * progress)

    elif effect == "shake":
        # Subtle camera shake for angry moments
        scale = 1.1
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        shake_x = int(8 * np.sin(progress * 20 * np.pi))
        shake_y = int(5 * np.cos(progress * 15 * np.pi))
        x = (w - crop_w) // 2 + shake_x
        y = (h - crop_h) // 2 + shake_y

    else:  # hold_steady
        scale = 1.05
        crop_w = int(target_w / scale)
        crop_h = int(target_h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

    # Clamp
    x = max(0, min(x, w - crop_w))
    y = max(0, min(y, h - crop_h))
    crop_w = min(crop_w, w - x)
    crop_h = min(crop_h, h - y)

    # Crop
    cropped = img_array[y:y+crop_h, x:x+crop_w]

    # Resize to target
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(cropped)
    pil_img = pil_img.resize((target_w, target_h), PILImage.LANCZOS)

    return np.array(pil_img)


# ──────────────────────────────────────────────────────────────
# Subtitle rendering
# ──────────────────────────────────────────────────────────────

def render_subtitle(frame: np.ndarray, text: str, emotion: str,
                    style: str = "full") -> np.ndarray:
    """
    Render emotion-colored subtitle onto a frame.

    Args:
        frame: numpy array (H, W, 3)
        text: Subtitle text (speaker tag stripped)
        emotion: Emotion tag for color
        style: "full" (text + emotion tag), "minimal" (text only), "none"

    Returns:
        Frame with subtitle overlay
    """
    if style == "none":
        return frame

    h, w = frame.shape[:2]
    color_hex, _, bg_alpha = EMOTION_VISUALS.get(emotion, DEFAULT_VISUAL)

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    # Try to find a good font
    font = None
    font_small = None
    font_paths = [
        "/System/Library/Fonts/SFProText-Semibold.otf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText-Semibold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, FONT_SIZE)
                font_small = ImageFont.truetype(fp, FONT_SIZE_EMOTION)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()
        font_small = font

    # Strip speaker tag for display
    display_text = re.sub(r"^(CLAUDE|GPT|GROK|GEMINI|KYRA)\s*:\s*", "", text,
                          flags=re.IGNORECASE).strip()

    # Word wrap
    max_chars = int(w / (FONT_SIZE * 0.55))  # rough estimate
    lines = []
    words = display_text.split()
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        if len(test) > max_chars:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    # Calculate subtitle box dimensions
    line_height = FONT_SIZE + 6
    emotion_height = FONT_SIZE_EMOTION + 8 if style == "full" else 0
    total_text_height = len(lines) * line_height + emotion_height
    box_height = total_text_height + SUBTITLE_PADDING * 2

    # Position at bottom
    box_y = h - box_height - SUBTITLE_MARGIN_BOTTOM
    box_x = int(w * 0.05)
    box_w = int(w * 0.9)

    # Draw semi-transparent background
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_height],
        radius=12,
        fill=(15, 15, 15, bg_alpha)
    )
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(pil_img)

    # Draw emotion tag
    if style == "full":
        # Parse hex color
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        tag_text = f"▌{emotion.upper()}"
        tag_y = box_y + SUBTITLE_PADDING
        draw.text(
            (box_x + SUBTITLE_PADDING, tag_y),
            tag_text,
            font=font_small,
            fill=(r, g, b)
        )
        text_start_y = tag_y + emotion_height
    else:
        text_start_y = box_y + SUBTITLE_PADDING

    # Draw text lines
    for i, line in enumerate(lines):
        y_pos = text_start_y + i * line_height
        # Center each line
        try:
            bbox = font.getbbox(line)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(line) * FONT_SIZE * 0.55
        x_pos = box_x + (box_w - text_w) / 2
        draw.text(
            (x_pos, y_pos),
            line,
            font=font,
            fill=(240, 240, 240)
        )

    return np.array(pil_img)


# ──────────────────────────────────────────────────────────────
# Crossfade between scenes
# ──────────────────────────────────────────────────────────────

def crossfade_frames(frame_a: np.ndarray, frame_b: np.ndarray,
                     alpha: float) -> np.ndarray:
    """Blend two frames. alpha=0 → all A, alpha=1 → all B."""
    return (frame_a * (1.0 - alpha) + frame_b * alpha).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# Main compositor
# ──────────────────────────────────────────────────────────────

def composite_animated_short(
    stills_dir: str,
    audio_path: str,
    emotion_map_path: str,
    output_path: str,
    resolution: tuple = (1920, 1080),
    transition_ms: int = 500,
    subtitle_style: str = "full",
    fps: int = 24,
):
    """
    Composite an animated short from stills + audio + emotion map.

    Args:
        stills_dir: Directory containing scene_00.png - scene_NN.png
        audio_path: Path to combined audio file (mp3)
        emotion_map_path: Path to emotion map JSON
        output_path: Output .mp4 path
        resolution: (width, height) tuple
        transition_ms: Crossfade duration between scenes in ms
        subtitle_style: "full", "minimal", or "none"
        fps: Frames per second
    """
    from moviepy import AudioFileClip, VideoClip

    target_w, target_h = resolution
    print("=" * 60)
    print("PROSODY INTELLIGENCE — Layer 6: Animated Short Compositor")
    print(f"Stills:     {stills_dir}")
    print(f"Audio:      {audio_path}")
    print(f"Emotion:    {emotion_map_path}")
    print(f"Output:     {output_path}")
    print(f"Resolution: {target_w}x{target_h}")
    print(f"Transition: {transition_ms}ms crossfade")
    print(f"Subtitles:  {subtitle_style}")
    print(f"FPS:        {fps}")
    print("=" * 60)

    # ── Load emotion map ──
    with open(emotion_map_path) as f:
        emotion_map = json.load(f)
    num_scenes = len(emotion_map)
    print(f"\n[L6] Loaded emotion map: {num_scenes} scenes")

    # ── Get audio segment durations ──
    # Look for individual segment files to get per-scene timing
    stills_path = Path(stills_dir)
    audio_dir = Path(audio_path).parent
    audio_stem = Path(audio_path).stem.replace("_full", "")

    segment_durations = []  # in seconds
    for i in range(num_scenes):
        seg_path = audio_dir / f"{audio_stem}_seg{i:03d}.mp3"
        if seg_path.exists():
            seg = AudioSegment.from_mp3(str(seg_path))
            segment_durations.append(len(seg) / 1000.0)
        else:
            # Fallback: divide total evenly
            total_audio = AudioSegment.from_mp3(audio_path)
            even_dur = len(total_audio) / 1000.0 / num_scenes
            segment_durations.append(even_dur)
            print(f"  [WARN] Segment {i} file not found, using even split: {even_dur:.1f}s")

    # Calculate cumulative timestamps
    timestamps = []
    cumulative = 0.0
    for dur in segment_durations:
        timestamps.append({"start": cumulative, "end": cumulative + dur})
        cumulative += dur
    total_duration = cumulative

    print(f"[L6] Total duration: {total_duration:.1f}s")
    for i, (ts, em) in enumerate(zip(timestamps, emotion_map)):
        print(f"  Scene {i:2d} | {ts['start']:6.1f}s - {ts['end']:6.1f}s | "
              f"{segment_durations[i]:5.1f}s | {em['emotion']:12s}")

    # ── Load stills ──
    print(f"\n[L6] Loading stills from {stills_dir}...")
    stills = []
    for i in range(num_scenes):
        # Try multiple naming patterns
        candidates = [
            stills_path / f"scene_{i:02d}.png",
            stills_path / f"scene_{i:02d}.jpg",
            stills_path / f"scene_{i:02d}.jpeg",
            stills_path / f"scene_{i:02d}.webp",
            stills_path / f"{i:02d}.png",
            stills_path / f"{i:02d}.jpg",
        ]
        loaded = False
        for cand in candidates:
            if cand.exists():
                img = Image.open(cand).convert("RGB")
                # Pre-scale to slightly larger than target for Ken Burns headroom
                scale_factor = 1.3
                img = img.resize(
                    (int(target_w * scale_factor), int(target_h * scale_factor)),
                    Image.LANCZOS
                )
                stills.append(np.array(img))
                print(f"  Scene {i:2d}: {cand.name} ({img.size[0]}x{img.size[1]})")
                loaded = True
                break

        if not loaded:
            # Reuse the most recent still if available (for fewer stills than scenes)
            if stills:
                stills.append(stills[-1])
                print(f"  Scene {i:2d}: [REUSE scene_{i-1:02d}] — no new image, holding previous")
            else:
                # Generate placeholder still as last resort
                print(f"  Scene {i:2d}: [PLACEHOLDER] — no image found")
                emotion = emotion_map[i]["emotion"]
                color_hex = EMOTION_VISUALS.get(emotion, DEFAULT_VISUAL)[0]
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)

                placeholder = Image.new("RGB",
                                        (int(target_w * 1.3), int(target_h * 1.3)),
                                        (r // 4, g // 4, b // 4))
                pd = ImageDraw.Draw(placeholder)

                pfont = None
                for fp in ["/System/Library/Fonts/Helvetica.ttc",
                           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                    if os.path.exists(fp):
                        try:
                            pfont = ImageFont.truetype(fp, 48)
                            break
                        except Exception:
                            continue
                if pfont is None:
                    pfont = ImageFont.load_default()

                pd.text((target_w * 0.3, target_h * 0.5),
                        f"SCENE {i:02d}\n{emotion.upper()}",
                        font=pfont, fill=(r, g, b))
                stills.append(np.array(placeholder))

    print(f"\n[L6] Loaded {len(stills)} stills")

    # ── Build frame generator ──
    transition_s = transition_ms / 1000.0

    def make_frame(t):
        """Generate frame at time t."""
        # Find which scene we're in
        scene_idx = 0
        for i, ts in enumerate(timestamps):
            if ts["start"] <= t < ts["end"]:
                scene_idx = i
                break
        else:
            scene_idx = num_scenes - 1

        # Scene progress (0.0 - 1.0)
        ts = timestamps[scene_idx]
        scene_dur = ts["end"] - ts["start"]
        if scene_dur > 0:
            progress = (t - ts["start"]) / scene_dur
        else:
            progress = 0.0
        progress = max(0.0, min(1.0, progress))

        # Get emotion visuals
        emotion = emotion_map[scene_idx]["emotion"]
        _, kb_effect, _ = EMOTION_VISUALS.get(emotion, DEFAULT_VISUAL)

        # Apply Ken Burns to current scene
        frame = apply_ken_burns(stills[scene_idx], kb_effect, progress,
                                target_w, target_h)

        # Check for crossfade transition
        time_into_scene = t - ts["start"]
        time_remaining = ts["end"] - t

        # Crossfade IN from previous scene
        if scene_idx > 0 and time_into_scene < transition_s:
            prev_ts = timestamps[scene_idx - 1]
            prev_emotion = emotion_map[scene_idx - 1]["emotion"]
            _, prev_kb, _ = EMOTION_VISUALS.get(prev_emotion, DEFAULT_VISUAL)
            prev_progress = 1.0  # End of previous scene

            prev_frame = apply_ken_burns(stills[scene_idx - 1], prev_kb,
                                         prev_progress, target_w, target_h)
            alpha = time_into_scene / transition_s
            frame = crossfade_frames(prev_frame, frame, alpha)

        # Render subtitle
        text = emotion_map[scene_idx]["text"]
        frame = render_subtitle(frame, text, emotion, subtitle_style)

        return frame

    # ── Build video ──
    print(f"\n[L6] Rendering {total_duration:.1f}s video at {fps}fps...")
    print(f"     ({int(total_duration * fps)} total frames)")

    video = VideoClip(make_frame, duration=total_duration).with_fps(fps)

    # ── Composite audio ──
    print(f"[L6] Compositing audio track...")
    audio_clip = AudioFileClip(audio_path)
    video = video.with_audio(audio_clip)

    # ── Export ──
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[L6] Exporting to {output_path}...")
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
        audio_bitrate="192k",
        preset="medium",
        logger="bar",
        ffmpeg_params=["-strict", "experimental", "-movflags", "+faststart"],
    )

    # Filesize
    size_mb = output_p.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"[L6] ✓ Animated short exported: {output_path}")
    print(f"     Duration: {total_duration:.1f}s | Size: {size_mb:.1f}MB")
    print(f"     Resolution: {target_w}x{target_h} @ {fps}fps")
    print(f"{'=' * 60}")

    return str(output_path)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prosody Intelligence Layer 6 — Animated Short Compositor"
    )
    parser.add_argument("--stills", required=True,
                        help="Directory containing scene_00.png - scene_NN.png")
    parser.add_argument("--audio", required=True,
                        help="Combined audio file (mp3)")
    parser.add_argument("--emotion-map", required=True,
                        help="Emotion map JSON file")
    parser.add_argument("--output", required=True,
                        help="Output .mp4 path")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Output resolution WxH (default: 1920x1080)")
    parser.add_argument("--transition-ms", type=int, default=500,
                        help="Crossfade transition duration in ms (default: 500)")
    parser.add_argument("--subtitle-style", choices=["full", "minimal", "none"],
                        default="full",
                        help="Subtitle overlay style (default: full)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second (default: 24)")

    args = parser.parse_args()

    # Parse resolution
    match = re.match(r"(\d+)x(\d+)", args.resolution)
    if not match:
        print(f"ERROR: Invalid resolution format: {args.resolution}")
        print("Expected format: WIDTHxHEIGHT (e.g., 1920x1080)")
        sys.exit(1)
    resolution = (int(match.group(1)), int(match.group(2)))

    composite_animated_short(
        stills_dir=args.stills,
        audio_path=args.audio,
        emotion_map_path=args.emotion_map,
        output_path=args.output,
        resolution=resolution,
        transition_ms=args.transition_ms,
        subtitle_style=args.subtitle_style,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
