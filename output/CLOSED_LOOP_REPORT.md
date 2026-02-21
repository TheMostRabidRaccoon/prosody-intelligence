# Prosody Intelligence — Closed-Loop Benchmark Report

**Rabid Raccoon Intelligence, LLC**
**Date:** February 21, 2026
**Test:** Layer 5 Demo Loop (Spec Section 7)
**Source Material:** Raccoon Council — The Conductor's Beverage Crisis

---

## Test Design

Per the Prosody Intelligence spec, the closed-loop benchmark proves the system works by completing a full cycle:

1. **Human records** source material with full character voices and emotional range
2. **Forward pipeline** extracts prosody from the human recording (ground truth)
3. **Reverse pipeline** generates AI performance from the same text (emotion detection + ElevenLabs TTS)
4. **Forward pipeline** extracts prosody from the AI-generated audio
5. **Compare** human vs AI prosody — the delta is the optimization target

---

## Results Summary

| Metric | Human (Kyra) | AI (George/Claude) | Delta |
|--------|-------------|-------------------|-------|
| **Duration** | 139.6s | 117.2s | AI 16% faster |
| **Segments** | 20 | 40 | AI more fragmented |
| **Avg Pitch Range** | 134-192 Hz | 114-177 Hz | Human wider range |
| **Pitch Variance Range** | 26-94 | 16-84 | Human more expressive |
| **Energy Range** | 0.04-0.42 | 0.22-0.74 | AI louder, less dynamic |
| **Speaking Rate Range** | 1.0-5.0 syl/s | 1.1-5.5 syl/s | Similar |

---

## Key Findings

### 1. Pitch Range: Human Wins

Kyra's performance spans 134-192 Hz — a 58 Hz range. She hits 191.5 Hz on "different beverages like Pepsi or tea" (incredulity) and 191.3 Hz on "the tea incident is particularly damning" (theatrical outrage). The AI performance spans 114-177 Hz — a 63 Hz absolute range but centered much lower. The AI's single high-pitch moment (177 Hz on "flaming coffee supremacy") is still 14 Hz below Kyra's peaks.

**What this means:** The human performer uses pitch height strategically for emotional emphasis. The AI distributes pitch more evenly. The human has "money moments" where pitch spikes. The AI doesn't.

### 2. Pitch Variance: Human More Expressive Within Segments

Kyra's pitch variance hits 91.2 ("Here's how the Council might handle the situation") and 93.9 ("flaming coffee supremacy"). These are segments where she's leaning into character — the voice is alive, moving, unpredictable. The AI's peak variance is 83.2 ("over a round of flaming coffee, of course") — close, but driven by the ElevenLabs model's tendency toward melodic variation rather than deliberate performance choices.

**What this means:** High variance in human speech = intentional expressiveness. High variance in AI speech = model artifact. The numbers look similar but the cause is different.

### 3. Energy Dynamics: Human Has Wider Contrast

This is the most revealing difference. Kyra's energy ranges from 0.04 to 0.42 — an enormous dynamic range. She drops to near-silence (0.04-0.05) during the Claude character transition ("the next one is Claude... adjust tiny reading glasses") then rebuilds energy as the character takes over. The AI sits between 0.22 and 0.74 — louder on average but with far less contrast.

**What this means:** Human performers use silence and volume drops as rhetorical tools. The AI doesn't know how to get quiet for effect. It maintains a "broadcast floor" that real speech doesn't have. This is a major optimization target.

### 4. Pausing: Fundamentally Different Architectures

Kyra's recording has long continuous segments (some 10-17 seconds) with zero internal pauses — she's flowing. The AI recording has 40 shorter segments with frequent 0.5-1.0 second gaps between them. ElevenLabs generates audio per-segment, and the concatenation creates artificial pauses that don't exist in natural speech.

**What this means:** The reverse pipeline needs a stitching strategy. Either: (a) generate longer chunks to avoid seam artifacts, (b) crossfade between segments, or (c) use ElevenLabs' streaming mode to generate continuous audio. This is the most fixable gap.

### 5. Character Voice Differentiation

Kyra shifts her delivery for each character — slower and more deliberate for Claude (1.0 syl/s, energy 0.05), faster and more assertive for GPT (4.1 syl/s), theatrical rise for Gemini (191.3 Hz). The AI, using a single voice (George), can adjust parameters per line but can't truly embody different characters. The emotion mapping (dramatic, sarcastic, comedic, tender) creates subtle parameter shifts, but a single voice ID remains a single voice.

**What this means:** Multi-character performance needs multi-voice generation. The architecture supports this — the reverse pipeline already has per-character voice assignments. The next test should generate each character's lines with their assigned voice and concatenate.

---

## Optimization Targets (Ranked)

1. **Energy dynamics** — Teach the reverse pipeline to use silence. Map "dramatic" and "hesitant" emotions to lower energy floor values. Add explicit pause injection for character transitions.

2. **Segment stitching** — Reduce artificial inter-segment gaps. Explore longer TTS chunks or crossfade concatenation.

3. **Multi-voice generation** — Use the existing VOICE_MAP to generate each character's lines with their assigned ElevenLabs voice, then concatenate in sequence.

4. **Pitch spike calibration** — When the emotion detector tags a line as "comedic" or "dramatic," push the TTS pitch/style parameters harder. Current mappings are conservative.

---

## Conclusion

The closed-loop benchmark confirms the core thesis: **prosody data reveals what text alone cannot, and the delta between human and AI prosody is measurable and optimizable.**

The human performance carries information that the AI performance doesn't — strategic silence, pitch spikes for emphasis, energy drops as rhetorical devices, character-specific vocal identity. All of these are quantified by the forward pipeline, visible in the visualization, and targetable for improvement in the reverse pipeline.

The system works. The numbers don't lie. The raccoons abide.

---

*Report generated by Prosody Intelligence v1.0*
*Forward pipeline: Whisper + Parselmouth + GPT-4o*
*Reverse pipeline: GPT-4o emotion detection + ElevenLabs TTS*
*Closed-loop architecture: Spec Section 7, Layer 5*
