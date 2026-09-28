# Prosody Intelligence 🦝

**Rabid Raccoon Intelligence, LLC** — Bidirectional Acoustic Analysis & Synthesis

An acoustic measurement and creative speech-synthesis toolkit. Measurement schema 2
records feature values, units, validity, and source identity. Interpretation is
optional. Synthesis records voice settings and exact assembly timing; calibration
checks descriptive targets without automatically tuning the voices.

**Current contract and migration:** [Measurement integrity, schema 2](docs/MEASUREMENT_V2.md).
Personal baselines, residuals, structured disagreement and the adaptive controller
are follow-up work. Existing consumers must handle nullable measurements.

**Repo:** `https://github.com/TheMostRabidRaccoon/prosody-intelligence`

---

## What It Does

### Forward Pipeline (Analysis Direction)

1. **Measurement-only mode** — normalize and measure audio locally with no model API calls.
2. **Optional transcription** — OpenAI Whisper supplies text and ASR boundaries for segment measurements.
3. **Acoustic observations** — mean F0, pitch standard deviation, uncalibrated intensity, digital RMS, and an explicitly unvalidated intensity-crossing proxy.
4. **Validity and provenance** — source hashes, extractor configuration, per-feature status, frame counts and declared speaker context.
5. **Optional interpretation** — a separate LLM readout that distinguishes observations from hypotheses.

Unavailable features remain null. Transcript gaps are not verified silence, and
intensity crossings are not syllables. Pitch is never used to manufacture speaker identities.

### Reverse Pipeline (Synthesis Direction)

Feed it text — a script, a document, a dialogue transcript.

1. **Emotion Detection** — LLM classifies the emotional register of every line
2. **Parameter Mapping** — Each of 14 emotions maps to tuned TTS parameters (stability, expressiveness, pacing)
3. **Voice Routing** — Six distinct AI voices, assigned per speaker. Tag your script with speaker names and the system routes each line to the right voice with the right emotional delivery
4. **TTS Rendering** — ElevenLabs generates each segment with emotion-specific parameters
5. **Crossfade Assembly** — Segments are stitched with crossfade so there are no awkward gaps between speakers

**Output:** Multi-voice audio plus a generation manifest with source line IDs, actual voice/model/settings, clip hashes, generation status, and exact crossfade timing.

### Calibration (Descriptive Target Checks)

Calibration verifies a generation manifest, then measures each successful source
clip independently. It reports target-range coverage against the existing manual
pitch-SD and relative-intensity ranges. The old syllable-rate target is explicitly
unassessable by the current proxy. Available-feature counts accompany every score.

These percentages do not measure emotion accuracy and are not directly comparable
to historical three-feature scores. Calibration writes observations, not learned
voice parameters. Failed renders cannot be silently scored under another line's
label or published as a complete production.

### Video Compositor (Session Director)

Give it a script and a folder of images, and the compositor builds an animated short.

1. **Ken Burns camera movement** matched to emotion — slow zoom for dramatic, quick zoom for comedy, drift for hesitation, shake for anger
2. **Emotion-colored subtitles** tagged with speaker name
3. **Crossfade transitions** between scenes
4. **Full audio compositing** with proper encoding

The Session Director automates end-to-end: raw document in → parse dialogue from narration → route speakers to voices → generate audio → composite final video. One input, one output.

---

## The Emotion Palette

14 base emotions, each with distinct vocal parameters and expected acoustic signatures:

| Emotion | What It Sounds Like | Key Parameter |
|---------|-------------------|---------------|
| Sarcastic | Deliberate, dry. The edge lands through slowness. | Low stability, slow pacing |
| Comedic | Maximum expressiveness. Timing is everything. | Max pitch variation |
| Dramatic | Theatrical weight. Slow, full style. | Max expressiveness, theatrical slow |
| Excited | Fast, high energy, can't contain it. | High speed, high energy |
| Angry | Intense, forceful, controlled instability. | High energy, low stability |
| Urgent | Clipped, pressured. Speed drives it. | Fast, compressed |
| Contempt | Cold superiority. Stable, slow, dripping. | High stability, slow |
| Disgusted | Visceral. The words taste bad. | Low stability, low energy |
| Confident | Steady, authoritative. Stability is power. | High stability |
| Analytical | Clinical, precise. Minimal style. | High stability, minimal expressiveness |
| Neutral | Baseline conversational. | Default parameters |
| Hesitant | Uncertain, trailing off. The voice shrinks. | Low energy, trailing speed |
| Tender | Warm, gentle. Stable but hushed. | High stability, low energy |
| Resigned | Flat, drained. Near-monotone. Slowest delivery. | Max stability, min expressiveness |

### Palette Expansion (In Development)

The base palette covers static single emotions. Human vocal expression routinely uses more complex acoustic structures — multi-phase vocalizations, emotions containing their own opposite, and meaningful silence. A palette expansion system is in development that learns new acoustic contours from recorded human reference performances, extending the engine's expressive range beyond what text prompt tags can describe.

See `docs/` for the expansion registry.

---

## Exploratory Comparison

The two-arm comparison uses identical instructions and supplies either lexical
text alone or text plus measured acoustics. The lexical-only arm omits timestamps.
It remains unscored: differing narratives do not establish improved accuracy.
The proposed four-arm baseline/disagreement experiment is not implemented yet.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            FORWARD PIPELINE                 │
│  Audio → Whisper → Parselmouth/Praat →      │
│  Alignment → Measurements → Optional LLM    │
├─────────────────────────────────────────────┤
│           REVERSE PIPELINE                  │
│  Text → Emotion Detection → Parameter Map → │
│  Voice Routing → ElevenLabs TTS → Assembly  │
├─────────────────────────────────────────────┤
│          CALIBRATION LOOP                   │
│  Generated Audio → Forward Pipeline →       │
│  Compare intended vs achieved → Log deltas  │
│  → Versioned observation logs               │
├─────────────────────────────────────────────┤
│          VIDEO COMPOSITOR                   │
│  Script + Images → Ken Burns motion →       │
│  Emotion-colored subtitles → Final video    │
├─────────────────────────────────────────────┤
│             WEB API                         │
│  Flask REST · /api/reverse · /api/measure   │
│  Visualization serving · HTML templates     │
└─────────────────────────────────────────────┘
         │                    ▲
         ▼                    │
┌─────────────────────────────────────────────┐
│      RACCOON SWARM (separate repo)          │
│  Dialogue export → SPEAKER-prefixed TXT →   │
│  Feeds directly into Reverse Pipeline       │
│  Swarm sessions become animated shorts      │
└─────────────────────────────────────────────┘
```

---

## Voice Cast

Six voices available for multi-speaker production. In the RRI swarm configuration:

| Speaker | Voice | ElevenLabs ID |
|---------|-------|---------------|
| Claude | George | `JBFqnCBsd6RMkjVDRZzb` |
| Grok | Callum | `N2lVS1w4EtoT3dr4eOWO` |
| Gemini | Adam | `pNInz6obpgDQGcFmaJgB` |
| GPT | Brian | `nPczCjzI2devNBz1zQrb` |
| Kyra | Kyra | `DLm68MJvI3f80aZijHAn` |
| Narrator | The Narrator | `Aqqzjc8no56A9UgQcOnP` |

Each model in the RRI swarm selected its own voice. The voice assignments are canonical across all Chitterverse productions.

---

## Chitterverse Integration

This system is the voice engine for the Chitterverse — an animated series produced end-to-end by the RRI swarm. The production pipeline:

1. **Source:** Real swarm session transcripts
2. **Script:** Written by Grok
3. **Dialogue export:** Raccoon Swarm server emits SPEAKER-prefixed TXT
4. **Prosody processing:** This repo — emotion detection → per-line parameter mapping → multi-voice TTS
5. **Art:** Rendered by Gemini
6. **Assembly:** Claude Code

Every script passes through the Reverse Prosody Engine. The stability, style, and similarity values, the prompt tags (`[exhausted][flat][dry]`, `[southern accent][hesitant][timid]`), and timing direction are all output of the engine, not hand-tuned per panel.

The reads work because the synthesis step is governed by structured prosody data, not vibes. Gemini's voice performance in the WiFi sketch is a *performance*, not a TTS read. Drew the pigeon's flat "Motherfucker." in PIGEONS — the flatness is the joke, and the engine produced the parameters that make the flatness land.

Production time: under 3 hours per episode.

---

## Research Context

This work inverts a clinical neuropsychology research axis.

**Analysis direction (2006):** The RAVLT classification study (Schoenberg, Dawson et al., *Archives of Clinical Neuropsychology*) measured how humans encode and retrieve verbally presented information — how acoustic-temporal features of speech affect downstream cognition.

**Synthesis direction (2025–present):** Prosody Intelligence asks the inverse question — how do we generate acoustically precise presentation from text? Same research axis, opposite directions. The clinical research measured the downstream effects of prosody on human cognition. This system produces the prosody.

**Calibration loop:** Closes the bidirectional circuit. Forward analysis validates reverse synthesis. The system measures itself using the same methodology it uses on human speech.

**Connected publications:**
- Schoenberg, Dawson et al. (2006). RAVLT Classification Statistics. *Archives of Clinical Neuropsychology*, 21(7).
- Ruwe et al. (2008). Computer-Based vs Face-to-Face Cognitive Rehabilitation. *Professional Psychology*, 39(2).
- Dawson, K. (2026). Coordination Structure as a Behavioral Determinant in Multi-Model AI Orchestration. SSRN 6311560.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Transcription | OpenAI Whisper (word-level timestamps) |
| Prosody extraction | Parselmouth / Praat (pitch, energy, rate, pauses) |
| Alignment | Custom sync engine (text + acoustic signature per segment) |
| LLM analysis | GPT-4o with prosody-aware system prompt |
| Reverse synthesis | GPT-4o emotion detection + ElevenLabs TTS (6 voices) |
| Proof testing | A/B comparison framework (text-only vs prosody-aware) |
| Calibration | Forward analysis on generated audio + delta logging |
| Video compositor | MoviePy + Ken Burns effects + emotion-colored subtitles |
| Session director | End-to-end automation (document in → video out) |
| Web API | Flask REST with visualization serving |

---

## Setup

```bash
# Clone
git clone https://github.com/TheMostRabidRaccoon/prosody-intelligence.git
cd prosody-intelligence

# Python 3.12 is the tested runtime
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# System dependencies
# Praat (via Parselmouth) — installs with pip
# ffmpeg — required for audio/video processing
#   macOS: brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg

# For hosted transcription, interpretation or synthesis only:
cp .env.example .env
# Edit .env: OPENAI_API_KEY, ELEVENLABS_API_KEY

# Run the web API
python src/app.py
```

## Usage

### Forward Analysis (CLI)
```bash
python src/prosody_pipeline.py input/recording.m4a --measure-only --visualize
# Output: measurements and plot in output/recordings/<recording-id>/
```

### Reverse Synthesis (CLI)
```bash
python src/reverse_pipeline.py input/script.txt --multi-voice
# Output: multi-voice audio in output/
```

### Session Director (End-to-End)
```bash
python src/session_director.py input/script.docx input/frames/
# Output: complete animated short in output/
```

### Web API
```bash
# Forward analysis
curl -X POST http://localhost:5050/api/measure -F "audio=@recording.m4a"

# Reverse synthesis
curl -X POST http://localhost:5050/api/reverse -H 'Content-Type: application/json' \
  -d '{"text":"CLAUDE: Hello.","multi_voice":true,"generate_audio":true}'
```

---

## Project Structure

```
prosody-intelligence/
├── src/              # Core Python modules
├── docs/             # Documentation and specs
├── input/            # Input files (scripts, audio)
├── output/           # Generated artifacts (gitignored)
├── templates/        # Flask HTML templates
├── test_audio/       # Test recordings (gitignored)
└── .env.example      # Environment variable template
```

---

## Roadmap

- [ ] Palette expansion system — learn new acoustic contours from recorded human reference performances
- [ ] Conductor voice input pipeline — route audio through forward pipeline, generate structured prosody metadata alongside transcript for richer model input
- [ ] Witness Prep Analyzer integration — analysis-direction tool tuned for courtroom register assessment
- [ ] Swarm daemon integration — direct pipeline from SwarmDaemon output to Session Director
- [ ] Real-time forward analysis — streaming prosody extraction during live conversation
- [ ] Calibration dataset publication — empirical data on TTS emotion accuracy across parameter configurations

---

## License

Proprietary — Rabid Raccoon Intelligence, LLC. Eight provisional patents filed (November 2025).

---

*The base palette has 14 emotions. Human vocal expression has infinity. This system is where the gap gets smaller.* 🦝

## Offline regression tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic audio and fake provider clients; they require ffmpeg but no
API credentials. CI runs the suite on Python 3.12.
