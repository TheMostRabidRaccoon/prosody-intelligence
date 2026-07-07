# Prosody Intelligence 🦝

**Rabid Raccoon Intelligence, LLC** — Bidirectional Acoustic Analysis & Synthesis

A system that understands *how things are said*, not just what is said. Analyzes the acoustic fingerprint of human speech, generates emotionally calibrated multi-voice audio from text, and closes its own loop through self-correcting calibration.

**Repo:** `https://github.com/TheMostRabidRaccoon/prosody-intelligence`

---

## What It Does

### Forward Pipeline (Analysis Direction)

Feed it audio — a meeting recording, a podcast, a voicemail, a therapy session, a sales call, a courtroom transcript.

1. **Transcription** — OpenAI Whisper with word-level timestamps
2. **Prosody Extraction** — Parselmouth/Praat pulls acoustic features from every segment: fundamental frequency (pitch), pitch direction and variance, vocal energy, speaking rate, and silence duration between utterances
3. **Alignment** — Custom sync engine pairs every sentence with its acoustic signature (text + numbers together)
4. **LLM Analysis** — GPT-4o reads the annotated transcript and analyzes what the voice reveals that words alone don't: rising pitch on "I'm fine," the 1.2-second pause before answering a direct question, the energy drop when someone mentions a specific name

**Output:** Deep analysis of emotional dynamics, power relationships, hesitation patterns, and subtext — grounded in measurable acoustic data, not guesswork.

### Reverse Pipeline (Synthesis Direction)

Feed it text — a script, a document, a dialogue transcript.

1. **Emotion Detection** — LLM classifies the emotional register of every line
2. **Parameter Mapping** — Each of 14 emotions maps to tuned TTS parameters (stability, expressiveness, pacing)
3. **Voice Routing** — Six distinct AI voices, assigned per speaker. Tag your script with speaker names and the system routes each line to the right voice with the right emotional delivery
4. **TTS Rendering** — ElevenLabs generates each segment with emotion-specific parameters
5. **Crossfade Assembly** — Segments are stitched with crossfade so there are no awkward gaps between speakers

**Output:** Multi-voice, emotionally nuanced audio from plain text. A six-minute script with five speakers and twelve emotional shifts becomes a polished audio performance in under two minutes.

### Calibration Loop (Self-Correction)

After generating audio, the system runs it back through the forward pipeline — the same prosody extraction it uses on human speech. It measures the pitch variance, energy, and speaking rate the TTS actually produced, compares those against the acoustic signatures each emotion is *supposed* to hit, and logs the deltas.

Over time, this builds an empirical dataset: here is what "sarcastic" actually sounds like when ElevenLabs renders it with these parameters. Here is where "analytical" overshoots on pitch variance. Here is where "comedic" needs a speed bump.

The system prints specific tuning recommendations:
```
Analytical accuracy is 37.5%. Pitch variance is too high
(24.6 vs expected 3-20). Increase stability from 0.75 to 0.85.
```

**Output:** A self-correcting system that gets more accurate with every run, with receipts.

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

## The Proof Test

Built-in A/B validation. The system runs the same transcript through the LLM twice — once with just the text, once with text plus prosody data — and shows the difference side by side.

The prosody-aware analysis consistently catches things the text-only version misses:
- Contradictions between words and tone
- Masked emotions
- Power asymmetries in conversation
- Moments where silence says more than speech

This is the clinical trial for the system's core claim: acoustic data changes what AI can understand about human communication.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            FORWARD PIPELINE                 │
│  Audio → Whisper → Parselmouth/Praat →      │
│  Alignment → LLM Analysis → Report          │
├─────────────────────────────────────────────┤
│           REVERSE PIPELINE                  │
│  Text → Emotion Detection → Parameter Map → │
│  Voice Routing → ElevenLabs TTS → Assembly  │
├─────────────────────────────────────────────┤
│          CALIBRATION LOOP                   │
│  Generated Audio → Forward Pipeline →       │
│  Compare intended vs achieved → Log deltas  │
│  → Tuning recommendations → Repeat          │
├─────────────────────────────────────────────┤
│          VIDEO COMPOSITOR                   │
│  Script + Images → Ken Burns motion →       │
│  Emotion-colored subtitles → Final video    │
├─────────────────────────────────────────────┤
│             WEB API                         │
│  Flask REST · /api/reverse · /api/forward   │
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
| ChatGPT | Eric | `cjVigY5qzO86Huf0OWal` |
| Perplexity | Daniel | `onwK4e9ZLuTAKqWW03F9` |
| Kyle | Liam | `TX3LPaxmHKxFdv7VOQHJ` |

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

# Install dependencies
pip install -r requirements.txt

# System dependencies
# Praat (via Parselmouth) — installs with pip
# ffmpeg — required for audio/video processing
#   macOS: brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY, ELEVENLABS_API_KEY

# Run the web API
python src/app.py
```

## Usage

### Forward Analysis (CLI)
```bash
python src/forward.py input/recording.m4a
# Output: prosody analysis report in output/
```

### Reverse Synthesis (CLI)
```bash
python src/reverse.py input/script.txt
# Output: multi-voice audio in output/
```

### Session Director (End-to-End)
```bash
python src/session_director.py input/script.docx --images input/frames/
# Output: complete animated short in output/
```

### Web API
```bash
# Forward analysis
curl -X POST http://localhost:5050/api/forward -F "audio=@recording.m4a"

# Reverse synthesis
curl -X POST http://localhost:5050/api/reverse -F "script=@script.txt"
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
