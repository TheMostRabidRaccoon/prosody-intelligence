# Prosody Intelligence

**Built by Rabid Raccoon Intelligence, LLC**

---

## The Short Version

Prosody Intelligence is a system that understands *how* things are said, not just *what* is said.

It analyzes the acoustic fingerprint of human speech — pitch, energy, pacing, pauses, vocal contour — and uses that data to reveal what tone of voice actually communicates. Then it flips the pipeline: given plain text, it generates emotionally accurate multi-voice audio that sounds like it was *performed*, not read aloud by a robot.

The system closes its own loop. It measures whether the audio it generates actually hits the emotional targets it intended, and adjusts. That makes it a calibrated controller, not a vibes engine.

---

## What It Actually Does

### It Listens

Feed it any audio — a meeting recording, a podcast, a voicemail, a therapy session, a sales call.

The forward pipeline transcribes the audio with word-level timestamps, then extracts acoustic features from every segment: fundamental frequency (pitch), pitch direction and variance, vocal energy, speaking rate, and the silences between utterances. It aligns all of that with the transcript so every sentence carries both its words and its acoustic signature.

Then an LLM reads the annotated transcript — text *and* numbers — and analyzes what the voice reveals that the words alone don't. The rising pitch on "I'm fine." The 1.2-second pause before answering a direct question. The energy drop mid-sentence when someone mentions a specific name.

**What you get:** A deep analysis of emotional dynamics, power relationships, hesitation patterns, and subtext — grounded in measurable acoustic data, not guesswork.

### It Speaks

Feed it text — a script, a document, a transcript — and the reverse pipeline detects the emotional register of every line, maps it to voice parameters, and generates audio through ElevenLabs TTS.

This isn't "pick a voice and hit play." Each of the 14 emotions in the palette has a tuned set of parameters that control stability, expressiveness, and pacing:

- *Sarcastic* gets low stability and slow pacing so the contempt lands.
- *Dramatic* gets maximum expressiveness and theatrical slowness for weight.
- *Hesitant* gets low energy and trailing speed — the voice actually shrinks.
- *Comedic* gets maximum pitch variation because timing is everything.
- *Resigned* goes near-monotone, the slowest delivery — flat, drained, done.

It supports six distinct AI voices that can be mixed in a single production. Tag your script with speaker names and the system routes each line to the right voice with the right emotional delivery. The segments are crossfade-stitched so there are no awkward gaps between speakers.

**What you get:** Multi-voice, emotionally nuanced audio from plain text. A six-minute script with five speakers and twelve emotional shifts becomes a polished audio performance in under two minutes.

### It Watches Itself

The calibration loop is the piece that makes this more than a demo.

After generating audio, the system runs it back through the forward pipeline — the same prosody extraction it uses on human speech. It measures the pitch variance, energy, and speaking rate that the TTS *actually produced*, compares those against the acoustic signatures each emotion is *supposed* to hit, and logs the deltas.

Over time, this builds an empirical dataset: here is what "sarcastic" actually sounds like when ElevenLabs renders it with these parameters. Here is where "analytical" overshoots on pitch variance. Here is where "comedic" needs a speed bump.

The system prints specific tuning recommendations: "Analytical accuracy is 37.5%. Pitch variance is too high (24.6 vs expected 3-20). Increase stability from 0.75 to 0.85."

**What you get:** A self-correcting system that gets more accurate with every run, with receipts.

### It Produces

Give it a script and a folder of images, and the compositor builds an animated short.

Each image gets Ken Burns camera movement matched to the emotion of its scene — slow zoom for dramatic moments, quick zoom for comedy, drift for hesitation, shake for anger. Subtitles are color-coded by emotion and tagged with the speaker name. Transitions crossfade between scene changes. The audio track is composited with proper encoding for universal playback.

The Session Director automates the entire pipeline end-to-end: drop in a raw document, and it parses dialogue from narration, routes speakers to voices, generates the audio, and composites the final video. One input, one output.

**What you get:** A finished animated short from a text file and some stills. No video editing software. No manual timing. No post-production.

---

## The Proof

The system includes a built-in A/B test called the Proof Test. It runs the same transcript through the LLM twice — once with just the text, once with text plus prosody data — and shows you the difference side by side.

The prosody-aware analysis consistently catches things the text-only version misses: contradictions between words and tone, masked emotions, power asymmetries in conversation, and moments where silence says more than speech.

This is the clinical trial for the system's core claim: that acoustic data changes what AI can understand about human communication.

---

## The Emotion Palette

14 emotions, each with distinct vocal parameters and expected acoustic signatures:

| | Emotion | What It Sounds Like |
|---|---|---|
| | **Sarcastic** | Deliberate, dry. Low stability lets the edge land. |
| | **Comedic** | Maximum expressiveness. Pitch variation is everything. |
| | **Dramatic** | Theatrical weight. Slow, full style, money moments. |
| | **Excited** | Fast, high energy, can't contain it. |
| | **Angry** | Intense, forceful, controlled instability. |
| | **Urgent** | Clipped, pressured. Speed drives it. |
| | **Contempt** | Cold superiority. Stable, slow, dripping with disdain. |
| | **Disgusted** | Visceral. The words taste bad. |
| | **Confident** | Steady, authoritative. Stability is the power move. |
| | **Analytical** | Clinical, precise. High stability, minimal style. |
| | **Neutral** | Baseline conversational. |
| | **Hesitant** | Uncertain, trailing off. The voice shrinks. |
| | **Tender** | Warm, gentle. Stable but hushed. |
| | **Resigned** | Flat, drained. Near-monotone. Slowest delivery. |

---

## The Stack

| Layer | Function | Technology |
|---|---|---|
| **1A** | Audio transcription | OpenAI Whisper (word-level timestamps) |
| **1B** | Prosody extraction | Parselmouth / Praat (pitch, energy, rate, pauses) |
| **2** | Alignment | Custom sync engine (text + acoustic signature per segment) |
| **3** | LLM analysis | GPT-4o with prosody-aware system prompt |
| **4** | Reverse synthesis | GPT-4o emotion detection + ElevenLabs TTS (6 voices) |
| **5** | Proof testing | A/B comparison framework (text-only vs prosody-aware) |
| **5.5** | Calibration loop | Forward analysis on generated audio + delta logging |
| **6** | Video compositor | MoviePy + Ken Burns effects + emotion-colored subtitles |
| **7** | Session director | End-to-end automation (document in, video out) |
| **Web** | API + UI | Flask REST API with visualization serving |

---

## Who This Is For

**Researchers and clinicians** studying communication patterns — therapeutic rapport, diagnostic conversations, conflict dynamics. The forward pipeline turns subjective impressions into measurable acoustic data.

**Leadership and executive coaches** working on presence, delivery, and relational fluency. The system quantifies what "how you said that landed differently" actually means.

**Content creators and storytellers** who need emotionally authentic multi-voice audio without a recording studio. The reverse pipeline and compositor turn scripts into productions.

**AI developers** building systems that need to understand or generate emotionally appropriate speech. The calibration loop provides the training signal that most emotion-in-speech systems are missing.

**Anyone who has ever said "it's not what you said, it's how you said it"** and wished they could prove it with data.

---

## Current Status

Prosody Intelligence is a working system with a full bidirectional pipeline, a self-calibrating feedback loop, and automated multimedia production capabilities. It has been tested on multi-speaker transcripts, dramatic performances, and real-world audio.

The calibration loop has completed its first empirical runs across 77 TTS segments, identifying specific parameter adjustments needed per emotion and establishing baseline accuracy metrics.

It is built, it runs, and it has receipts.

---

*Rabid Raccoon Intelligence, LLC*
*The forest is under new management.*
