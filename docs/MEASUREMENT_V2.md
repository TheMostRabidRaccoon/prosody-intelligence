# Measurement integrity: schema 2

This release establishes acoustic observations and their provenance. Personal
baselines, robust residuals, structured disagreement, a four-arm evaluation, and
an adaptive TTS controller remain follow-up work.

## Run without model API calls

Use Python 3.12 and ffmpeg. Install `requirements.txt`; for the offline test suite,
install `requirements-dev.txt` and run `python -m pytest -q`.

```bash
python src/prosody_pipeline.py recording.wav --measure-only --visualize
curl -X POST http://localhost:5050/api/measure -F 'audio=@recording.wav'
```

Measurement-only mode emits one observation for the whole recording and does not
transcribe it. Use short, single-speaker clips when that is the intended unit.
It leaves transcript-gap fields unavailable. No personal baseline is fitted.

For transcript-aligned measurements, use the CLI without `--measure-only` or
`POST /api/analyze`. This calls OpenAI transcription. Interpretation is off by
default; request `--interpret` or the form field `skip_llm=false` to enable it.
The browser offers these three modes explicitly, defaulting to acoustics only.

The optional `speaker_id`, `channel`, and `register` fields (CLI flags use hyphens)
describe context supplied by the caller. A speaker ID declares that the entire
recording belongs to one speaker; it is not inferred or diarized. Without it,
speaker identity is unassigned. Pitch categories are no longer labeled speakers.

## Recording and feature contract

Each input is copied into `uploads/<generated-recording-id>/source.<extension>`.
The original filename is metadata; it never determines a write path. The
recording metadata contains the ID, original filename, source SHA-256 and byte
count. Repeated filenames have independent input/output directories. Set
`PROSODY_OUTPUT_DIR` to redirect artifacts; uploads live alongside that directory.
HTTP uploads are limited to 500 MiB by default, configurable with
`PROSODY_MAX_UPLOAD_MB`. This limit does not increase a provider's upload limit.

All supported audio, including WAV, is normalized to mono 16 kHz PCM16 for
measurement. The cache key includes source content and conversion configuration;
conversions and JSON artifacts are published atomically. Records include the
normalized audio hash, extractor version and configuration. Mono downmixing does
not separate speakers. Clipping statistics refer to the normalized samples.

An annotated segment has `schema_version`, `segment_id`, `source`, `start`, `end`,
`text`, `speaker_id`, `speaker_identity_source`, `channel`, `register`,
`measurement`, `validity`, `units`, `quality`, and `extractor` fields.

| Measurement | Definition and qualification |
| --- | --- |
| `f0_mean_hz` | Mean of finite positive pitch frames in the interval, Hz. |
| `f0_std_hz` | Population standard deviation of at least two voiced frames, Hz. |
| `f0_delta_hz` | Difference between means of the final and initial thirds of voiced frames; requires six frames. |
| `intensity_mean_db` | Arithmetic mean of finite Praat intensity frames, in uncalibrated dB relative to Praat's reference. Not a calibrated physical SPL. |
| `rms_amplitude` | RMS of the normalized digital samples. Zero is valid for silence. |
| `rms_dbfs` | `20 log10(rms_amplitude)`; unavailable for digital silence. |
| `intensity_crossings_per_second` | Downward crossings of mean intensity divided by interval duration. Always marked `proxy_unvalidated`; **not syllables per second**. |
| `transcript_gap_before_s`, `transcript_gap_after_s` | Nonnegative gaps between ASR boundaries; **not verified acoustic silence**. The final gap is unavailable. |

Full numerical precision is stored; display rounding happens in the UI. Missing
values are JSON `null`, never NaN or Infinity. Validity is per feature: `ok`,
`unvoiced`, `silent`, `insufficient_frames`, `invalid_interval`, `out_of_bounds`,
`extraction_failed`, `proxy_unvalidated`, `asr_boundary_gap`, `unavailable`, or
`no_transcript`. Very short recordings may have valid RMS measurements while
Praat cannot extract pitch or intensity; these failures remain explicit.
An invalid/reversed/out-of-bounds interval is not clamped into a fabricated valid
measurement. Quality includes sample/frame counts, voiced fraction and the
fraction of normalized samples near full scale. These are observable quality
indicators, not calibrated measurement uncertainty or psychological confidence.

`prosody` is a deprecated compatibility dictionary. `avg_pitch` aliases mean F0;
`pitch_variance` aliases standard deviation, despite its old name; `energy` is the
legacy clipped intensity display mapping; pause fields alias transcript gaps.
**`speaking_rate` is always null.** Consumers must handle nullable aliases and
should migrate to the versioned measurement fields. `speaker_threshold` remains
in the API response as null. Existing consumers that assume all numbers are
present need updates before deploying this version.

## Synthesis, calibration and video

Every generation run writes to `output/renders/<run-id>/`. Its
`generation_manifest.json` includes every source line, stable segment ID,
annotation/fallback status, original and spoken text, intended delivery, actual
voice ID, model ID, voice settings, clip hash, duration and status. Successfully
assembled runs additionally include the exact millisecond onset/end/overlap of
each clip, combined duration and combined-file hash.

Failed clips remain attached to their original line and target. An incomplete
run does not publish `combined.mp3` or return a successful audio URL. Python
raises `IncompleteGenerationError` with the manifest path; the HTTP route returns
422 with `complete=false` and `manifest_url`.

The emotion tagger can annotate source line IDs but cannot replace source text or
reorder the script. Missing/invalid annotations receive an explicit neutral
fallback status. This does not establish the accuracy of emotion annotations.

```bash
python src/reverse_pipeline.py script.txt --multi-voice
python src/calibration.py output/renders/RUN_ID/combined.mp3
```

Calibration verifies the manifest and hashes, then measures each original clip
from zero through its decoded duration. It reports the manifest's assembled
timestamps as provenance. It never estimates timing using an even split or a
glob of segment filenames. An optional emotion-map argument must match the
manifest's texts, delivery labels and parameters. Video uses the same verified
timeline, with incoming scenes taking precedence during crossfade overlaps.

Older productions without manifests must be regenerated before scoring or video
assembly with this release. They cannot be reliably reconstructed from names or
durations alone. Historical checked-in calibration logs are preserved unchanged.

New scores use `target_range_coverage_pct`, `assessed_feature_count`, and
`target_feature_count`. They describe compliance with the **legacy manual**
pitch-SD and relative-intensity targets. The syllable-rate target is unassessable
and excluded; the report explicitly marks partial coverage. Percentages are not
directly comparable to historical three-feature scores and do not represent
emotion accuracy. Reports keep historical and schema-2 summaries separate.
No parameter learning or automatic tuning occurs. New runs
have unique IDs and separate atomic JSON files rather than rewriting shared
history. `--reference` (legacy alias `--ground-truth`) accepts schema-2 records
with compatible extractor settings and reports unmatched descriptive means,
not proof of equivalent emotion.

## Interpretation and comparison

Optional interpretation receives feature units and validity and must distinguish
observations from hypotheses. The current two-arm comparison uses identical
instructions; lexical-only T omits timing, speaker labels and acoustics. It is
explicitly unscored. It does not yet implement the proposed four-arm experiment,
reference labels, measured confidence calibration, or held-out evaluation.

## Integration boundary

This patch is based on GitHub main at `443966b`. The separately maintained local
batching/MCP work and the unmerged large-recording/diarization branch are not
overwritten or merged by it. Those changes need a deliberate integration before
deploying this branch over a customized local service. In particular, this patch
does not add long-file transcription chunking or persistent speaker diarization.
