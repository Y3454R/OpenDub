# OpenDub 🥥

Open-source pipeline for dubbing English video into Bangla — translation, speech synthesis, voice conversion, and prosody transfer, assembled into a single CLI.

---

## Pipeline

```
Input video (.mkv / .mp4)
        │
        ├─[ffmpeg]──────► english.wav + english.srt
        │
        ├─[Demucs]──────► vocals.wav + background.wav
        │
        ├─[pysrt]───────► segments[] with per-segment audio slices
        │
        ├─[NLLB-200]────► segments[].bn  (Bangla text)
        │
        ├─[MMS-TTS-ben]─► segments[].bn_audio  (Bangla speech)
        │
        ├─[kNN-VC]──────► voice-converted Bangla audio  (optional)
        │
        ├─[pyworld]─────► F0 + energy transferred from EN speaker  (optional)
        │
        ├─[rubberband]──► duration-aligned, pitch-preserved audio
        │
        └─[ffmpeg]──────► dubbed.mkv  (voice + background mixed)
```

---

## Installation

```bash
git clone https://github.com/Y3454R/OpenDub.git
cd OpenDub
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**macOS — rubberband backend (required for pitch-preserving time stretch):**
```bash
brew install rubberband
pip install pyrubberband
```

---

## Usage

**Fast test — skips voice cloning and prosody transfer:**
```bash
python cli.py \
  --input  test_files/clip.mkv \
  --srt    test_files/english.srt \
  --output outputs/dubbed/dubbed.mkv \
  --no-voice-clone \
  --no-prosody
```

**Full pipeline (requires kNN-VC checkpoint):**
```bash
python cli.py \
  --input         movie.mkv \
  --output        dubbed.mkv \
  --vc-checkpoint /path/to/knnvc_checkpoint
```

**Clip a time window:**
```bash
python cli.py \
  --input  movie.mkv \
  --start  00:23:00 \
  --end    00:27:00 \
  --output outputs/dubbed/clip.mkv \
  --no-voice-clone --no-prosody
```

**External SRT (skip subtitle extraction):**
```bash
python cli.py \
  --input  movie.mkv \
  --srt    subtitles.srt \
  --output dubbed.mkv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Input video file |
| `--output` | `dubbed.mkv` | Output video file |
| `--srt` | `None` | External SRT file; skips subtitle extraction from video |
| `--start` | `None` | Clip start time, e.g. `00:23:00` |
| `--end` | `None` | Clip end time, e.g. `00:27:00` |
| `--no-voice-clone` | off | Skip kNN-VC voice conversion |
| `--no-prosody` | off | Skip pyworld prosody transfer |
| `--vc-checkpoint` | `None` | Path to kNN-VC model checkpoint |
| `--output-dir` | `outputs` | Working directory for intermediate files |

---

## Modules

| Module | Model | Purpose |
|---|---|---|
| `audio/extractor.py` | ffmpeg | Extract mono 16kHz WAV + SRT from video |
| `audio/separator.py` | Demucs `htdemucs` | Split vocals from background music/effects |
| `subtitles/parser.py` | pysrt | Parse SRT, clean tags, slice per-segment audio |
| `translation/translator.py` | NLLB-200-distilled-600M | English → Bangla text translation |
| `tts/synthesizer.py` | MMS-TTS-ben | Bangla text → speech |
| `voice/cloning.py` | kNN-VC | Convert TTS voice to match source speaker |
| `voice/prosody.py` | pyworld (WORLD vocoder) | Transfer F0 contour and energy from EN speaker |
| `audio/stretcher.py` | rubberband / librosa TSM | Pitch-preserving duration alignment |
| `audio/mixer.py` | ffmpeg | Assemble segments, mix background, mux to video |

---

## Research context

This pipeline is a baseline for studying **isochrony-aware Bangla dubbing** — matching dubbed speech duration to the original without degrading naturalness.

Key open problems:
- **Voice cloning for Bangla** — kNN-VC was trained on English; cross-lingual transfer to Bangla TTS output is untested
- **Isochrony** — Bangla text is typically longer than its English source; the isochrony report printed after each run quantifies the gap per segment
- **Prosody transfer** — F0 contour from the English speaker is transferred via WORLD vocoder; interaction with Bangla tone patterns is an open question

The isochrony report at the end of each run shows mean/std/max duration mismatch across segments — this is the metric the research aims to minimize.

---

## Output structure

```
outputs/
├── segments/
│   ├── english.wav
│   ├── vocals.wav
│   ├── background.wav
│   ├── seg_0001_en.wav
│   ├── seg_0001_bn.wav
│   └── ...
└── dubbed/
    └── dubbed.mkv
```

---

## Requirements

- Python 3.10+
- ffmpeg (system install: `brew install ffmpeg` / `apt install ffmpeg`)
- rubberband (macOS: `brew install rubberband`)
- CUDA GPU recommended for full pipeline; M-series Apple Silicon works for development
