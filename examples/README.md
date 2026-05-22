# Examples

Place a sample `.srt` file here (e.g. `sample.srt`) to test the pipeline
without running a full video extraction.

```bash
# Quick test — skip voice cloning and prosody, use an external SRT
python cli.py \
  --input   /path/to/movie.mkv \
  --srt     examples/sample.srt \
  --output  outputs/dubbed/test.mkv \
  --no-voice-clone \
  --no-prosody
```
