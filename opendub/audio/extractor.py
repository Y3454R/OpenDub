import os
import subprocess


def extract(video_path: str, output_dir: str, start: str = None, end: str = None) -> tuple[str, str]:
    """Extract audio and subtitles from a video file using ffmpeg.

    Returns (audio_path, srt_path). srt_path is None if no subtitle stream found.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "english.wav")
    srt_path   = os.path.join(output_dir, "english.srt")

    time_args = []
    if start:
        time_args += ["-ss", start]
    if end:
        time_args += ["-to", end]

    # Extract mono 16kHz WAV
    cmd = ["ffmpeg", "-y", "-i", video_path] + time_args + [
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", audio_path
    ]
    _run(cmd, f"audio extraction from {video_path}")

    # Extract first subtitle stream (soft subs)
    cmd_sub = ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_path]
    try:
        _run(cmd_sub, "subtitle extraction")
    except RuntimeError:
        print("  No embedded subtitle stream found — provide an external .srt via --srt")
        srt_path = None

    return audio_path, srt_path


def _run(cmd: list[str], label: str):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({label}):\n{result.stderr[-800:]}")
