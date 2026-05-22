import os
import subprocess
import numpy as np
import librosa
import soundfile as sf


def assemble_audio(
    segments: list[dict],
    total_duration: float,
    sample_rate: int,
    output_path: str,
) -> str:
    """Place each segment's dubbed audio onto a timeline and write a WAV file.

    Overlapping segments are summed; the result is peak-normalized.
    Returns output_path.
    """
    timeline = np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    for seg in segments:
        if not os.path.exists(seg.get("bn_audio", "")):
            continue
        y, _ = librosa.load(seg["bn_audio"], sr=sample_rate)
        if len(y) == 0:
            continue
        start = int(seg["start"] * sample_rate)
        end   = start + len(y)
        if end > len(timeline):
            y   = y[: len(timeline) - start]
            end = len(timeline)
        timeline[start:end] += y

    peak = np.max(np.abs(timeline))
    if peak > 0:
        timeline /= peak

    sf.write(output_path, timeline, sample_rate)
    return output_path


def mix_with_video(
    video_path: str,
    dubbed_audio_path: str,
    background_audio_path: str | None,
    output_path: str,
    bg_volume: float = 0.15,
) -> str:
    """Mux dubbed voice + optional background music back into the video.

    If background_audio_path is None the dubbed audio is used as-is.
    Returns output_path.
    """
    if background_audio_path and os.path.exists(background_audio_path):
        # Mix dubbed voice with background at reduced volume
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-i", background_audio_path,
            "-filter_complex",
            f"[1:a]volume=1.0[dub];[2:a]volume={bg_volume}[bg];[dub][bg]amix=inputs=2:duration=first[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-800:]}")
    return output_path


def isochrony_report(segments: list[dict]) -> None:
    diffs = [s["bn_dur"] - s["dur"] for s in segments if "bn_dur" in s]
    if not diffs:
        return

    import numpy as np
    print("\n── Isochrony Report ──────────────────────────────")
    print(f"Segments processed  : {len(diffs)}")
    print(f"Mean duration diff  : {np.mean(diffs):+.3f}s")
    print(f"Std duration diff   : {np.std(diffs):.3f}s")
    print(f"Max over            : {max(diffs):+.3f}s")
    print(f"Max under           : {min(diffs):+.3f}s")

    worst = sorted(
        [s for s in segments if "bn_dur" in s],
        key=lambda x: abs(x["bn_dur"] - x["dur"]),
        reverse=True,
    )[:3]
    print("\nWorst isochrony segments:")
    for s in worst:
        print(f"  Seg {s['id']:3d}: EN={s['dur']:.2f}s BN={s['bn_dur']:.2f}s | {s['en'][:50]}")
