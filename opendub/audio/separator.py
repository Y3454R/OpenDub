import os
import subprocess
import shutil


def separate(audio_path: str, output_dir: str) -> tuple[str, str]:
    """Split an audio file into vocals and background using Demucs (htdemucs model).

    Returns (vocals_path, background_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "--out", output_dir,
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed:\n{result.stderr[-800:]}")

    # Demucs writes to <output_dir>/htdemucs/<stem>/<filename>
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_dir  = os.path.join(output_dir, "htdemucs", stem)
    raw_vocals  = os.path.join(demucs_dir, "vocals.wav")
    raw_no_vocal = os.path.join(demucs_dir, "no_vocals.wav")

    vocals_path = os.path.join(output_dir, "vocals.wav")
    bg_path     = os.path.join(output_dir, "background.wav")
    shutil.copy2(raw_vocals,   vocals_path)
    shutil.copy2(raw_no_vocal, bg_path)

    return vocals_path, bg_path
