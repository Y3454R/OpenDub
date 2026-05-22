import os
import numpy as np
import soundfile as sf
import librosa
import torch
from torchaudio.transforms import Resample


def separate(audio_path: str, output_dir: str, model_name: str = "htdemucs") -> tuple[str, str]:
    """Separate vocals from background using Demucs Python API.

    Avoids torchaudio.load/save entirely (both require torchcodec in torchaudio>=2.6).
    Uses soundfile + librosa for I/O, torchaudio.transforms.Resample for resampling.
    Returns (vocals_path, background_path).
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    os.makedirs(output_dir, exist_ok=True)

    # Demucs MPS support is unreliable — use CPU on Apple Silicon
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Loading Demucs ({model_name}) on {device}...")

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Load with soundfile → torch tensor [channels, samples]
    data, sr = sf.read(audio_path, always_2d=True)  # [samples, channels]
    waveform = torch.from_numpy(data.T).float()      # [channels, samples]

    # Resample to demucs native rate (44100) using pure tensor transform
    if sr != model.samplerate:
        waveform = Resample(sr, model.samplerate)(waveform)

    # Mono → stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    print("Separating vocals from background...")
    with torch.no_grad():
        sources = apply_model(
            model,
            waveform[None].to(device),
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )[0]  # [num_sources, channels, samples]

    source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    vocals_idx   = source_names.index("vocals")
    vocals       = sources[vocals_idx]          # [2, samples]

    # Background = sum of all non-vocal stems
    bg_stems = [sources[i] for i in range(len(source_names)) if i != vocals_idx]
    background = torch.stack(bg_stems).sum(dim=0)  # [2, samples]

    def save_mono_16k(tensor: torch.Tensor, path: str):
        mono = tensor.mean(dim=0).cpu().numpy().astype(np.float32)  # stereo → mono
        mono_16k = librosa.resample(mono, orig_sr=model.samplerate, target_sr=16000)
        sf.write(path, mono_16k, 16000)

    vocals_path = os.path.join(output_dir, "vocals.wav")
    bg_path     = os.path.join(output_dir, "background.wav")
    save_mono_16k(vocals,     vocals_path)
    save_mono_16k(background, bg_path)

    print(f"Vocals saved:     {vocals_path}")
    print(f"Background saved: {bg_path}")
    return vocals_path, bg_path
