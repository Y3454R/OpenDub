import argparse
from opendub.pipeline import Pipeline

parser = argparse.ArgumentParser(description="OpenDub — English to Bangla video dubbing")
parser.add_argument("--input",         required=True,             help="Input video (.mkv, .mp4, …)")
parser.add_argument("--output",        default="dubbed.mkv",      help="Output video file")
parser.add_argument("--srt",           default=None,              help="External SRT file (skips subtitle extraction)")
parser.add_argument("--start",         default=None,              help="Clip start time, e.g. 00:23:00")
parser.add_argument("--end",           default=None,              help="Clip end time,   e.g. 00:27:00")
parser.add_argument("--no-voice-clone", action="store_true",      help="Skip voice cloning (faster, for testing)")
parser.add_argument("--no-prosody",     action="store_true",      help="Skip prosody transfer")
parser.add_argument("--vc-checkpoint",  default=None,             help="Path to kNN-VC checkpoint")
parser.add_argument("--output-dir",     default="outputs",        help="Working directory for intermediate files")
args = parser.parse_args()

pipeline = Pipeline(
    voice_cloning    = not args.no_voice_clone,
    prosody_transfer = not args.no_prosody,
    vc_checkpoint    = args.vc_checkpoint,
    output_dir       = args.output_dir,
)
pipeline.run(args.input, args.output, args.start, args.end, args.srt)
