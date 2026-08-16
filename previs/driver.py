"""Entry point executed inside Blender.

Invoked as::

    blender --background --factory-startup --python previs/driver.py -- \
        <shot.json> --out <video.mp4> [--assets <dir>]

Everything after the bare ``--`` is ours; Blender consumes the rest.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Blender starts with its own sys.path, so make the project importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from previs.compiler import compile_camera_path, compile_shot  # noqa: E402
from previs.schema import load_shot  # noqa: E402


def parse_args(argv):
    args = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    parser = argparse.ArgumentParser(prog="previs-driver")
    parser.add_argument("shot", help="path to a shot JSON file")
    parser.add_argument("--out", required=True, help="output video path")
    parser.add_argument("--assets", default=None, help="assets root directory")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--preview-frame", type=int, default=None,
        help="render only this one frame (a PNG, not the video) plus a scene manifest",
    )
    parser.add_argument(
        "--camera-path", default=None, choices=("top", "angle"),
        help="render the camera's trajectory as a visible trail from an external "
        "static camera, instead of the shot's own animated camera",
    )
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv)
    try:
        shot = load_shot(args.shot)
        if args.camera_path:
            compile_camera_path(
                shot, args.out, assets_root=args.assets, verbose=not args.quiet,
                mode=args.camera_path,
            )
            return
        compile_shot(
            shot, args.out, assets_root=args.assets, verbose=not args.quiet,
            preview_frame=args.preview_frame,
        )
    except Exception:
        # Blender swallows a bare traceback's exit code, so be explicit.
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
