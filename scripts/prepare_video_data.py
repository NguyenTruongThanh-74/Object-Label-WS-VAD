from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def source_relative_path(source: Path, source_root: Path) -> Path:
    try:
        return source.resolve().relative_to(source_root.resolve())
    except ValueError:
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
        return Path("external") / digest / source.name


def prepare(input_csv: Path, source_root: Path, frames_root: Path, output_csv: Path,
            project_root: Path, ffmpeg: str, fps: float | None) -> None:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if not reader.fieldnames or not {"video", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{input_csv} must have 'video,label' columns")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No videos found in {input_csv}")
    output_rows = []
    seen_targets = set()
    for row in rows:
        video_value = (row.get("video") or "").strip()
        if not video_value:
            raise ValueError("Input manifest contains an empty video path")
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid integer label for video '{video_value}'") from error

        source = Path(video_value)
        if not source.is_absolute():
            source = source_root / source
        source = source.resolve()
        if source.is_dir():
            if not any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in source.iterdir()):
                raise ValueError(f"No image frames found in source directory: {source}")
            prepared_video = project_path(source, project_root)
        elif source.is_file():
            relative_video = source_relative_path(source, source_root)
            frame_dir = (frames_root / relative_video.with_suffix("")).resolve()
            if frame_dir in seen_targets:
                raise ValueError(f"Multiple input rows map to the same output directory: {frame_dir}")
            seen_targets.add(frame_dir)
            existing_frames = [
                path for path in frame_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ] if frame_dir.is_dir() else []
            if not existing_frames:
                if frame_dir.exists():
                    raise FileExistsError(f"Output directory exists but has no frames: {frame_dir}")
                frame_dir.mkdir(parents=True)
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source)]
                if fps is not None:
                    command.extend(["-vf", f"fps={fps:g}", "-vsync", "0"])
                command.extend(["-q:v", "2", str(frame_dir / "%06d.jpg")])
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    shutil.rmtree(frame_dir)
                    raise RuntimeError(f"FFmpeg failed for {source} with exit code {result.returncode}")
                existing_frames = [
                    path for path in frame_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ]
                if not existing_frames:
                    shutil.rmtree(frame_dir)
                    raise RuntimeError(f"FFmpeg produced no frames for {source}")
            prepared_video = project_path(frame_dir, project_root)
        else:
            raise FileNotFoundError(f"Input video or frame directory not found: {source}")

        output_rows.append({"video": prepared_video, "label": label})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["video", "label"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare video anomaly manifests and extract videos to frame folders with FFmpeg"
    )
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="CSV with video,label; video paths may point to videos or frame directories")
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="Base directory for relative paths in the input CSV")
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=float, default=None,
                        help="Optional extraction rate; by default preserve every source frame")
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be greater than zero")
    frames_root = args.frames_root
    if not frames_root.is_absolute():
        frames_root = args.project_root / frames_root
    prepare(
        args.input_csv,
        args.source_root,
        frames_root,
        args.output_csv,
        args.project_root,
        args.ffmpeg,
        args.fps,
    )


if __name__ == "__main__":
    main()