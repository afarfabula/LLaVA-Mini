#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def load_annotations(annotation_path: Path):
    if annotation_path.suffix == ".json":
        data = json.loads(annotation_path.read_text())
        if isinstance(data, dict):
            return data
        raise ValueError("JSON annotation file must be a path->caption dict.")

    mapping = {}
    with annotation_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            path = item.get("path") or item.get("video") or item.get("file_name")
            caption = item.get("caption") or item.get("text") or item.get("response")
            if not path or not caption:
                raise ValueError("Each JSONL line must contain path/video/file_name and caption/text/response.")
            mapping[path] = caption
    return mapping


def iter_videos(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video-root", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument(
        "--prompt",
        default="<video>\nPlease describe the video content in detail.",
        help="Human prompt used for all samples.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_video_root).resolve()
    annotation_file = Path(args.annotation_file).resolve()
    output_root = Path(args.output_root).resolve()
    output_video_root = output_root / "videos"
    output_json = output_root / "train.json"
    output_meta = output_root / "subset_meta.json"

    output_video_root.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(annotation_file)

    candidates = []
    for video_path in iter_videos(source_root):
        rel = video_path.relative_to(source_root).as_posix()
        caption = annotations.get(rel) or annotations.get(video_path.name)
        if caption:
            candidates.append((video_path, rel, caption, video_path.stat().st_size))

    if not candidates:
        raise SystemExit("No videos matched the provided annotations.")

    random.Random(args.seed).shuffle(candidates)
    target_bytes = int(args.target_gb * 1024 * 1024 * 1024)

    chosen = []
    total_bytes = 0
    for item in candidates:
        chosen.append(item)
        total_bytes += item[3]
        if total_bytes >= target_bytes:
            break

    records = []
    for idx, (src, rel, caption, size_bytes) in enumerate(chosen):
        dst = output_video_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)

        records.append(
            {
                "id": f"video-sft-{idx:06d}",
                "video": rel,
                "conversations": [
                    {"from": "human", "value": args.prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        )

    output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    output_meta.write_text(
        json.dumps(
            {
                "source_video_root": str(source_root),
                "annotation_file": str(annotation_file),
                "output_root": str(output_root),
                "target_gb": args.target_gb,
                "selected_videos": len(chosen),
                "selected_bytes": total_bytes,
                "selected_gb": round(total_bytes / 1024 / 1024 / 1024, 4),
                "copy_mode": args.copy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"selected_videos={len(chosen)}")
    print(f"selected_gb={total_bytes / 1024 / 1024 / 1024:.4f}")
    print(f"train_json={output_json}")
    print(f"video_root={output_video_root}")


if __name__ == "__main__":
    main()
