#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

CATEGORIES = ['detailed_description', 'video_qa', 'long_video_summary']


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--dataset', action='append', nargs=3, metavar=('NAME','PROCESSED_ROOT','VIDEO_ROOT'), required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    videos_root = output_root / 'videos'
    processed_root = output_root / 'processed'
    videos_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    merged = {k: [] for k in CATEGORIES}
    meta = []

    for name, processed_dir, video_dir in args.dataset:
        processed_dir = Path(processed_dir).resolve()
        video_dir = Path(video_dir).resolve()
        link_path = videos_root / name
        if not link_path.exists():
            os.symlink(video_dir, link_path)

        for category in CATEGORIES:
            src = processed_dir / f'{category}.json'
            if not src.exists():
                continue
            items = load_json(src)
            for item in items:
                item = dict(item)
                item['video'] = f'{name}/{item["video"]}'
                merged[category].append(item)
        meta.append({'name': name, 'processed_root': str(processed_dir), 'video_root': str(video_dir)})

    for category, items in merged.items():
        (processed_root / f'{category}.json').write_text(json.dumps(items, ensure_ascii=False, indent=2))

    (output_root / 'merge_meta.json').write_text(json.dumps({'datasets': meta}, ensure_ascii=False, indent=2))
    print(output_root)
    for category, items in merged.items():
        print(category, len(items))


if __name__ == '__main__':
    main()
