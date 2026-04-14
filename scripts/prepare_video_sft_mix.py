#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

CATEGORY_ORDER = [
    'detailed_description',
    'video_qa',
    'long_video_summary',
]


def load_items(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f'{path} must contain a JSON list')
    return data


def estimate_item_bytes(item, video_root: Path):
    video_rel = item['video']
    video_path = video_root / video_rel
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    return video_path.stat().st_size


def select_items(items, target_bytes, video_root: Path, rng):
    items = list(items)
    rng.shuffle(items)
    selected = []
    total = 0
    for item in items:
        size_bytes = estimate_item_bytes(item, video_root)
        if total >= target_bytes:
            break
        enriched = dict(item)
        enriched['_video_bytes'] = size_bytes
        selected.append(enriched)
        total += size_bytes
    return selected, total


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-root', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--detailed-gb', type=float, default=6.0)
    parser.add_argument('--qa-gb', type=float, default=3.0)
    parser.add_argument('--summary-gb', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    video_root = Path(args.video_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    quotas = {
        'detailed_description': int(args.detailed_gb * 1024**3),
        'video_qa': int(args.qa_gb * 1024**3),
        'long_video_summary': int(args.summary_gb * 1024**3),
    }
    rng = random.Random(args.seed)

    mix = []
    summary = {}
    for category in CATEGORY_ORDER:
        src = processed_root / f'{category}.json'
        if not src.exists():
            raise FileNotFoundError(src)
        items = load_items(src)
        selected, total = select_items(items, quotas[category], video_root, rng)
        for item in selected:
            item['subset_category'] = category
        mix.extend(selected)
        summary[category] = {
            'source_file': str(src),
            'target_gb': quotas[category] / 1024**3,
            'selected_items': len(selected),
            'selected_gb': round(total / 1024**3, 4),
        }

    for idx, item in enumerate(mix):
        item.setdefault('id', f'mix-{idx:07d}')
        item.pop('_video_bytes', None)

    write_json(output_root / 'train.json', mix)
    write_json(output_root / 'mix_summary.json', {
        'video_root': str(video_root),
        'processed_root': str(processed_root),
        'total_items': len(mix),
        'categories': summary,
    })

    print(f'total_items={len(mix)}')
    for key, value in summary.items():
        print(key, value)
    print(output_root / 'train.json')


if __name__ == '__main__':
    main()
