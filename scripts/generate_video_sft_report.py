#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples-per-category', type=int, default=3)
    args = parser.parse_args()

    train_json = Path(args.train_json).resolve()
    output_md = Path(args.output_md).resolve()
    data = json.loads(train_json.read_text())

    rng = random.Random(args.seed)
    buckets = {}
    for item in data:
        category = item.get('subset_category', 'unknown')
        buckets.setdefault(category, []).append(item)

    lines = ['# Video SFT Data Samples', '']
    lines.append(f'- Train JSON: `{train_json}`')
    lines.append(f'- Total Samples: `{len(data)}`')
    lines.append('')

    for category in sorted(buckets):
        lines.append(f'## {category}')
        lines.append('')
        sample_items = list(buckets[category])
        rng.shuffle(sample_items)
        for idx, item in enumerate(sample_items[: args.samples_per_category], start=1):
            convs = item.get('conversations', [])
            human = convs[0]['value'] if len(convs) > 0 else ''
            answer = convs[1]['value'] if len(convs) > 1 else ''
            lines.append(f'### Sample {idx}')
            lines.append(f'- Video Path: `{item.get("video", "")}`')
            lines.append(f'- Prompt: `{human}`')
            lines.append(f'- Answer: `{answer}`')
            lines.append('')

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text('\n'.join(lines))
    print(output_md)


if __name__ == '__main__':
    main()
