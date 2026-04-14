#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

DEFAULT_DETAILED_PROMPTS = [
    '<video>\nDescribe the cooking step shown in this video segment in detail.',
    '<video>\nWhat is happening in this cooking clip?',
    '<video>\nExplain the action being performed in this cooking video segment.',
]

DEFAULT_QA_PROMPTS = [
    '<video>\nWhat is the person doing in this clip?',
    '<video>\nWhich cooking action is shown in the video?',
    '<video>\nDescribe the step being performed in this short video.',
]

DEFAULT_SUMMARY_PROMPTS = [
    '<video>\nSummarize this cooking video from beginning to end.',
    '<video>\nDescribe the full recipe process shown in the video.',
    '<video>\nWhat happens throughout this cooking video?',
]


def build_video_index(video_root: Path):
    index = {}
    by_video = {}
    for path in video_root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.mp4', '.webm', '.avi', '.mov', '.mkv'}:
            stem = path.stem
            rel = path.relative_to(video_root).as_posix()
            index.setdefault(stem, rel)
            if '_' in stem:
                video_id, seg_id = stem.rsplit('_', 1)
                if seg_id.isdigit():
                    by_video.setdefault(video_id, {})[int(seg_id)] = rel
    return index, by_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-json', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    annotation_json = Path(args.annotation_json).resolve()
    video_root = Path(args.video_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    raw = json.loads(annotation_json.read_text())['database']
    video_index, by_video = build_video_index(video_root)

    detailed = []
    qa = []
    summary = []
    missing = []

    for video_id, meta in raw.items():
        segment_map = by_video.get(video_id, {})
        if not segment_map:
            missing.append(video_id)
            continue

        anns = sorted(meta.get('annotations', []), key=lambda x: x['segment'][0])
        if not anns:
            continue

        joined_summary = ' '.join(x['sentence'].strip() for x in anns if x.get('sentence'))
        summary_video = segment_map.get(0) or segment_map.get(min(segment_map))
        if joined_summary and summary_video:
            summary.append({
                'id': f'youcook2-summary-{video_id}',
                'video': summary_video,
                'subset_category': 'long_video_summary',
                'dataset': 'YouCook2',
                'conversations': [
                    {'from': 'human', 'value': rng.choice(DEFAULT_SUMMARY_PROMPTS)},
                    {'from': 'gpt', 'value': joined_summary},
                ],
            })

        for ann in anns:
            sentence = ann.get('sentence', '').strip()
            if not sentence:
                continue
            ann_id = ann.get('id', 0)
            rel_video = segment_map.get(ann_id)
            if rel_video is None:
                continue
            detailed.append({
                'id': f'youcook2-detail-{video_id}-{ann_id}',
                'video': rel_video,
                'subset_category': 'detailed_description',
                'dataset': 'YouCook2',
                'segment': ann.get('segment'),
                'conversations': [
                    {'from': 'human', 'value': rng.choice(DEFAULT_DETAILED_PROMPTS)},
                    {'from': 'gpt', 'value': sentence},
                ],
            })
            qa.append({
                'id': f'youcook2-qa-{video_id}-{ann_id}',
                'video': rel_video,
                'subset_category': 'video_qa',
                'dataset': 'YouCook2',
                'segment': ann.get('segment'),
                'conversations': [
                    {'from': 'human', 'value': rng.choice(DEFAULT_QA_PROMPTS)},
                    {'from': 'gpt', 'value': sentence},
                ],
            })

    (output_root / 'detailed_description.json').write_text(json.dumps(detailed, ensure_ascii=False, indent=2))
    (output_root / 'video_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2))
    (output_root / 'long_video_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_root / 'normalization_stats.json').write_text(json.dumps({
        'annotation_json': str(annotation_json),
        'video_root': str(video_root),
        'matched_videos': len(video_index) - len(missing),
        'missing_videos': len(missing),
        'missing_video_ids': missing[:200],
        'detailed_samples': len(detailed),
        'qa_samples': len(qa),
        'summary_samples': len(summary),
    }, ensure_ascii=False, indent=2))

    print('detailed_samples', len(detailed))
    print('qa_samples', len(qa))
    print('summary_samples', len(summary))
    print('missing_videos', len(missing))


if __name__ == '__main__':
    main()
