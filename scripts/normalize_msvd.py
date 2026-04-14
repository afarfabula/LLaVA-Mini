#!/usr/bin/env python3
import argparse
import json
import zipfile
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def load_mapping(path: Path):
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw_name, vid = line.split()
        mapping[vid] = raw_name + '.avi'
        if vid.startswith('vid') and vid[3:].isdigit():
            mapping[vid[3:]] = raw_name + '.avi'
    return mapping


def load_msvd_qa(zip_path: Path):
    qa = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        target = None
        for name in names:
            low = name.lower()
            if low.endswith('.json') and 'train' in low:
                target = name
                break
        if target is None:
            raise FileNotFoundError('No train json found in MSVD-QA.zip')
        data = json.loads(zf.read(target))
    for item in data:
        key = item.get('video_id') or item.get('video') or item.get('id')
        if key is None:
            continue
        qa.setdefault(str(key), []).append(item)
    return qa


def maybe_video_path(video_root: Path, rel_name: str):
    direct = video_root / rel_name
    if direct.exists():
        return rel_name
    matches = list(video_root.rglob(rel_name))
    if matches:
        return matches[0].relative_to(video_root).as_posix()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--val-json', required=True)
    parser.add_argument('--test-json', required=True)
    parser.add_argument('--msvd-qa-zip', required=True)
    parser.add_argument('--youtube-mapping', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    caption_items = load_json(Path(args.train_json)) + load_json(Path(args.val_json)) + load_json(Path(args.test_json))
    qa_map = load_msvd_qa(Path(args.msvd_qa_zip))
    id_mapping = load_mapping(Path(args.youtube_mapping))

    detailed = []
    summary = []
    video_qa = []
    missing = []

    for item in caption_items:
        rel_video = maybe_video_path(video_root, item['video'])
        if rel_video is None:
            missing.append(item['video'])
            continue
        captions = item.get('caption', [])
        if captions:
            detailed.append({
                'id': f"msvd-detail-{item['video_id']}",
                'video': rel_video,
                'subset_category': 'detailed_description',
                'dataset': 'MSVD',
                'conversations': [
                    {'from': 'human', 'value': '<video>\nDescribe this video in detail.'},
                    {'from': 'gpt', 'value': captions[0]},
                ],
            })
            summary.append({
                'id': f"msvd-summary-{item['video_id']}",
                'video': rel_video,
                'subset_category': 'long_video_summary',
                'dataset': 'MSVD',
                'conversations': [
                    {'from': 'human', 'value': '<video>\nSummarize this video from beginning to end.'},
                    {'from': 'gpt', 'value': ' '.join(captions[:5])},
                ],
            })

        qa_lookup_keys = [item['video_id']]
        mapped_vid = None
        for k, v in id_mapping.items():
            if v == item['video']:
                mapped_vid = k
                break
        if mapped_vid:
            qa_lookup_keys.append(mapped_vid)
            if isinstance(mapped_vid, str) and mapped_vid.startswith('vid') and mapped_vid[3:].isdigit():
                qa_lookup_keys.append(mapped_vid[3:])

        for qa_key in qa_lookup_keys:
            for qa_item in qa_map.get(qa_key, []):
                q = qa_item.get('question') or qa_item.get('q') or qa_item.get('question_text')
                a = qa_item.get('answer') or qa_item.get('a') or qa_item.get('answers')
                if isinstance(a, list):
                    a = a[0] if a else None
                if not q or not a:
                    continue
                video_qa.append({
                    'id': f"msvd-qa-{item['video_id']}-{len(video_qa)}",
                    'video': rel_video,
                    'subset_category': 'video_qa',
                    'dataset': 'MSVD-QA',
                    'conversations': [
                        {'from': 'human', 'value': f'<video>\n{q}'},
                        {'from': 'gpt', 'value': str(a)},
                    ],
                })

    for name, data in [
        ('detailed_description.json', detailed),
        ('video_qa.json', video_qa),
        ('long_video_summary.json', summary),
    ]:
        (output_root / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    (output_root / 'normalization_stats.json').write_text(json.dumps({
        'detailed_samples': len(detailed),
        'qa_samples': len(video_qa),
        'summary_samples': len(summary),
        'missing_videos': len(set(missing)),
    }, ensure_ascii=False, indent=2))

    print('detailed_samples', len(detailed))
    print('qa_samples', len(video_qa))
    print('summary_samples', len(summary))
    print('missing_videos', len(set(missing)))


if __name__ == '__main__':
    main()
