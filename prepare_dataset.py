"""Materialise verified two-class annotations for the restart, without images.

This preparation does not train, load fitted weights or run inference. It reuses
verified official-mask bounding boxes, validates the complete known-lesion cohort,
and writes only NEW manifests/labels. School image paths are references; their
existence and decoding must be checked on the school server before training.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

CLASSES = {'Benign': 0, 'Malignant': 1}
SPLITS = ('train', 'val', 'reassessment')
BOX_FIELDS = ('center_x', 'center_y', 'box_width', 'box_height')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path, key):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if key not in (reader.fieldnames or []):
            raise ValueError('Missing key column')
        rows = list(reader)
    result = {}
    for row in rows:
        identifier = row[key]
        if not identifier or identifier in result:
            raise ValueError('Empty or duplicate image ID')
        result[identifier] = row
    return result


def prepare(source_file, protocol_file, output):
    source_file, protocol_file, output = map(Path, (source_file, protocol_file, output))
    if output.exists():
        raise ValueError('Use a new output directory; preserve prior versions')
    sources = json.loads(source_file.read_text())
    protocol = json.loads(protocol_file.read_text())
    if protocol.get('classes') != {'0': 'Benign', '1': 'Malignant'}:
        raise ValueError('Restart requires the declared two-class mapping')
    paths = {name: Path(sources[name]['path']) for name in
             ('official_metadata', 'official_mask_boxes', 'candidate_manifest')}
    paths = {name: (path if path.is_absolute() else source_file.resolve().parent / path)
             for name, path in paths.items()}
    for name, path in paths.items():
        if output.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('Keep outputs outside source directories')
        if digest(path) != sources[name]['sha256']:
            raise ValueError(f'Source hash differs: {name}')
    before = {str(p): digest(p) for p in [source_file, protocol_file, *paths.values()]}
    meta = read_rows(paths['official_metadata'], 'isic_id')
    boxes = read_rows(paths['official_mask_boxes'], 'image_id')
    manifest = read_rows(paths['candidate_manifest'], 'image_id')
    if set(boxes) != set(meta):
        raise ValueError('Full metadata and verified box IDs must match')
    expected = {i for i, r in meta.items() if r['lesion_id'].strip()}
    if set(manifest) != expected:
        raise ValueError('Use all and only known-lesion images; no outcome-based selection')
    selected = {s: [] for s in SPLITS}
    for i in sorted(manifest):
        row, m, box = manifest[i], meta[i], boxes[i]
        category, split = m['diagnosis_1'], row['candidate_split']
        if category not in CLASSES or row['diagnosis'] != category or split not in selected:
            raise ValueError('Diagnosis or split mismatch')
        if any(row[f] != m[f] for f in ('lesion_id', 'patient_id')):
            raise ValueError('Do not change or impute recorded identifiers')
        if Path(box['image_path']).stem != i:
            raise ValueError('Image path does not match image ID')
        x, y, w, h = (float(box[k]) for k in BOX_FIELDS)
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0
                or min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7):
            raise ValueError('Invalid normalised bounding box')
        selected[split].append(i)
    overlap = {}
    for field in ('lesion_id', 'patient_id', 'assignment_component'):
        groups = {s: {manifest[i][field] for i in selected[s] if manifest[i][field]} for s in SPLITS}
        overlap[field] = {f'{a}_{b}': len(groups[a] & groups[b])
                          for a, b in itertools.combinations(SPLITS, 2)}
        if any(overlap[field].values()):
            raise ValueError(f'Known grouping crosses splits: {field}')
    counts = {}
    for split, ids in selected.items():
        class_counts = Counter(meta[i]['diagnosis_1'] for i in ids)
        counts[split] = {'images': len(ids), **{c: class_counts[c] for c in CLASSES}}
        wanted = protocol['split_roles'][split]
        if any(counts[split][k] != wanted[k] for k in ('images', *CLASSES)):
            raise ValueError('Counts differ from the declared protocol')
    output.mkdir(parents=True, exist_ok=False)
    fields = ['image_id', 'image_path', 'class_id', 'diagnosis', *BOX_FIELDS,
              'lesion_id', 'patient_id', 'historical_split']
    for split, ids in selected.items():
        labels = output / split / 'labels'
        labels.mkdir(parents=True)
        with (output / f'{split}.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in ids:
                category, box = meta[i]['diagnosis_1'], boxes[i]
                writer.writerow({'image_id': i, 'image_path': box['image_path'],
                                 'class_id': CLASSES[category], 'diagnosis': category,
                                 **{k: box[k] for k in BOX_FIELDS},
                                 'lesion_id': meta[i]['lesion_id'], 'patient_id': meta[i]['patient_id'],
                                 'historical_split': manifest[i]['historical_split']})
                (labels / f'{i}.txt').write_text(str(CLASSES[category])+' '+
                                                ' '.join(box[k] for k in BOX_FIELDS)+'\n')
    if before != {p: digest(p) for p in before}:
        raise RuntimeError('A source changed during preparation')
    summary = {'status': 'prepared_labels_only_no_training', 'classes': CLASSES,
               'counts': counts, 'known_group_overlap': overlap,
               'source_sha256': before, 'images_copied_or_modified': 0,
               'source_boxes_recomputed': False, 'old_weights_loaded': False,
               'server_image_existence_verified': False, 'course_acceptance_confirmed': False,
               'patient_independent_claim': False, 'fresh_blind_test': False,
               'missing_patient_ids': sum(not meta[i]['patient_id'] for i in manifest),
               'historical_membership': {s: dict(Counter(manifest[i]['historical_split'] for i in ids))
                                          for s, ids in selected.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.protocol, args.output), indent=2))
