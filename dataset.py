"""Audit manifests and connect the project's loader to the pinned YOLO trainer.

Grouping is implemented in plan_group_splits.py; this module never resplits data.
Images stay unchanged. Runtime labels/cache live in a new staging directory.
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from runtime import NAMES, SPLITS, digest, load_config, read_json, write_json


def read_manifests(prepared, cfg):
    prepared = Path(prepared)
    rows, seen, hashes = {}, set(), {}
    for split in SPLITS:
        path = prepared / f'{split}.csv'
        hashes[split] = digest(path)
        if hashes[split] != cfg['manifest_sha256'][split]:
            raise ValueError(f'Manifest hash differs: {split}')
        with path.open(newline='') as f:
            rows[split] = list(csv.DictReader(f))
        counts = Counter()
        for row in rows[split]:
            i, c = row['image_id'], int(row['class_id'])
            if i in seen or Path(i).name != i or not i or c not in NAMES:
                raise ValueError('Duplicate/invalid image ID or class')
            seen.add(i)
            if row['diagnosis'] != NAMES[c] or not row['lesion_id']:
                raise ValueError('Unknown lesion or inconsistent diagnosis')
            box = [float(row[k]) for k in ('center_x', 'center_y', 'box_width', 'box_height')]
            x, y, w, h = box
            if (not all(map(math.isfinite, box)) or min(w, h) <= 0
                    or min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7):
                raise ValueError('Invalid normalized bounding box')
            label = (prepared / split / 'labels' / f'{i}.txt').read_text().split()
            if len(label) != 5 or int(label[0]) != c or list(map(float, label[1:])) != box:
                raise ValueError(f'Label differs from audited manifest: {i}')
            counts[NAMES[c]] += 1
        if dict(counts) != cfg['split_class_counts'][split]:
            raise ValueError(f'Unexpected cohort counts: {split}')
    for field in ('lesion_id', 'patient_id'):
        groups = [{r[field] for r in rows[s] if r[field]} for s in SPLITS]
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f'Known {field} crosses splits')
    return rows, hashes


def stage_dataset(prepared, config, output, image_root=None):
    """Decode every image; reject failures instead of silently dropping samples."""
    from PIL import Image
    cfg, output = load_config(config), Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new staging directory')
    rows, hashes = read_manifests(prepared, cfg)
    files, pixel_hash_splits = {}, {}
    for split in SPLITS:
        for row in rows[split]:
            source = Path(row['image_path'])
            if image_root:
                source = Path(image_root) / source.name
            source = source.resolve(strict=True)
            if source.stem != row['image_id']:
                raise ValueError('Source filename does not match ID')
            with Image.open(source) as im:
                im.load()
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('Resolve nontrivial EXIF orientation before using the existing boxes')
                rgb = im.convert('RGB')
                import hashlib
                pixel_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                size = list(im.size)
            if pixel_hash in pixel_hash_splits and pixel_hash_splits[pixel_hash] != split:
                raise ValueError('Exact decoded image duplicate crosses splits; review before training')
            pixel_hash_splits[pixel_hash] = split
            files[row['image_id']] = {'source': str(source), 'size': size,
                                      'file_sha256': digest(source), 'pixel_sha256': pixel_hash}
    output.mkdir(parents=True)
    for split in SPLITS:
        for part in ('images', 'labels'):
            (output / split / part).mkdir(parents=True)
        for row in rows[split]:
            i = row['image_id']; src = Path(files[i]['source'])
            (output / split / 'images' / src.name).symlink_to(src)
            label = Path(prepared) / split / 'labels' / f'{i}.txt'
            (output / split / 'labels' / f'{i}.txt').write_bytes(label.read_bytes())
        # Separate evaluation YAMLs; the training YAML has no reassessment/test key.
    import yaml
    common = {'path': str(output), 'train': 'train/images', 'names': NAMES, 'nc': 2}
    for name, val in [('trainval', 'val'), ('reassessment', 'reassessment')]:
        (output / f'{name}.yaml').write_text(yaml.safe_dump({**common, 'val': f'{val}/images'}))
    audit = {'manifest_sha256': hashes, 'config_sha256': digest(config), 'files': files,
             'counts': {s: len(rows[s]) for s in SPLITS}, 'images_modified': 0,
             'fresh_blind_test': False, 'duplicate_check_proves_patient_independence': False}
    write_json(output / 'audit.json', audit)
    return audit


def verify_staged(stage, prepared, cfg, splits=('train', 'val')):
    """Recheck labels and image bytes before loading; reject stale/wrong staging."""
    rows, hashes = read_manifests(prepared, cfg)
    stage = Path(stage); audit = read_json(stage / 'audit.json')
    if audit['manifest_sha256'] != hashes:
        raise ValueError('Staged cohort differs')
    for split in splits:
        wanted = {r['image_id'] for r in rows[split]}
        images = {p.stem: p for p in (stage / split / 'images').iterdir() if p.is_file()}
        if set(images) != wanted or len(list((stage / split / 'images').iterdir())) != len(wanted):
            raise ValueError('Extra/missing runtime images')
        if {p.stem for p in (stage / split / 'labels').glob('*.txt')} != wanted:
            raise ValueError('Extra/missing runtime labels')
        for i, image in images.items():
            if digest(image) != audit['files'][i]['file_sha256']:
                raise ValueError('Source image changed after staging')
            if (stage / split / 'labels' / f'{i}.txt').read_bytes() != (Path(prepared) / split / 'labels' / f'{i}.txt').read_bytes():
                raise ValueError('Runtime label changed')
    return rows, hashes


def get_dataloader(trainer, path, batch_size=16, rank=-1, mode='train'):
    """Used by ProjectTrainer, with upstream transforms/collation explicitly retained."""
    from ultralytics.data import build_dataloader
    from preprocessing import build_project_dataset
    from ultralytics.utils.torch_utils import unwrap_model
    if mode not in ('train', 'val') or rank not in (-1, 0):
        raise ValueError('This protocol supports single-device train/val loading only')
    stride = max(int(unwrap_model(trainer.model).stride.max()), 32)
    data = build_project_dataset(trainer.args, path, batch_size, trainer.data,
                                 mode=mode, stride=stride)
    # Optional installed Albumentations must not silently add blur/CLAHE/etc.
    data.transforms.transforms = [t for t in data.transforms.transforms
                                  if type(t).__name__ != 'Albumentations']
    loader = build_dataloader(data, batch=batch_size, workers=trainer.args.workers,
                              shuffle=mode == 'train', rank=rank, drop_last=False)
    with (trainer.save_dir / 'project_loader_calls.jsonl').open('a') as f:
        import json
        f.write(json.dumps({'mode': mode, 'images': len(data), 'batch': batch_size,
                            'augment': data.augment, 'loader': 'dataset.get_dataloader',
                            'augmentation_parameters': {k: getattr(trainer.args, k) for k in
                                ('mosaic', 'mixup', 'cutmix', 'hsv_h', 'hsv_s', 'hsv_v', 'flipud', 'fliplr')},
                            'transforms': [type(t).__name__ for t in data.transforms.transforms]})+'\n')
    return loader


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--image-root', type=Path)
    a = p.parse_args()
    result = stage_dataset(a.prepared, a.config, a.output, a.image_root)
    print({k: v for k, v in result.items() if k != 'files'})
