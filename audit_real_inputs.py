"""Check real image dimensions, decoding and staging without model execution."""
import argparse
import csv
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import platform

from dataset import read_manifests, stage_dataset, verify_staged
from runtime import digest, load_config, read_json, write_json


def audit_inputs(config, prepared, sources, output, image_root=None):
    from PIL import Image
    output, sources = Path(output).resolve(), Path(sources).resolve()
    if output.exists():
        raise ValueError('Choose a new audit output directory')
    cfg = load_config(config)
    rows, hashes = read_manifests(prepared, cfg)
    source_records = read_json(sources)
    for entry in source_records.values():
        if digest(sources.parent / entry['path']) != entry['sha256']:
            raise ValueError('Source snapshot checksum differs')
    box_path = sources.parent / source_records['official_mask_boxes']['path']
    with box_path.open(newline='') as stream:
        box_rows = list(csv.DictReader(stream))
    boxes = {r['image_id']: r for r in box_rows}
    if len(boxes) != len(box_rows):
        raise ValueError('Duplicate ID in reference boxes')
    output.mkdir(parents=True)
    report = {'status': 'started', 'model_executed': False, 'images_modified': 0,
              'patient_independence_proven': False, 'fresh_blind_test': False,
              'config_sha256': digest(config), 'manifest_sha256': hashes,
              'box_source_sha256': digest(box_path),
              'runtime': {'python': platform.python_version(), 'host': platform.node(),
                          'Pillow': version('Pillow'), 'PyYAML': version('PyYAML')},
              'image_root_override': str(Path(image_root).resolve()) if image_root else None}
    try:
        for records in rows.values():
            for row in records:
                source = Path(row['image_path'])
                if image_root:
                    source = Path(image_root) / source.name
                reference = boxes[row['image_id']]
                with Image.open(source) as im:
                    if im.size != (int(reference['width']), int(reference['height'])):
                        raise ValueError(f"Image dimensions differ from box source: {row['image_id']}")
        stage = output / 'staged'
        staged = stage_dataset(prepared, config, stage, image_root)
        verify_staged(stage, prepared, cfg, splits=('train', 'val', 'reassessment'))
        pixel_counts = Counter(r['pixel_sha256'] for r in staged['files'].values())
        report.update(status='passed', counts=staged['counts'], decoded_images=len(staged['files']),
                      all_dimensions_match_box_source=True, exif_orientation_valid=True,
                      cross_split_exact_decoded_duplicates=0,
                      same_split_exact_duplicate_groups=sum(v > 1 for v in pixel_counts.values()),
                      staged_audit_sha256=digest(stage / 'audit.json'),
                      staging_directory=str(stage))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / 'input_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    args = parser.parse_args()
    print(audit_inputs(args.config, args.prepared, args.sources, args.output, args.image_root))
