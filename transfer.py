"""Build and verify a local project transfer bundle; never upload or train.

The archive contains code, configuration, labels, provenance and report figures,
not the source photograph/mask collection or model weights. SHA-256 checks detect changed bytes, not scientific validity
or authenticity against an attacker who also replaces the manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from dataset import read_manifests
from runtime import digest, load_config

MANIFEST = 'TRANSFER_MANIFEST.json'
CODE = Path('recognition/YOLO_s4905914')


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts
            or '\\' in name or str(path) != name):
        raise ValueError(f'Invalid relative bundle path: {name}')
    return path


def regular_file(root, name):
    path = root / Path(safe_relative(name))
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f'Bundle inputs must not be symlinks: {name}')
        cursor = cursor.parent
    if not path.is_file():
        raise ValueError(f'Missing bundle file: {name}')
    return path


def collect_files(root):
    cfg = load_config(root / CODE / 'experiment.json')
    prepared = Path('data/known_lesion_v1')
    rows, _ = read_manifests(root / prepared, cfg)
    paths = [Path(n) for n in ('START_HERE_zh.txt', 'TRANSFER_GUIDE_zh.txt',
             'PROTOCOL.json', 'sources.json', 'MODEL_COMPONENTS_zh.txt',
             'TEACHER_QUESTIONS.txt', 'PREPARATION_VERIFICATION.json',
             'IMPLEMENTATION_VERIFICATION.json', 'CLEANUP_PLAN.txt')]
    paths += [Path(name) for name in ('FEASIBILITY_REVIEW.txt', 'READINESS_STATUS_zh.txt',
                                     'COURSE_REQUIREMENTS_AUDIT_zh.txt',
                                     'SUBMISSION_DECISION_zh.txt', 'PRE_SUBMISSION_CHECK_zh.txt',
                                     'PRE_SUBMISSION_VERIFICATION.json',
                                     'TEACHER_CONFIRMATION_EN.txt') if (root / name).is_file()]
    paths += [p.relative_to(root) for p in sorted((root / CODE).iterdir())
              if p.is_file() and not p.name.startswith('.')
              and p.suffix in ('.py', '.json', '.txt', '.md', '.sh')]
    paths += [p.relative_to(root) for p in sorted((root / CODE / 'assets').glob('*.svg'))]
    paths += [p.relative_to(root) for p in sorted((root / CODE / 'assets').glob('*.png'))]
    snapshots = Path('data/source_snapshots')
    sources = json.loads((root / 'sources.json').read_text())
    for source in sources.values():
        relative = Path(safe_relative(source['path']))
        if not relative.is_relative_to(snapshots):
            raise ValueError('Transfer sources must live inside data/source_snapshots')
        if digest(regular_file(root, relative.as_posix())) != source['sha256']:
            raise ValueError('Source snapshot checksum differs')
        paths.append(relative)
    paths += [snapshots / 'PROVENANCE.json']
    paths += [snapshots / 'historical_splits' / f'{s}.csv' for s in ('train', 'val', 'test')]
    paths += [prepared / 'summary.json']
    for split, records in rows.items():
        paths.append(prepared / f'{split}.csv')
        paths += [prepared / split / 'labels' / f"{r['image_id']}.txt" for r in records]
    return sorted({p.as_posix() for p in paths})


def build_bundle(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checksum = output.with_suffix('.zip.sha256')
    if output.suffix != '.zip' or output.exists() or checksum.exists():
        raise ValueError('Choose an unused .zip output and checksum path')
    payload = {name: regular_file(root, name).read_bytes() for name in collect_files(root)}
    manifest = {'format_version': 1, 'purpose': 'project_transfer_not_final_submission',
                'contains_source_dataset_images_masks_or_weights': False,
                'contains_report_figures': True,
                'starts_training_or_uploads': False,
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                          for name, data in payload.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x') as archive:
        for name, data in {**payload, MANIFEST: (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()}.items():
            info = zipfile.ZipInfo(name)  # Fixed ZIP date: unchanged inputs give identical archive bytes.
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    result = verify_bundle(output)
    checksum.write_text(f'{digest(output)}  {output.name}\n')
    return {**result, 'archive': str(output), 'archive_bytes': output.stat().st_size,
            'archive_sha256': digest(output), 'checksum_file': str(checksum)}


def checked_manifest(data):
    value = json.loads(data)
    if value.get('format_version') != 1 or not isinstance(value.get('files'), dict) or not value['files']:
        raise ValueError('Unrecognized or empty transfer manifest')
    for name in value['files']:
        safe_relative(name)
        if name == MANIFEST:
            raise ValueError('Manifest must not list itself')
    return value


def check_bytes(name, data, expected):
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError(f'File differs from transfer manifest: {name}')


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries')
        manifest = checked_manifest(archive.read(MANIFEST))
        if set(names) != set(manifest['files']) | {MANIFEST}:
            raise ValueError('Archive has missing or unlisted files')
        for name, expected in manifest['files'].items():
            check_bytes(name, archive.read(name), expected)
    return {'status': 'verified', 'payload_files': len(manifest['files']),
            'checks': 'Exact archive membership and each payload size/SHA-256'}


def verify_directory(root):
    root = Path(root).resolve()
    manifest = checked_manifest(regular_file(root, MANIFEST).read_bytes())
    for name, expected in manifest['files'].items():
        check_bytes(name, regular_file(root, name).read_bytes(), expected)
    return {'status': 'verified', 'payload_files': len(manifest['files']),
            'checks': 'Listed extracted files only; extra runtime outputs are ignored'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    build.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('verify')
    choice = verify.add_mutually_exclusive_group(required=True)
    choice.add_argument('--bundle', type=Path)
    choice.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_bundle(args.project_root, args.output)
    elif args.bundle:
        result = verify_bundle(args.bundle)
    else:
        result = verify_directory(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=False))
