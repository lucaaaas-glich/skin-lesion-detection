"""Small provenance helpers shared by the command-line entry points."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

NAMES = {0: 'Benign', 1: 'Malignant'}
SPLITS = ('train', 'val', 'reassessment')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path):
    cfg = read_json(path)
    if cfg['classes'] != {'0': 'Benign', '1': 'Malignant'}:
        raise ValueError('Expected exactly Benign=0, Malignant=1')
    if cfg['models'] != read_json(Path(__file__).with_name('weights_registry.json')):
        raise ValueError('Use only the recorded official n/s initialization assets')
    train, ev = cfg['train'], cfg['evaluation']
    if ev.get('preprocessing') != 'single_letterbox_v1':
        raise ValueError('Use the current evaluation preprocessing protocol')
    if train['optimizer'] == 'auto' or train['resume'] or train['single_cls']:
        raise ValueError('Explicit optimizer, fresh start and two classes are required')
    if train['batch'] != train['nbs'] or train['patience'] != 0:
        raise ValueError('Use the fixed update budget without accumulation or early stopping')
    if ev['multi_label'] or ev['agnostic_nms']:
        raise ValueError('This protocol uses single-label candidates and class-aware NMS')
    if not 0 <= ev['benign_false_alert_budget'] <= 1:
        raise ValueError('Invalid false-alert budget')
    if not ev['threshold_grid'] or any(not ev['score_floor'] <= t <= 1 for t in ev['threshold_grid']):
        raise ValueError('Threshold grid is outside the prediction cache range')
    return cfg


def stage_content_digest(stage):
    files = read_json(Path(stage) / 'audit.json')['files']
    content = {i: item['file_sha256'] for i, item in files.items()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def check_runtime():
    versions = {k: importlib.metadata.version(k) for k in ('ultralytics', 'torch', 'torchvision')}
    wanted = {'ultralytics': '8.3.221', 'torch': '2.8.0', 'torchvision': '0.23.0'}
    if any(versions[k].split('+')[0] != v for k, v in wanted.items()):
        raise RuntimeError(f'Install requirements.txt in an isolated environment: {versions}')
    return {'python': platform.python_version(), 'platform': platform.platform(), **versions}


def validate_lock(lock, config_path, checkpoint, manifest_hashes):
    if lock['selection_split'] != 'val' or lock['fresh_blind_test'] is not False:
        raise ValueError('Only a validation-selected supplementary reassessment is allowed')
    if lock['config_sha256'] != digest(config_path) or lock['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Configuration or checkpoint changed after selection')
    if lock['manifest_sha256'] != manifest_hashes:
        raise ValueError('Data changed after validation selection')
    cfg = load_config(config_path)
    if lock['threshold'] not in cfg['evaluation']['threshold_grid']:
        raise ValueError('Locked threshold is outside the predeclared grid')
