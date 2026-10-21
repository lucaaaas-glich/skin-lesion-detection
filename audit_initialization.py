"""Inspect original COCO -> two-class initialization without training or images.

Counts distinguish state-dict entries (including buffers) from scalar parameters.
An entry is transferable only when both its name and shape match. No accuracy
conclusion follows from the number of transferred entries.
"""
import argparse
from copy import deepcopy
import inspect
from pathlib import Path

from modules import build_detector
from runtime import check_runtime, digest, load_config, write_json


def inspect_model(cfg, weights_dir, size):
    import torch
    from ultralytics import YOLO

    path = Path(weights_dir) / cfg['models'][size]['file']
    if digest(path) != cfg['models'][size]['sha256']:
        raise ValueError('Original COCO initialization checksum differs')
    source = YOLO(str(path), task='detect').model.float()
    if source.model[-1].nc != 80:
        raise ValueError('Expected the original 80-class COCO head')
    torch.manual_seed(cfg['train']['seed'])
    rng_state = torch.get_rng_state()
    fresh = build_detector(deepcopy(source.yaml))
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    # Rebuild through the same project factory with the same initialization RNG.
    # This isolates loading from the random initialization of unmatched tensors.
    torch.set_rng_state(rng_state)
    target = build_detector(deepcopy(source.yaml), source)
    original, after = source.state_dict(), target.state_dict()
    transferred = {k for k in after if k in original and original[k].shape == after[k].shape}
    retained = set(after) - transferred
    if not all(torch.equal(after[k], original[k]) for k in transferred):
        raise RuntimeError('A transferable tensor differs from the original weight')
    if not all(torch.equal(after[k], before[k]) for k in retained):
        raise RuntimeError('An unmatched tensor differs from fresh initialization')
    if 'model.0.conv.weight' not in transferred:
        raise RuntimeError('Unexpected partial first-convolution loading path')

    head_index = len(target.model) - 1
    head_prefix = f'model.{head_index}.'
    if not all(k.startswith(head_prefix + 'cv3.') for k in retained):
        raise RuntimeError('Unexpected unmatched state outside classification branch')
    parameters = dict(target.named_parameters())
    buffers = dict(target.named_buffers())
    if set(parameters) | set(buffers) != set(after):
        raise RuntimeError('Parameter/buffer accounting does not cover the state dict')
    frozen = {k: p.numel() for k, p in parameters.items() if not p.requires_grad}
    if frozen != {head_prefix + 'dfl.conv.weight': 16}:
        raise RuntimeError('Unexpected frozen parameters in the project factory')
    if not torch.equal(target.model[-1].dfl.conv.weight.flatten(), torch.arange(16).float()):
        raise RuntimeError('DFL integral weights differ from their fixed bin coordinates')
    if cfg['train']['freeze'] is not None:
        raise ValueError('This audit describes the no-user-freezing protocol')

    def region(name):
        index = int(name.split('.')[1])
        if index < len(target.yaml['backbone']):
            return 'backbone'
        if index < head_index:
            return 'neck'
        return {'cv2': 'box_branch', 'cv3': 'class_branch', 'dfl': 'fixed_DFL'}[name.split('.')[2]]

    regions = {}
    for name, tensor in after.items():
        counts = regions.setdefault(region(name), {'state_entries': 0, 'transferred_state_entries': 0,
                                                   'parameter_elements': 0, 'transferred_parameter_elements': 0})
        counts['state_entries'] += 1
        counts['transferred_state_entries'] += int(name in transferred)
        if name in parameters:
            counts['parameter_elements'] += tensor.numel()
            counts['transferred_parameter_elements'] += tensor.numel() if name in transferred else 0

    return {
        'original_weight_sha256': digest(path), 'source_classes': source.model[-1].nc,
        'target_classes': target.model[-1].nc, 'target_names': target.names,
        'state_entries': len(after), 'transferred_state_entries': len(transferred),
        'retained_fresh_state_entries': len(retained),
        'parameter_elements': sum(p.numel() for p in parameters.values()),
        'transferred_parameter_elements': sum(p.numel() for k, p in parameters.items() if k in transferred),
        'trainable_parameter_elements': sum(p.numel() for p in parameters.values() if p.requires_grad),
        'frozen_parameter_elements_by_name': frozen,
        'classification_hidden_channels': {
            'source': source.model[-1].cv3[0][0].conv.out_channels,
            'target': target.model[-1].cv3[0][0].conv.out_channels},
        'regions': regions,
        'unmatched_entries': [{'name': k, 'kind': 'parameter' if k in parameters else 'buffer',
                               'source_shape': list(original[k].shape) if k in original else None,
                               'target_shape': list(after[k].shape)} for k in sorted(retained)],
        'checks': {'all_matching_tensors_equal_original': True,
                   'all_unmatched_tensors_equal_fresh_initialization': True,
                   'all_backbone_neck_and_box_state_transferred': True,
                   'only_fixed_DFL_parameters_frozen_at_construction': True},
    }


def audit(config, weights_dir, output):
    import torch
    from ultralytics.engine import trainer
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import block, head
    from ultralytics.utils import torch_utils
    import modules

    cfg = load_config(config)
    versions = check_runtime()
    output = Path(output)
    if output.exists():
        raise ValueError('Choose an unused audit output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    report = {'status': 'started', 'purpose': 'read_only_initialization_audit',
              'runtime': versions, 'config_sha256': digest(config), 'ISIC_images_used': 0,
              'optimizer_steps': 0, 'checkpoint_saved': False, 'performance_evidence': False,
              'source_sha256': {m.__name__: digest(inspect.getfile(m))
                                for m in (modules, tasks, head, block, torch_utils, trainer)},
              'audit_script_sha256': digest(__file__),
              'counting_note': 'State entries include buffers; counts are not parameter percentages.',
              'models': {}}
    try:
        for size in ('n', 's'):
            report['models'][size] = inspect_model(cfg, weights_dir, size)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / 'initialization.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--weights-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.weights_dir, args.output)
    for size, row in result['models'].items():
        print(f"{size}: {row['transferred_state_entries']}/{row['state_entries']} state entries transferred; "
              f"{row['trainable_parameter_elements']} trainable scalar parameters")
    print(f"Evidence: {args.output / 'initialization.json'}")
