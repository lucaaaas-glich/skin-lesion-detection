"""Small synthetic runtime check for the intended device, batch and resolution.

No patient images, dataset splits or performance metrics are read. Two temporary
AdamW steps exercise forward/loss/backward/update; no checkpoint is saved.
This is not a full-trainer memory benchmark (it omits EMA, workers and validation).
"""
import argparse
from pathlib import Path
import time

from modules import build_detector
from runtime import check_runtime, digest, load_config, write_json


def normalize_device(value, cuda_available, cuda_count):
    if value == 'cpu':
        return value
    index = value.removeprefix('cuda:')
    if not index.isdecimal():
        raise ValueError('Use cpu, one CUDA index, or cuda:N')
    if not cuda_available or int(index) >= cuda_count:
        raise RuntimeError(f'Requested CUDA device {index} is unavailable; do not silently switch to CPU')
    return f'cuda:{int(index)}'


def validate_probe_size(device, batch, size):
    if batch < 2 or size < 64 or size % 32:
        raise ValueError('Use batch >= 2 and image size >= 64 divisible by 32')
    if device == 'cpu' and (batch > 2 or size > 160):
        raise ValueError('For the small local CPU check, explicitly use --batch 2 --imgsz 128')


def probe_model(cfg, weights_dir, size, device, batch_size, image_size):
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    weights = Path(weights_dir) / cfg['models'][size]['file']
    if digest(weights) != cfg['models'][size]['sha256']:
        raise ValueError('Initialization checksum differs from the official asset registry')
    torch.manual_seed(cfg['train']['seed'])
    source = YOLO(str(weights), task='detect').model
    model = build_detector(source.yaml, source).to(device).float().train()
    model.args = get_cfg(overrides=cfg['train'])
    # A simple optimizer for the throwaway resource probe, not the full trainer's
    # grouped weight-decay implementation. It exercises AdamW state allocation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr0'],
                                  weight_decay=cfg['train']['weight_decay'])
    is_cuda = device.startswith('cuda:')
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    steps = []
    for step in range(2):
        image = torch.rand(batch_size, 3, image_size, image_size, device=device)
        batch = {'img': image, 'batch_idx': torch.arange(batch_size, device=device),
                 'cls': (torch.arange(batch_size, device=device) % 2).float().view(-1, 1),
                 'bboxes': torch.tensor([[.5, .5, .4, .5]], device=device).repeat(batch_size, 1)}
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, components = model(batch)
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite synthetic loss')
        loss.sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError('Missing or nonfinite gradients')
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError('All gradients are zero')
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError('Nonfinite parameters after update')
        if is_cuda:
            torch.cuda.synchronize(device)
        steps.append({'step': step+1, 'loss_components': components.detach().cpu().tolist(),
                      'seconds_including_verification': time.perf_counter()-start})
    return {'model': size, 'nc': model.model[-1].nc, 'parameters': sum(p.numel() for p in model.parameters()),
            'input_shape': [batch_size, 3, image_size, image_size], 'dtype': 'float32',
            'finite_forward_backward_and_update': True, 'optimizer_steps': 2, 'steps': steps,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if is_cuda else None,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if is_cuda else None,
            'initial_weight_sha256': digest(weights)}


def preflight(config, weights_dir, output, device, batch=None, imgsz=None):
    import torch
    cfg = load_config(config); versions = check_runtime()
    resolved = normalize_device(device, torch.cuda.is_available(), torch.cuda.device_count())
    batch = cfg['train']['batch'] if batch is None else batch
    imgsz = cfg['train']['imgsz'] if imgsz is None else imgsz
    validate_probe_size(resolved, batch, imgsz)
    output = Path(output)
    if output.exists():
        raise ValueError('Choose an unused preflight output directory')
    output.mkdir(parents=True)
    if resolved == 'cpu':
        torch.set_num_threads(2)
    report = {'status': 'started', 'purpose': 'synthetic_runtime_preflight', 'device': resolved,
              'runtime': versions, 'torch_cuda_build': torch.version.cuda,
              'config_sha256': digest(config), 'ISIC_images_used': 0,
              'checkpoint_saved': False, 'clinical_performance_evidence': False,
              'same_batch_and_resolution_as_experiment': batch == cfg['train']['batch'] and imgsz == cfg['train']['imgsz'],
              'full_trainer_memory_guaranteed': False,
              'timing_scope': 'Synthetic forward, loss, backward, AdamW update and verification; excludes data IO; not inference FPS',
              'models': {}}
    if resolved.startswith('cuda:'):
        info = torch.cuda.get_device_properties(resolved)
        free, total = torch.cuda.mem_get_info(resolved)
        report['gpu'] = {'name': info.name, 'total_bytes': total, 'free_bytes_before_probe': free}
    try:
        for size in ('n', 's'):
            report['models'][size] = probe_model(cfg, weights_dir, size, resolved, batch, imgsz)
            if resolved.startswith('cuda:'):
                torch.cuda.empty_cache()
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / 'preflight.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--weights-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', required=True)
    p.add_argument('--batch', type=int)
    p.add_argument('--imgsz', type=int)
    a = p.parse_args()
    print(preflight(a.config, a.weights_dir, a.output, a.device, a.batch, a.imgsz))
