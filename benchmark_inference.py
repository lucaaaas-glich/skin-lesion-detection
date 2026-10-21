"""Measure serial batch-one inference on validation images; no model selection.

Wall time includes read/decode through filtered predictions copied to CPU. Model
loading, warm-up, drawing, file saving and UI are excluded. OS caches may be warm.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from evaluation import inference_args
from modules import load_trained
from runtime import check_runtime, digest, load_config, read_json, validate_lock, write_json


def stats(values):
    arr = np.asarray(values, dtype=float)
    if not len(arr) or not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError('Invalid timings')
    return {'mean_ms': float(arr.mean()), 'median_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)), 'observations': len(arr)}


def benchmark(config, checkpoint, lock, manifest, image_root, output, device='0'):
    cfg = load_config(config); chosen = read_json(lock)
    validate_lock(chosen, config, checkpoint, cfg['manifest_sha256'])
    if digest(manifest) != cfg['manifest_sha256']['val']:
        raise ValueError('Timing uses the declared validation images only')
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh benchmark output directory')
    output.mkdir(parents=True)
    versions = check_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('The declared resource benchmark requires CUDA')
    records = list(csv.DictReader(Path(manifest).open()))
    paths = [Path(image_root) / (row['image_id']+'.jpg') for row in records]
    if any(not path.is_file() for path in paths):
        raise ValueError('Missing original image')
    model = load_trained(checkpoint)
    args = inference_args(cfg, device)
    def predict(path):
        result = model.predict(str(path), **args)[0]
        kept = result.boxes[result.boxes.conf > chosen['threshold']]
        _ = kept.data.cpu().numpy().copy()
        return result.speed
    for index in range(10):
        predict(paths[index % len(paths)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for repeat in range(3):
        for path in paths:
            torch.cuda.synchronize()
            start = time.perf_counter()
            library = predict(path)
            torch.cuda.synchronize()
            measurements.append({'repeat': repeat+1, 'image_id': path.stem,
                                 'wall_ms': (time.perf_counter()-start)*1000,
                                 **{f'library_{k}_ms': float(v) for k,v in library.items()}})
    report = {'status': 'completed', 'split': 'val', 'metrics_or_model_selection': False,
              'config_sha256': digest(config), 'checkpoint_sha256': digest(checkpoint),
              'selection_lock_sha256': digest(lock), 'manifest_sha256': digest(manifest),
              'source_sha256': digest(__file__), 'runtime': versions,
              'gpu': torch.cuda.get_device_name(), 'device': str(device),
              'batch': 1, 'imgsz': cfg['train']['imgsz'], 'dtype': 'FP32',
              'warmup_calls': 10, 'timed_repeats': 3, 'images_per_repeat': len(paths),
              'confidence_floor': cfg['evaluation']['score_floor'], 'operating_threshold': chosen['threshold'],
              'nms_iou': cfg['evaluation']['nms_iou'],
              'wall_timing_scope': 'Serial predict(path): read/decode, preprocessing, device transfer, model, NMS, operating-threshold filter and CPU prediction materialization; synchronized CUDA. Excludes model startup, warmup, plots, saves and UI.',
              'cache_note': 'Repeated original image paths; OS/network file caches may be warm. Not cold-start or mobile latency.',
              'wall': stats([m['wall_ms'] for m in measurements]),
              'library_inference': stats([m['library_inference_ms'] for m in measurements]),
              'serial_images_per_second': len(measurements)*1000/sum(m['wall_ms'] for m in measurements),
              'peak_cuda_allocated_bytes_after_warmup': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes_after_warmup': torch.cuda.max_memory_reserved()}
    (output/'nvidia_smi.txt').write_text(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout)
    write_json(output/'measurements.json', measurements)
    write_json(output/'benchmark.json', report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'lock', 'manifest', 'image-root', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device', default='0')
    a = parser.parse_args()
    benchmark(a.config, a.checkpoint, a.lock, a.manifest, a.image_root, a.output, a.device)
