"""Plan, fit and evaluate the restart using the declared local interfaces.

Full training requires a separately locked configuration. A provisional study
can be released with course acceptance explicitly recorded as unconfirmed.
Synthetic smoke runs test integration and cannot be reported as lesion results.
"""
import argparse
import json
from copy import copy
from pathlib import Path
import subprocess
import time

from dataset import get_dataloader, read_manifests, verify_staged
from evaluation import collect_predictions, inference_args, score_records, select_threshold, validator_class
from modules import build_detector, load_trained
from runtime import check_runtime, digest, load_config, read_json, stage_content_digest, validate_lock, write_json
from schedule import optimizer_snapshot, schedule_summary
from checkpoints import checkpoint_event, inspect_saved_checkpoint, selection_summary


def release_check(cfg, rows):
    if cfg['purpose'] == 'synthetic_smoke':
        if (cfg['train']['epochs'] != 1 or cfg['train']['imgsz'] > 160
                or sum(map(len, rows.values())) > 20
                or any(not r['image_id'].startswith('SYNTH_') for rs in rows.values() for r in rs)):
            raise ValueError('Smoke mode is restricted to tiny synthetic fixtures')
    elif not (cfg['ready_for_full_training'] and cfg['status'] == 'locked_before_training'):
        raise ValueError('Full experiment remains a draft; record the study release decision and lock its config first')


def trainer_class():
    from ultralytics.models.yolo.detect import DetectionTrainer
    class ProjectTrainer(DetectionTrainer):
        def save_model(self):
            event = checkpoint_event(self)
            super().save_model()
            with (self.save_dir / 'checkpoint_history.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')

        def optimizer_step(self):
            observed = optimizer_snapshot(self)
            super().optimizer_step()
            self.project_optimizer_steps = getattr(self, 'project_optimizer_steps', 0) + 1
            with (self.save_dir / 'optimizer_steps.jsonl').open('a') as stream:
                stream.write(json.dumps(observed, allow_nan=False)+'\n')

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build_detector(cfg, weights, verbose)
            write_json(self.save_dir / 'project_model_interface.json',
                       {'builder': 'modules.build_detector', 'nc': model.model[-1].nc,
                        'names': model.names, 'upstream': 'Ultralytics 8.3.221'})
            return model

        def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode='train'):
            return get_dataloader(self, dataset_path, batch_size, rank, mode)

        def get_validator(self):
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
            return validator_class()(self.test_loader, save_dir=self.save_dir,
                                     args=copy(self.args), _callbacks=self.callbacks)
    return ProjectTrainer


def plan(config, prepared):
    cfg = load_config(config); rows, hashes = read_manifests(prepared, cfg)
    import math
    batches = math.ceil(len(rows['train']) / cfg['train']['batch'])
    return {'purpose': cfg['purpose'], 'ready_for_full_training': cfg['ready_for_full_training'],
            'config_sha256': digest(config), 'manifest_sha256': hashes,
            'counts': {s: len(r) for s, r in rows.items()},
            'models': cfg['models'], 'epochs_per_model': cfg['train']['epochs'],
            'training_batches_per_epoch': batches,
            'planned_optimizer_steps_per_model': batches * cfg['train']['epochs'],
            'optimizer_step_count_means': 'Calls to optimizer.step, not a guarantee of changed parameters',
            'learning_rate_schedule': schedule_summary(cfg, len(rows['train'])),
            'gradient_accumulation': 1, 'full_training_started': False,
            'image_decoding_and_GPU_preflight_still_required': True,
            'checkpoint_selection': cfg['checkpoint_selection']}


def check_yaml(stage, name):
    import yaml
    stage = Path(stage).resolve()
    value = yaml.safe_load((stage / f'{name}.yaml').read_text())
    expected_val = 'val/images' if name == 'trainval' else 'reassessment/images'
    expected = {'path': str(stage), 'train': 'train/images', 'val': expected_val,
                'names': {0: 'Benign', 1: 'Malignant'}, 'nc': 2}
    if value != expected:
        raise ValueError('Staged YAML differs; do not substitute reassessment into train/val')


def fit(config, prepared, stage, weights_dir, model_size, output, device):
    cfg = load_config(config); rows, hashes = verify_staged(stage, prepared, cfg)
    release_check(cfg, rows); check_yaml(stage, 'trainval')
    if ',' in str(device):
        raise ValueError('Single-device training is required for this fixed update budget')
    if read_json(Path(stage) / 'audit.json')['config_sha256'] != digest(config):
        raise ValueError('Restage data after changing the experiment config')
    versions = check_runtime()
    from ultralytics import YOLO
    import torch
    weight = Path(weights_dir) / cfg['models'][model_size]['file']
    if digest(weight) != cfg['models'][model_size]['sha256']:
        raise ValueError('Initialization is not the recorded original COCO asset')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a fresh run directory; old results must not be overwritten')
    output.mkdir(parents=True)
    record = {'status': 'started', 'purpose': cfg['purpose'], 'runtime': versions,
              'config_sha256': digest(config), 'config': cfg, 'manifest_sha256': hashes,
              'initial_weights_sha256': digest(weight), 'model': model_size, 'device': str(device),
              'staged_image_content_sha256': stage_content_digest(stage),
              'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'fresh_blind_test': False, 'clinical_performance_claim': False}
    write_json(output / 'run.json', record)
    freeze = subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'environment.freeze.txt').write_text(freeze.stdout)
    ev = cfg['evaluation']
    args = {**cfg['train'], 'data': str(Path(stage).resolve() / 'trainval.yaml'),
            'device': device, 'project': str(output), 'name': 'fit', 'exist_ok': False,
            'conf': ev['score_floor'], 'iou': ev['nms_iou'], 'max_det': ev['max_det'],
            'agnostic_nms': False}
    model = YOLO(str(weight), task='detect')
    def reset_cuda_peak(trainer):
        if trainer.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.device)
    model.add_callback('on_train_start', reset_cuda_peak)
    started = time.perf_counter()
    try:
        model.train(trainer=trainer_class(), **args)
        if model.trainer.device.type == 'cuda':
            torch.cuda.synchronize(model.trainer.device)
        best = output / 'fit/weights/best.pt'; last = output / 'fit/weights/last.pt'
        if not best.is_file() or not last.is_file():
            raise RuntimeError('Missing saved checkpoints')
        trace_path = output / 'fit/optimizer_steps.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        planned = schedule_summary(cfg, len(rows['train']))
        if len(trace) != planned['planned_optimizer_step_calls'] or len(trace) != model.trainer.project_optimizer_steps:
            raise RuntimeError('Completed run has an unexpected optimizer-step budget')
        history_path = output / 'fit/checkpoint_history.jsonl'
        selection = selection_summary(history_path)
        if selection['completed_epochs'] != cfg['train']['epochs']:
            raise RuntimeError('Checkpoint history does not cover the complete epoch budget')
        record.update(status='completed', elapsed_seconds=time.perf_counter()-started,
                      optimizer_steps=model.trainer.project_optimizer_steps,
                      optimizer_step_count_means='Completed optimizer.step calls; some may have zero learning rate',
                      optimizer_trace_sha256=digest(trace_path),
                      zero_lr_optimizer_step_calls=sum(t['all_learning_rates_zero'] for t in trace),
                      first_step_lrs=trace[0]['lr'], last_step_lrs=trace[-1]['lr'],
                      checkpoint_selection=selection,
                      checkpoint_history_sha256=digest(history_path),
                      best_checkpoint_storage=inspect_saved_checkpoint(best),
                      best_sha256=digest(best), last_sha256=digest(last),
                      parameters=sum(p.numel() for p in model.model.parameters()),
                      peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(model.trainer.device) if model.trainer.device.type == 'cuda' else None))
    except Exception as error:
        record.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        write_json(output / 'run.json', record)
    return record


def evaluate(config, prepared, stage, checkpoint, output, device, split='val', lock=None):
    cfg = load_config(config)
    rows, hashes = verify_staged(stage, prepared, cfg, splits=(split,))
    release_check(cfg, rows)
    training = read_json(Path(checkpoint).resolve().parents[2] / 'run.json')
    if (Path(checkpoint).name != 'best.pt' or training['status'] != 'completed'
            or training['config_sha256'] != digest(config)
            or training['manifest_sha256'] != hashes
            or training['best_sha256'] != digest(checkpoint)
            or training['staged_image_content_sha256'] != stage_content_digest(stage)):
        raise ValueError('Use best.pt from a completed run of this exact configuration and cohort')
    if split == 'reassessment':
        if lock is None:
            raise ValueError('Supply a validation selection lock before supplementary evaluation')
        chosen = read_json(lock)
        validate_lock(chosen, config, checkpoint, hashes)
    elif split != 'val':
        raise ValueError('Unsupported evaluation split')
    name = 'trainval' if split == 'val' else 'reassessment'
    check_yaml(stage, name)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new evaluation directory')
    output.mkdir(parents=True)
    model = load_trained(checkpoint)
    args = inference_args(cfg, device)
    metrics = model.val(validator=validator_class(), data=str(Path(stage).resolve() / f'{name}.yaml'),
                        batch=cfg['train']['batch'], workers=cfg['train']['workers'],
                        project=str(output), name='ap', plots=True, **args)
    ap = {str(int(c)): {'name': model.names[int(c)], 'AP50': float(metrics.box.ap50[i]),
                        'AP50_95': float(metrics.box.ap[i])} for i, c in enumerate(metrics.box.ap_class_index)}
    records, timing = collect_predictions(model, rows[split], Path(stage) / split, cfg, device)
    write_json(output / 'predictions.json', records)
    report = {'purpose': cfg['purpose'], 'split': split, 'fresh_blind_test': False,
              'config_sha256': digest(config), 'checkpoint_sha256': digest(checkpoint),
              'training_run_sha256': digest(Path(checkpoint).resolve().parents[2] / 'run.json'),
              'checkpoint_selection': training.get('checkpoint_selection'),
              'evaluator_source_sha256': digest(Path(__file__).with_name('evaluation.py')),
              'manifest_sha256': hashes, 'ap': ap,
              'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
              'ap_score_floor': cfg['evaluation']['score_floor'], 'multi_label': False,
              'timing_scope': 'Library per-image stage timings; not end-to-end or mobile throughput',
              'library_speed_ms': timing}
    if split == 'val':
        selection = select_threshold(records, cfg['evaluation'])
        report['operating_point'] = selection
        if selection['selected'] is not None:
            selected = selection['selected']
            write_json(output / 'selection_lock.json', {
                'selection_split': 'val', 'threshold': selected['threshold'],
                'checkpoint_sha256': digest(checkpoint), 'config_sha256': digest(config),
                'manifest_sha256': hashes, 'fresh_blind_test': False,
                'purpose': cfg['purpose'], 'benign_false_alert_budget': cfg['evaluation']['benign_false_alert_budget'],
                'rule': selection['rule'], 'model_policy': cfg['model_policy']})
    else:
        report['operating_point'] = score_records(records, chosen['threshold'], cfg['evaluation']['match_iou'])
        report['selection_lock_sha256'] = digest(lock)
    write_json(output / 'evaluation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'fit', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--prepared', required=True, type=Path)
        if name != 'plan':
            p.add_argument('--stage', required=True, type=Path)
            p.add_argument('--output', required=True, type=Path)
            p.add_argument('--device', default='0')
        if name == 'fit':
            p.add_argument('--weights-dir', required=True, type=Path)
            p.add_argument('--model', choices=('n', 's'), required=True)
        if name == 'evaluate':
            p.add_argument('--checkpoint', required=True, type=Path)
            p.add_argument('--split', choices=('val', 'reassessment'), default='val')
            p.add_argument('--lock', type=Path)
    a = parser.parse_args()
    if a.command == 'plan':
        result = plan(a.config, a.prepared)
    elif a.command == 'fit':
        result = fit(a.config, a.prepared, a.stage, a.weights_dir, a.model, a.output, a.device)
    else:
        result = evaluate(a.config, a.prepared, a.stage, a.checkpoint, a.output, a.device, a.split, a.lock)
    import json
    print(json.dumps({k: v for k, v in result.items() if k not in ('library_speed_ms', 'config', 'source_sha256', 'operating_point')}, indent=2))
