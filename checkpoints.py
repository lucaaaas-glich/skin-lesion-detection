"""Preserve selection provenance before upstream finalization strips metadata."""
import json
import math
from pathlib import Path


def checkpoint_event(trainer):
    fitness = float(trainer.fitness)
    metric = float(trainer.metrics['metrics/mAP50-95(B)'])
    if not math.isfinite(fitness) or fitness != metric:
        raise RuntimeError('Checkpoint fitness differs from declared validation mAP50:95')
    if trainer.ema is None or not trainer.ema.enabled:
        raise RuntimeError('Expected the pinned trainer to use an enabled EMA')
    return {'epoch': trainer.epoch+1, 'validation_mAP50_95': metric,
            'best_replaced': trainer.best_fitness == trainer.fitness,
            'ema_updates': trainer.ema.updates,
            'training_parameter_dtypes': sorted({str(p.dtype) for p in trainer.model.parameters()}),
            'ema_parameter_dtypes': sorted({str(p.dtype) for p in trainer.ema.ema.parameters()}),
            'validation_weight_source': 'EMA, as verified in pinned validator source',
            'saved_weight_source': 'EMA', 'upstream_save_dtype': 'float16'}


def selection_summary(path):
    history = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not history or [e['epoch'] for e in history] != list(range(1, len(history)+1)):
        raise ValueError('Checkpoint history must contain each completed epoch exactly once')
    selected = [e for e in history if e['best_replaced']]
    expected = max(history, key=lambda e: (e['validation_mAP50_95'], e['epoch']))
    if not selected or selected[-1] != expected:
        raise ValueError('Saved best checkpoint does not follow the declared metric/tie rule')
    return {'best_epoch': expected['epoch'], 'completed_epochs': len(history),
            'selection_validation_mAP50_95': expected['validation_mAP50_95'],
            'last_epoch_validation_mAP50_95': history[-1]['validation_mAP50_95'],
            'weight_source': 'EMA', 'tie_rule': 'later epoch wins exact ties',
            'score_scope': 'Training-time validation of in-memory EMA before FP16 serialization',
            'finalized_file_epoch_is_not_selection_epoch': True}


def inspect_saved_checkpoint(path):
    """Inspect our own newly written checkpoint; never load an unknown download."""
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') or checkpoint.get('model')
    if model is None:
        raise ValueError('Checkpoint has no saved model')
    return {'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
            'epoch_field': checkpoint.get('epoch'),
            'optimizer_state_present': checkpoint.get('optimizer') is not None,
            'ema_field_present': checkpoint.get('ema') is not None,
            'model_field_present': checkpoint.get('model') is not None}
