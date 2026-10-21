"""End-to-end synthetic plumbing check; creates no ISIC performance evidence."""
import argparse
import csv
from copy import deepcopy
from pathlib import Path

from dataset import stage_dataset
from modules import load_trained
from predict import predict
from runtime import digest, load_config, write_json
from train import fit, evaluate


def make_fixture(base_config, output):
    from PIL import Image, ImageDraw
    cfg = deepcopy(load_config(base_config))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    prepared = output / 'prepared'; prepared.mkdir()
    images = output / 'images'; images.mkdir()
    cfg.update(purpose='synthetic_smoke', status='synthetic_only', ready_for_full_training=False)
    cfg['train'].update(epochs=1, imgsz=128, batch=2, nbs=2, workers=0, warmup_epochs=0.0)
    cfg['checkpoint_selection'] = 'Synthetic one-epoch integration check only'
    # Add the no-alert endpoint for plumbing coverage only, not the real experiment.
    cfg['evaluation']['threshold_grid'].append(1.0)
    cfg['manifest_sha256'], cfg['split_class_counts'] = {}, {}
    for si, (split, count) in enumerate([('train', 8), ('val', 4), ('reassessment', 4)]):
        rows = []; labels = prepared / split / 'labels'; labels.mkdir(parents=True)
        for n in range(count):
            i = f'SYNTH_{si}_{n:03d}'; c = n % 2
            im = Image.new('RGB', (192 + 8*n, 128 + 4*si), (140+5*n, 110+15*si, 90))
            w, h = im.size; draw = ImageDraw.Draw(im)
            draw.rectangle((w*.3, h*.25, w*.7, h*.75), fill=(60+30*c, 35, 30))
            path = images / f'{i}.png'; im.save(path)
            rows.append(dict(image_id=i, image_path=str(path.resolve()), class_id=c,
                             diagnosis=('Benign', 'Malignant')[c], center_x='0.5', center_y='0.5',
                             box_width='0.4', box_height='0.5', lesion_id=f'L_{i}', patient_id=f'P_{i}', historical_split='synthetic'))
            (labels / f'{i}.txt').write_text(f'{c} 0.5 0.5 0.4 0.5\n')
        manifest = prepared / f'{split}.csv'
        with manifest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        cfg['manifest_sha256'][split] = digest(manifest)
        cfg['split_class_counts'][split] = {'Benign': count//2, 'Malignant': count//2}
    config = output / 'smoke_config.json'; write_json(config, cfg)
    return config, prepared


def run_smoke(base_config, weights_dir, output, warmup_epochs=0.0):
    import torch
    torch.set_num_threads(2)
    output = Path(output).resolve()
    config, prepared = make_fixture(base_config, output)
    if not 0 <= warmup_epochs <= 3:
        raise ValueError('Synthetic warmup probe must be between 0 and 3 epochs')
    cfg = load_config(config)
    cfg['train']['warmup_epochs'] = warmup_epochs
    write_json(config, cfg)
    stage = output / 'staged'
    stage_dataset(prepared, config, stage)
    report = {'purpose': 'synthetic_smoke', 'ISIC_images_used': 0, 'clinical_or_lesion_performance_evidence': False,
              'models': {}}
    for size in ('n', 's'):
        trained = fit(config, prepared, stage, weights_dir, size, output / f'train_{size}', 'cpu')
        assert trained['optimizer_steps'] == 4
        assert trained['zero_lr_optimizer_step_calls'] == (1 if warmup_epochs > 0 else 0)
        assert trained['checkpoint_selection']['best_epoch'] == 1
        assert trained['best_checkpoint_storage']['epoch_field'] == -1
        assert trained['best_checkpoint_storage']['parameter_dtypes'] == ['torch.float16']
        checkpoint = output / f'train_{size}/fit/weights/best.pt'
        model = load_trained(checkpoint)
        assert {p.dtype for p in model.model.parameters()} == {torch.float32}
        model.model.float().eval()
        with torch.no_grad():
            raw = model.model(torch.zeros(1, 3, 640, 640))[0]
        assert tuple(raw.shape) == (1, 6, 8400), raw.shape
        assert torch.isfinite(raw).all()
        assert ((raw[:, 4:] >= 0) & (raw[:, 4:] <= 1)).all()
        val_dir = output / f'val_{size}'
        evaluate(config, prepared, stage, checkpoint, val_dir, 'cpu')
        lock = val_dir / 'selection_lock.json'
        assert lock.is_file()
        evaluate(config, prepared, stage, checkpoint, output / f'reassessment_{size}', 'cpu', 'reassessment', lock)
        image = next((stage / 'val/images').glob('*.png'))
        predict(config, checkpoint, lock, image, output / f'predict_{size}', 'cpu')
        report['models'][size] = {'saved_and_reloaded': True, 'forward_640_shape': list(raw.shape),
                                 'optimizer_steps': trained['optimizer_steps'],
                                 'warmup_epochs': warmup_epochs,
                                 'zero_lr_optimizer_step_calls': trained['zero_lr_optimizer_step_calls'],
                                 'first_step_lrs': trained['first_step_lrs'],
                                 'last_step_lrs': trained['last_step_lrs'],
                                 'checkpoint_selection': trained['checkpoint_selection'],
                                 'stored_parameter_dtype': 'float16', 'loaded_parameter_dtype': 'float32',
                                 'validation_selection_and_locked_reassessment_ran': True,
                                 'prediction_overlay_created': True}
    write_json(output / 'smoke_verification.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', required=True, type=Path)
    p.add_argument('--weights-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--warmup-epochs', type=float, default=0.0)
    a = p.parse_args()
    print(run_smoke(a.base_config, a.weights_dir, a.output, a.warmup_epochs))
