"""Exercise actual upstream checkpoint saving/stripping with controlled scores."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from checkpoints import inspect_saved_checkpoint, selection_summary
from train import trainer_class


class CheckpointTests(unittest.TestCase):
    def test_best_uses_ema_later_tie_and_preserves_epoch_outside_stripped_file(self):
        import torch
        from ultralytics.utils.torch_utils import ModelEMA, strip_optimizer
        with tempfile.TemporaryDirectory() as tmp:
            trainer_type = trainer_class()
            trainer = trainer_type.__new__(trainer_type)
            trainer.save_dir = Path(tmp); trainer.wdir = Path(tmp) / 'weights'
            trainer.best = trainer.wdir / 'best.pt'; trainer.last = trainer.wdir / 'last.pt'
            trainer.args = SimpleNamespace(save_period=-1)
            trainer.save_period = -1
            trainer.model = torch.nn.Linear(1, 1)
            with torch.no_grad(): trainer.model.weight.fill_(-5.)
            trainer.ema = ModelEMA(trainer.model)
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
            trainer.scaler = torch.amp.GradScaler('cpu', enabled=False)
            trainer.read_results_csv = lambda: {}
            trainer.best_fitness = -1.
            for epoch, score in enumerate([.4, .7, .7, .6]):
                trainer.epoch = epoch; trainer.fitness = score
                trainer.best_fitness = max(trainer.best_fitness, score)
                trainer.metrics = {'metrics/mAP50-95(B)': score}
                trainer.ema.updates = epoch+1
                with torch.no_grad(): trainer.ema.ema.weight.fill_(epoch+1.12345)
                trainer.save_model()
            summary = selection_summary(Path(tmp) / 'checkpoint_history.jsonl')
            self.assertEqual(summary['best_epoch'], 3)
            self.assertEqual(summary['completed_epochs'], 4)
            saved = torch.load(trainer.best, map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 2)  # Native checkpoint epoch is zero-based.
            self.assertIsNone(saved['model'])
            expected = torch.tensor([[3.12345]], dtype=torch.float32).half()
            self.assertTrue(torch.equal(saved['ema'].weight, expected))
            self.assertFalse(torch.equal(saved['ema'].weight.float(), trainer.model.weight))
            self.assertEqual(trainer.model.weight.dtype, torch.float32)
            self.assertEqual(trainer.ema.ema.weight.dtype, torch.float32)
            strip_optimizer(trainer.best)
            finalized = inspect_saved_checkpoint(trainer.best)
            self.assertEqual(finalized['epoch_field'], -1)
            self.assertEqual(finalized['parameter_dtypes'], ['torch.float16'])
            self.assertFalse(finalized['optimizer_state_present'])
            self.assertFalse(finalized['ema_field_present'])
            self.assertTrue(finalized['model_field_present'])
            self.assertEqual(selection_summary(Path(tmp) / 'checkpoint_history.jsonl')['best_epoch'], 3)

    def test_changed_metric_cannot_silently_drive_best_selection(self):
        from checkpoints import checkpoint_event
        trainer = SimpleNamespace(fitness=.9, metrics={'metrics/mAP50-95(B)': .7})
        with self.assertRaisesRegex(RuntimeError, 'fitness differs'):
            checkpoint_event(trainer)


if __name__ == '__main__':
    unittest.main()
