"""Verify schedule claims against pinned upstream optimizer/scheduler behavior."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from runtime import load_config
from schedule import expected_lrs, schedule_summary


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).with_name('experiment.json'))

    def test_declared_schedule_budget_and_warmup_boundaries(self):
        result = schedule_summary(self.cfg, 768)
        self.assertEqual(result['planned_optimizer_step_calls'], 2880)
        self.assertEqual(result['planned_zero_lr_step_calls'], 1)
        self.assertEqual(result['warmup_last_global_batch_index_inclusive'], 144)
        self.assertEqual(result['first_step_lrs'], [0., 0., 0.])
        self.assertAlmostEqual(result['last_step_lrs'][0], .0000265)
        self.assertAlmostEqual(expected_lrs(self.cfg['train'], 3, 0, 48)[0], .0009505)

    def test_projection_matches_actual_upstream_scheduler_for_all_60_epochs(self):
        import torch
        from ultralytics.engine.trainer import BaseTrainer
        args = self.cfg['train']
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))], lr=args['lr0'])
        harness = SimpleNamespace(args=SimpleNamespace(**args), epochs=60, optimizer=optimizer)
        BaseTrainer._setup_scheduler(harness)
        harness.scheduler.last_epoch = -1
        without_warmup = {**args, 'warmup_epochs': 0.0}
        for epoch in range(60):
            # Mark the dummy optimizer as stepped to avoid a PyTorch ordering warning;
            # use the actual pinned scheduler, with the same initial epoch index.
            optimizer.step(); harness.scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected_lrs(without_warmup, epoch, 0, 48)[0])

    def test_upstream_adamw_groups_and_zero_lr_step(self):
        import torch
        from ultralytics.engine.trainer import BaseTrainer
        args = self.cfg['train']
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        harness = SimpleNamespace(args=SimpleNamespace(**args))
        optimizer = BaseTrainer.build_optimizer(harness, model, name='AdamW', lr=args['lr0'],
                                               momentum=args['momentum'], decay=args['weight_decay'])
        self.assertEqual([g['weight_decay'] for g in optimizer.param_groups], [0., .0005, 0.])
        for group in optimizer.param_groups:
            self.assertEqual(group['betas'], (.9, .999))
            self.assertNotIn('momentum', group)  # Upstream warmup checks this key, not beta1.
            group['lr'] = 0.
        before = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters(): p.grad = torch.ones_like(p)
        optimizer.step()
        for p, old in zip(model.parameters(), before):
            self.assertTrue(torch.equal(p, old))
            self.assertEqual(optimizer.state[p]['step'].item(), 1)
            self.assertGreater(torch.count_nonzero(optimizer.state[p]['exp_avg']).item(), 0)


if __name__ == '__main__':
    unittest.main()
