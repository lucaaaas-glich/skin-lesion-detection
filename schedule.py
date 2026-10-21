"""Projected schedule and observed optimizer-step audit for the pinned trainer.

The projection follows Ultralytics 8.3.221's zero-based linear scheduler and
inclusive warm-up branch. It is not a real training result. The actual trainer's
optimizer groups are checked and recorded before every completed step call.
"""
import math


def expected_lrs(args, epoch, batch_index, batches_per_epoch):
    if args['cos_lr'] or args['optimizer'] != 'AdamW' or args['nbs'] != args['batch']:
        raise ValueError('Schedule audit requires the declared linear AdamW/no-accumulation protocol')
    target = args['lr0'] * (max(1 - epoch / args['epochs'], 0) * (1 - args['lrf']) + args['lrf'])
    warmup = max(round(args['warmup_epochs'] * batches_per_epoch), 100) if args['warmup_epochs'] > 0 else -1
    index = epoch * batches_per_epoch + batch_index
    if index <= warmup:
        start = [args['warmup_bias_lr'], 0.0, 0.0]
        return [value + (target-value) * index/warmup for value in start]
    return [target] * 3


def schedule_summary(cfg, training_images):
    args = cfg['train']; batches = math.ceil(training_images / args['batch'])
    steps = batches * args['epochs']
    rates = [expected_lrs(args, i//batches, i % batches, batches) for i in range(steps)]
    warmup = max(round(args['warmup_epochs'] * batches), 100) if args['warmup_epochs'] > 0 else -1
    return {'evidence': 'projection_from_pinned_schedule_not_observed_full_training',
            'batches_per_epoch': batches, 'planned_optimizer_step_calls': steps,
            'warmup_last_global_batch_index_inclusive': warmup,
            'planned_zero_lr_step_calls': sum(all(lr == 0 for lr in group) for group in rates),
            'first_step_lrs': rates[0], 'last_step_lrs': rates[-1],
            'maximum_lr': max(max(group) for group in rates),
            'AdamW_betas': [args['momentum'], .999],
            'warmup_momentum_changes_AdamW_betas': False,
            'positive_lr_does_not_guarantee_every_parameter_changes': True}


def optimizer_snapshot(trainer):
    """Read actual optimizer state; reject drift from this experiment's budget."""
    step = getattr(trainer, 'project_optimizer_steps', 0)
    batches = len(trainer.train_loader)
    epoch, batch = divmod(step, batches)
    if trainer.epoch != epoch or step >= batches * trainer.epochs or trainer.accumulate != 1:
        raise RuntimeError('Observed optimizer steps differ from the fixed epoch/batch budget')
    if type(trainer.optimizer).__name__ != 'AdamW':
        raise RuntimeError('Optimizer differs from the declared AdamW protocol')
    groups = trainer.optimizer.param_groups
    wanted = expected_lrs(vars(trainer.args), epoch, batch, batches)
    if len(groups) != 3 or any(not math.isclose(g['lr'], lr, rel_tol=1e-10, abs_tol=1e-12)
                               for g, lr in zip(groups, wanted)):
        raise RuntimeError('Observed learning rates differ from the pinned schedule')
    expected_decay = [0.0, trainer.args.weight_decay, 0.0]
    for group, decay in zip(groups, expected_decay):
        if tuple(group['betas']) != (trainer.args.momentum, .999) or group['weight_decay'] != decay:
            raise RuntimeError('AdamW betas/weight decay differ from the declared protocol')
    return {'optimizer_step_call': step+1, 'epoch': epoch+1, 'batch_in_epoch': batch+1,
            'lr': [float(g['lr']) for g in groups],
            'betas': [list(g['betas']) for g in groups],
            'weight_decay': [g['weight_decay'] for g in groups],
            'accumulate': trainer.accumulate,
            'all_learning_rates_zero': all(g['lr'] == 0 for g in groups)}
