"""Regression tests for mistakes that would invalidate the experimental claims."""
import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from dataset import read_manifests, stage_dataset, verify_staged
from evaluation import iou, score_records, select_threshold
from runtime import digest, load_config, read_json, stage_content_digest, validate_lock, write_json
from smoke_test import make_fixture
from train import check_yaml, evaluate, release_check

BASE = Path(__file__).with_name('experiment.json')


def record(i, c, predictions):
    return {'image_id': i, 'class_id': c, 'box': [0.2, 0.2, 0.8, 0.8], 'predictions': predictions}


def prediction(c, conf=0.8, box=None):
    return {'class_id': c, 'confidence': conf, 'box': box or [0.2, 0.2, 0.8, 0.8]}


class MetricTests(unittest.TestCase):
    def test_iou_intersection_over_union(self):
        self.assertAlmostEqual(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1/7)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)

    def test_good_localisation_wrong_class_is_not_tp(self):
        s = score_records([record('m', 1, [prediction(0)])], .05)
        m, b = s['classes']['Malignant'], s['classes']['Benign']
        self.assertEqual((m['tp'], m['fn'], b['fp']), (0, 1, 1))
        self.assertEqual(m['iou']['all_gt_class_agnostic']['mean'], 1)
        self.assertEqual(m['iou']['all_gt_class_correct']['mean'], 0)
        self.assertIsNone(m['iou']['tp_only']['mean'])

    def test_oversized_box_alert_but_detection_fp_and_fn(self):
        s = score_records([record('m', 1, [prediction(1, box=[0, 0, 1, 1])])], .05)
        m = s['classes']['Malignant']
        self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 1, 1))
        self.assertEqual(s['image_alerts']['sensitivity'], 1)

    def test_duplicate_predictions_cannot_match_same_gt_twice(self):
        boxes = [[.2, .2, .65, .8], [.35, .2, .8, .8], [.2, .2, .8, .65]]
        s = score_records([record('b', 0, [prediction(0, .9-.1*i, b) for i, b in enumerate(boxes)])], .05)
        b = s['classes']['Benign']
        self.assertEqual((b['tp'], b['fp'], b['fn']), (1, 2, 0))

    def test_removing_low_score_false_positive_changes_precision_not_recall(self):
        r = [record('m', 1, [prediction(1, .9), prediction(1, .1, [0, 0, .1, .1])])]
        low, high = [score_records(r, t)['classes']['Malignant'] for t in (.05, .5)]
        self.assertEqual((low['precision'], high['precision']), (.5, 1))
        self.assertEqual((low['recall'], high['recall']), (1, 1))

    def test_threshold_selection_respects_false_alert_budget(self):
        records = [record('m1', 1, [prediction(1, .8)]), record('m2', 1, [prediction(1, .3)]),
                   record('b1', 0, [prediction(1, .4)]), record('b2', 0, []), record('b3', 0, [])]
        policy = {'match_iou': .5, 'threshold_grid': [.1, .5, .9], 'benign_false_alert_budget': .1}
        self.assertEqual(select_threshold(records, policy)['selected']['threshold'], .5)
        with self.assertRaises(ValueError): select_threshold(records, policy, split='reassessment')

    def test_no_feasible_point_is_reported_not_silently_selected(self):
        records = [record('m', 1, [prediction(1)]), record('b', 0, [prediction(1)])]
        policy = {'match_iou': .5, 'threshold_grid': [.1, .5], 'benign_false_alert_budget': 0}
        self.assertIsNone(select_threshold(records, policy)['selected'])

    def test_duplicate_image_id_rejected(self):
        with self.assertRaises(ValueError): score_records([record('x', 0, []), record('x', 0, [])], .5)

    def test_score_boundary_matches_strict_upstream_filter(self):
        s = score_records([record('m', 1, [prediction(1, 1.0)])], 1.0)
        self.assertEqual(s['classes']['Malignant']['tp'], 0)
        self.assertEqual(s['image_alerts']['malignant_fn'], 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config, self.prepared = make_fixture(BASE, self.root / 'fixture')
        self.cfg = load_config(self.config)

    def test_real_experiment_draft_cannot_start_and_smoke_cannot_hide_real_ids(self):
        with self.assertRaises(ValueError): release_check(load_config(BASE), {})
        rows, _ = read_manifests(self.prepared, self.cfg)
        rows['train'][0]['image_id'] = 'ISIC_0012653'
        with self.assertRaises(ValueError): release_check(self.cfg, rows)

    def test_labels_and_group_leakage_are_checked_not_only_file_hashes(self):
        label = next((self.prepared / 'train/labels').glob('*.txt'))
        original = label.read_text(); label.write_text('1 0.5 0.5 0.1 0.1\n')
        with self.assertRaises(ValueError): read_manifests(self.prepared, self.cfg)
        label.write_text(original)
        path = self.prepared / 'val.csv'
        with path.open(newline='') as f: rows = list(csv.DictReader(f))
        rows[0]['patient_id'] = 'P_SYNTH_0_000'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        self.cfg['manifest_sha256']['val'] = digest(path)
        with self.assertRaises(ValueError): read_manifests(self.prepared, self.cfg)

    def test_staging_preserves_images_and_protects_labels(self):
        stage = self.root / 'stage'; audit = stage_dataset(self.prepared, self.config, stage)
        self.assertEqual(audit['counts'], {'train': 8, 'val': 4, 'reassessment': 4})
        rows, _ = verify_staged(stage, self.prepared, self.cfg)
        i = rows['train'][0]['image_id']
        self.assertTrue((stage / 'train/images' / f'{i}.png').is_symlink())
        (stage / 'train/labels' / f'{i}.txt').write_text('0 0.5 0.5 0.1 0.1\n')
        with self.assertRaises(ValueError): verify_staged(stage, self.prepared, self.cfg)

    def test_reassessment_cannot_be_injected_through_training_yaml(self):
        stage = self.root / 'stage'; stage_dataset(self.prepared, self.config, stage)
        check_yaml(stage, 'trainval')
        path = stage / 'trainval.yaml'
        path.write_text(path.read_text().replace('val: val/images', 'val: reassessment/images'))
        with self.assertRaises(ValueError): check_yaml(stage, 'trainval')

    def test_lock_rejects_changed_checkpoint_config_or_membership(self):
        weights = self.root / 'dummy.pt'; weights.write_bytes(b'not a model; hash test only')
        lock = dict(selection_split='val', fresh_blind_test=False, threshold=.5,
                    checkpoint_sha256=digest(weights), config_sha256=digest(self.config),
                    manifest_sha256=self.cfg['manifest_sha256'])
        validate_lock(lock, self.config, weights, self.cfg['manifest_sha256'])
        changed = deepcopy(lock); changed['manifest_sha256']['val'] = 'changed'
        with self.assertRaises(ValueError): validate_lock(changed, self.config, weights, self.cfg['manifest_sha256'])
        weights.write_bytes(b'new weights')
        with self.assertRaises(ValueError): validate_lock(lock, self.config, weights, self.cfg['manifest_sha256'])

    def test_evaluation_rejects_checkpoint_from_another_cohort_before_loading_model(self):
        stage = self.root / 'stage'; stage_dataset(self.prepared, self.config, stage)
        run = self.root / 'run'; weights = run / 'fit/weights/best.pt'
        weights.parent.mkdir(parents=True); weights.write_bytes(b'hash fixture, not a model')
        training = {'status': 'completed', 'config_sha256': digest(self.config),
                    'manifest_sha256': {'val': 'another cohort'}, 'best_sha256': digest(weights),
                    'staged_image_content_sha256': stage_content_digest(stage)}
        write_json(run / 'run.json', training)
        with self.assertRaisesRegex(ValueError, 'completed run'):
            evaluate(self.config, self.prepared, stage, weights, self.root / 'eval', 'cpu')
        self.assertFalse((self.root / 'eval').exists())


if __name__ == '__main__':
    unittest.main()
