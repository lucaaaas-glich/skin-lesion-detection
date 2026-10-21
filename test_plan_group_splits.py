"""Safety checks for group assignment, missing IDs and historical contamination."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from plan_group_splits import (connected_components, make_plan, read_metadata,
                               write_candidates)


def fixture():
    meta = {}
    for n, category in enumerate(['Benign'] * 40 + ['Malignant'] * 20 + ['Indeterminate']):
        i = f'ISIC_{n:07d}'
        meta[i] = dict(isic_id=i, diagnosis_1=category, lesion_id=f'L{n}', patient_id=f'P{n}')
    return meta


class GroupPlanTests(unittest.TestCase):
    def test_transitive_links_and_namespaces(self):
        m = fixture()
        m['ISIC_0000001']['lesion_id'] = m['ISIC_0000000']['lesion_id']
        m['ISIC_0000002']['patient_id'] = m['ISIC_0000001']['patient_id']
        m['ISIC_0000003']['patient_id'] = m['ISIC_0000000']['lesion_id']
        group = next(g for g in connected_components(m) if 'ISIC_0000000' in g)
        self.assertEqual(set(group), {'ISIC_0000000', 'ISIC_0000001', 'ISIC_0000002'})
        assignments, _, s = make_plan(m, 'known_lesion')
        self.assertEqual(len({assignments[i] for i in group}), 1)
        self.assertFalse(any(v for d in s['known_id_overlap_counts'].values() for v in d.values()))

    def test_missing_ids_not_fabricated_or_globally_claimed(self):
        m = fixture()
        for i in ('ISIC_0000000', 'ISIC_0000001'):
            m[i]['lesion_id'] = m[i]['patient_id'] = ''
        original = copy.deepcopy(m)
        a, components, s = make_plan(m, 'all_known_links')
        self.assertEqual(len(a), len(m))
        self.assertNotEqual(components['ISIC_0000000'], components['ISIC_0000001'])
        self.assertFalse(s['complete_patient_id_coverage_and_separation'])
        self.assertFalse(s['complete_lesion_id_coverage_and_separation'])
        self.assertFalse(s['fresh_blind_test'])
        self.assertEqual(m, original)
        strict, _, _ = make_plan(m, 'known_lesion')
        self.assertNotIn('ISIC_0000000', strict)
        self.assertNotIn('ISIC_0000001', strict)

    def test_excluded_bridge_still_constrains_included_images(self):
        m = fixture()
        m['ISIC_0000001']['patient_id'] = ''
        m['ISIC_0000001']['lesion_id'] = m['ISIC_0000000']['lesion_id']
        m['ISIC_0000002']['lesion_id'] = m['ISIC_0000000']['lesion_id']
        a, _, _ = make_plan(m, 'known_patient')
        self.assertNotIn('ISIC_0000001', a)
        self.assertEqual(a['ISIC_0000000'], a['ISIC_0000002'])

    def test_deterministic_and_singleton_training_only(self):
        m = fixture()
        a = make_plan(m, 'known_patient')
        self.assertEqual(a, make_plan(dict(reversed(list(m.items()))), 'known_patient'))
        self.assertEqual(a[0]['ISIC_0000060'], 'train')
        self.assertNotEqual(a[0], make_plan(m, 'known_patient', seed=99)[0])

    def test_reject_impossible_class_support(self):
        m = fixture()
        for row in m.values():
            row['lesion_id'] = 'one_lesion'
        with self.assertRaises(ValueError):
            make_plan(m, 'known_lesion')

    def test_reject_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'meta.csv'
            p.write_text('isic_id,diagnosis_1,lesion_id,patient_id\nA,Benign,L,P\nA,Benign,L,P\n')
            with self.assertRaises(ValueError):
                read_metadata(p)

    def test_output_is_new_and_inputs_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); inputs = root / 'inputs'; inputs.mkdir()
            m = fixture(); p = inputs / 'meta.csv'
            with p.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(next(iter(m.values())))); w.writeheader(); w.writerows(m.values())
            for split in ('train', 'val', 'test'):
                with (inputs / f'{split}.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['image_id', 'diagnosis']); w.writeheader()
                    if split == 'train':
                        w.writerows({'image_id': i, 'diagnosis': r['diagnosis_1']} for i, r in m.items())
            before = {f: f.read_bytes() for f in inputs.iterdir()}
            out = root / 'candidate'
            result = write_candidates(p, inputs, out)
            self.assertEqual(result['all_known_links']['included_images'], len(m))
            self.assertEqual(before, {f: f.read_bytes() for f in inputs.iterdir()})
            with self.assertRaises(ValueError):
                write_candidates(p, inputs, out)
            with self.assertRaises(ValueError):
                write_candidates(p, inputs, inputs / 'unsafe')


if __name__ == '__main__':
    unittest.main()
