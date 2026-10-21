import csv
import json
import tempfile
import unittest
from pathlib import Path
from prepare_dataset import digest, prepare


def fixture(root):
    inputs = root / 'inputs'; inputs.mkdir()
    meta, boxes, manifest = [], [], []
    for n in range(18):
        i, s, c = f'ISIC_{n:07d}', ('train', 'val', 'reassessment')[n // 6], ('Benign' if n % 6 < 4 else 'Malignant')
        meta.append(dict(isic_id=i, diagnosis_1=c, lesion_id=f'L{n}', patient_id=f'P{n}'))
        boxes.append(dict(image_id=i, image_path=f'/school/{i}.jpg', center_x='0.5', center_y='0.5', box_width='0.4', box_height='0.4'))
        manifest.append(dict(image_id=i, diagnosis=c, lesion_id=f'L{n}', patient_id=f'P{n}', assignment_component=f'C{n}', candidate_split=s, historical_split='train'))
    sources = {}
    for name, rows in [('official_metadata', meta), ('official_mask_boxes', boxes), ('candidate_manifest', manifest)]:
        p = inputs / f'{name}.csv'
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        sources[name] = {'path': str(p), 'sha256': digest(p)}
    source = root / 'sources.json'; source.write_text(json.dumps(sources))
    protocol = root / 'protocol.json'; protocol.write_text(json.dumps({
        'classes': {'0': 'Benign', '1': 'Malignant'},
        'split_roles': {s: {'images': 6, 'Benign': 4, 'Malignant': 2} for s in ('train', 'val', 'reassessment')}}))
    return source, protocol


def replace_input(source, name, transform):
    sources = json.loads(source.read_text()); path = Path(sources[name]['path'])
    with path.open() as f: rows = list(csv.DictReader(f))
    transform(rows)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sources[name]['sha256'] = digest(path); source.write_text(json.dumps(sources))


class PrepareTests(unittest.TestCase):
    def test_relative_sources_resolve_from_manifest_location(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); s, p = fixture(root)
            sources = json.loads(s.read_text())
            for entry in sources.values():
                entry['path'] = str(Path(entry['path']).relative_to(root))
            s.write_text(json.dumps(sources))
            summary = prepare(s, p, root / 'prepared')
            self.assertEqual(summary['counts']['train']['images'], 6)

    def test_labels_sources_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); s, p = fixture(root)
            before = {f: f.read_bytes() for f in (root / 'inputs').iterdir()}
            out = root / 'prepared'; summary = prepare(s, p, out)
            self.assertEqual(len(list(out.glob('*/labels/*.txt'))), 18)
            self.assertEqual((out / 'train/labels/ISIC_0000004.txt').read_text(), '1 0.5 0.5 0.4 0.4\n')
            self.assertEqual(before, {f: f.read_bytes() for f in before})
            self.assertFalse(summary['fresh_blind_test'])
            with self.assertRaises(ValueError): prepare(s, p, out)

    def test_reject_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); s, p = fixture(root)
            (root / 'inputs/official_metadata.csv').write_text('changed')
            with self.assertRaises(ValueError): prepare(s, p, root / 'prepared')

    def test_reject_group_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); s, p = fixture(root)
            for name in ('official_metadata', 'candidate_manifest'):
                replace_input(s, name, lambda rows: rows[6].update(patient_id='P0'))
            with self.assertRaises(ValueError): prepare(s, p, root / 'prepared')

    def test_reject_invalid_box(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); s, p = fixture(root)
            replace_input(s, 'official_mask_boxes', lambda rows: rows[0].update(box_width='nan'))
            with self.assertRaises(ValueError): prepare(s, p, root / 'prepared')

    def test_reject_selected_subset_or_imputed_id(self):
        for remove in (True, False):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); s, p = fixture(root)
                transform = (lambda rows: rows.pop()) if remove else (lambda rows: rows[0].update(lesion_id='invented'))
                replace_input(s, 'candidate_manifest', transform)
                with self.assertRaises(ValueError): prepare(s, p, root / 'prepared')


if __name__ == '__main__':
    unittest.main()
