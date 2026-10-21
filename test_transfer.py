"""Catch incomplete or changed transfers without any GPU or patient images."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from transfer import MANIFEST, verify_bundle, verify_directory


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = b'0 0.5 0.5 0.4 0.5\n'
        self.name = 'data/train/labels/SYNTH_0.txt'
        self.manifest = {'format_version': 1, 'files': {
            self.name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}}}
        self.archive = self.root / 'bundle.zip'
        with zipfile.ZipFile(self.archive, 'x') as z:
            z.writestr(self.name, data)
            z.writestr(MANIFEST, json.dumps(self.manifest))

    def test_transferred_label_change_or_missing_file_is_detected(self):
        self.assertEqual(verify_bundle(self.archive)['status'], 'verified')
        extracted = self.root / 'extracted'
        with zipfile.ZipFile(self.archive) as z:
            z.extractall(extracted)
        self.assertEqual(verify_directory(extracted)['status'], 'verified')
        label = extracted / self.name
        label.write_text('1 0.5 0.5 0.4 0.5\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_directory(extracted)
        label.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            verify_directory(extracted)

    def test_unlisted_archive_files_are_rejected(self):
        with zipfile.ZipFile(self.archive, 'a') as z:
            z.writestr('unlisted.txt', 'not in manifest')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            verify_bundle(self.archive)

    def test_paths_cannot_escape_transfer_directory(self):
        bad = {'format_version': 1, 'files': {'../outside.txt': {}}}
        (self.root / MANIFEST).write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'Invalid relative'):
            verify_directory(self.root)


if __name__ == '__main__':
    unittest.main()
