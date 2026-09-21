"""Explicit visual-review manifest boundaries for public PNG derivatives."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from scripts.audit_public_release import audit_file, reviewed_images


def png_bytes():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\0\xff\xff\xff')) + chunk(b'IEND', b'')


class ReviewedImagesTests(unittest.TestCase):
    def test_unreviewed_png_rejected_then_exact_review_accepted_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'figure.png'
            path.write_bytes(png_bytes())
            self.assertEqual(audit_file(root, path)[0].rule, 'unreviewed_image')
            folder = root / 'docs/research'
            folder.mkdir(parents=True)
            manifest = {'schema_version': 1, 'kind': 'reviewed_public_image_derivatives', 'files': [{'path': 'figure.png', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'width': 1, 'height': 1, 'review_status': 'visually_reviewed'}]}
            (folder / 'IMAGE_RELEASE_MANIFEST.json').write_text(json.dumps(manifest))
            self.assertEqual(audit_file(root, path, reviewed_pngs=reviewed_images(root)), [])
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                reviewed_images(root)

    def test_unconfirmed_visual_review_and_wrong_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'figure.png'
            path.write_bytes(png_bytes())
            folder = root / 'docs/research'
            folder.mkdir(parents=True)
            entry = {'path': 'figure.png', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'width': 1, 'height': 1, 'review_status': 'pending'}
            manifest = {'schema_version': 1, 'kind': 'reviewed_public_image_derivatives', 'files': [entry]}
            target = folder / 'IMAGE_RELEASE_MANIFEST.json'
            target.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                reviewed_images(root)
            entry.update(review_status='visually_reviewed', width=2)
            target.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                reviewed_images(root)
