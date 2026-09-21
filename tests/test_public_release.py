"""Public-reference provenance, numerical behavior, and release-boundary tests."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import zipfile

import torch

from research.production_reference import model_core as core
from research.production_reference.training import Objective, historical_model, objective_loss, synthetic_batch, train_tensor_epoch
from scripts.audit_public_release import ORIGINAL_PUBLIC_BASE, all_public_paths, audit_file, audit_pdf_text, main as audit_main, reviewed_artifacts, terminology_findings


class ArchivedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exported_ast_and_file_hash_match_recorded_provenance(self):
        folder = Path(core.__file__).parent
        provenance = json.loads((folder / "provenance.json").read_text())
        raw = (folder / "model_core.py").read_bytes()
        self.assertEqual(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), provenance["export_sha256"])
        nodes = {n.name: n for n in ast.parse(raw.decode()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for source in provenance["sources"]:
            for symbol in source["symbols"]:
                value = hashlib.sha256(ast.dump(nodes[symbol["name"]], include_attributes=False).encode()).hexdigest()
                self.assertEqual(value, symbol["ast_sha256"], symbol["name"])

    def test_historical_architecture_count_uses_unique_parameters(self):
        model = historical_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17_145_736)
        self.assertEqual(sum(t.numel() for t in model.state_dict().values()), 17_148_808)
        self.assertEqual(model.tokens_total, 40)

    def test_fractional_censor_likelihood_matches_hand_calculation(self):
        hazards = torch.tensor([[[0.2, 0.3]], [[0.2, 0.3]]], dtype=torch.float64)
        mixture = torch.ones(2, 1, dtype=torch.float64)
        duration = torch.tensor([1.5, 1.5], dtype=torch.float64)
        events = torch.tensor([1.0, 0.0], dtype=torch.float64)
        edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loss = core.mixture_survival_nll_discrete_vec(hazards, mixture, duration, events, edges)
        torch.testing.assert_close(loss, torch.tensor([-math.log(0.8 * 0.3), -math.log(0.8 * math.sqrt(0.7))], dtype=torch.float64))

    def test_survival_is_monotonic_for_mixture(self):
        torch.manual_seed(3)
        hazards = torch.rand(5, 3, 8)
        weights = torch.softmax(torch.randn(5, 3), dim=1)
        edges = torch.linspace(0, 504, 9)
        survival = core.mixture_surv_prob_at_times(hazards, weights, edges, torch.linspace(0, 504, 31))
        self.assertTrue(bool((survival[:, 1:] <= survival[:, :-1] + 1e-7).all()))
        torch.testing.assert_close(survival[:, 0], torch.ones(5))

    def test_short_censored_rows_have_no_binary_head_gradient(self):
        batch = synthetic_batch(2)
        batch.duration = torch.tensor([12.0, 100.0])
        batch.event = torch.zeros(2)
        hazard = torch.full((2, 1, 4), 0.2, requires_grad=True)
        head = torch.zeros(2, requires_grad=True)
        loss, _ = objective_loss((hazard, torch.ones(2, 1), head), batch, torch.linspace(0, 504, 5), Objective())
        loss.backward()
        self.assertEqual(float(head.grad[0]), 0.0)
        self.assertLess(float(head.grad[1]), 0.0)
        self.assertTrue(bool(torch.isfinite(hazard.grad).all()))

    def test_public_adapter_updates_model_on_synthetic_batch(self):
        torch.manual_seed(5)
        cfg = core.TrainConfig(d_model=16, n_latents=2, fusion_layers=1, n_heads=2, n_experts=2, n_bins=8, text_tokens=2, use_k8_vectors=True, use_legacy_img_vector=False)
        model = core.MultiModalSurvModel(3, [4, 4], cfg)
        before = model.tail_head.weight.detach().clone()
        metrics = train_tensor_epoch(model, [synthetic_batch()], torch.optim.AdamW(model.parameters(), lr=1e-3))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, model.tail_head.weight))

    def test_adapter_rejects_nonfinite_input(self):
        batch = synthetic_batch()
        batch.text[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            batch.validate()


class ReleaseAuditTests(unittest.TestCase):
    def test_credentials_and_private_paths_are_flagged_without_values(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "example.py"
            secret_value = "never" + "publish" + "this"
            path.write_text('password = "' + secret_value + '"\n' + 'database = "postgres' + 'ql://reader:' + secret_value + '@host/db"\n' + 'path = "C:' + '\\' + 'Users' + '\\' + 'private' + '\\' + 'record"\n')
            findings = audit_file(root, path)
            self.assertEqual({f.rule for f in findings}, {"literal_secret_assignment", "credential_in_url", "private_absolute_path"})
            self.assertNotIn(secret_value, repr(findings))

    def test_provider_fingerprint_is_blocked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "contract.md"
            path.write_text("fi" + "nn_id")
            self.assertEqual(audit_file(root, path)[0].rule, "provider_fingerprint")

    def test_source_neutral_gate_covers_vendor_terms_and_filenames(self):
        for value in ("fi" + "nn", "ti" + "se", "e" + "bay", "ama" + "zon", "face" + "book marketplace", "i" + "phone13"):
            self.assertTrue(terminology_findings("notes.md", value))
        name = "infra/scra" + "pers/adapter.py"
        self.assertEqual(terminology_findings(name, "generic adapter")[0].rule, "acquisition_terminology_in_path")
        self.assertEqual(terminology_findings("models.py", "PyTorch NumPy scikit-learn Perceiver"), [])

    def test_product_and_payment_fingerprints_are_blocked_but_browser_identifier_is_allowed(self):
        for value in ("App" + "le", "ai_rep_app" + "le", "Air" + "Pods", "Ear" + "Pods",
                      "Sam" + "sung", "Mac" + "Book", "vi" + "pps_only", "mobile" + "pay",
                      "app" + "lepay", "google" + "pay", "Beats" + "-style"):
            self.assertTrue(terminology_findings("notes.md", value))
        self.assertEqual(terminology_findings("adapter.py", "App" + "leWebKit/537.36"), [])

    def test_record_literals_and_urls_are_blocked_without_exposing_values(self):
        example_identifier = "123" + "456789"
        examples = [
            '"listing_id": ' + example_identifier,
            'WHERE item_id IN (' + example_identifier + ')',
            'WHERE listing_id IN (1001, ' + example_identifier + ')',
            '"entity_id":\n  "' + example_identifier + '"',
            'tool --only-ids ' + example_identifier + ' --dry-run',
            'tool --only-ids 1001,' + example_identifier + ' --dry-run',
            'https://example.invalid/item/' + example_identifier,
            'https://example.invalid/view?listing_id=' + example_identifier,
        ]
        for value in examples:
            findings = terminology_findings("example.md", value)
            self.assertTrue(findings)
            self.assertNotIn(example_identifier, repr(findings))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "example.json"
            path.write_text(examples[2])
            self.assertEqual(audit_file(root, path)[0].rule, "record_identifier_literal")

    def test_aggregate_counts_dates_dimensions_and_hashes_are_not_record_ids(self):
        safe = '"database_size_bytes": 29106213679\n"parameters": 17145736\n"date": "2026-09-21"\n"dtype": "int64"\n"sha256": "' + 'a' * 64 + '"\n"listing_id": "fixture-alpha"\nlisting_id = row[0]'
        self.assertEqual(terminology_findings("aggregate.json", safe), [])

    def test_all_public_scope_excludes_missing_files_and_uses_git_ignores(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "notes.md").write_text("public")
            with patch("scripts.audit_public_release.git", return_value=b"notes.md\0deleted.md\0") as call:
                self.assertEqual(all_public_paths(root), [root / "notes.md"])
                self.assertIn("--exclude-standard", call.call_args.args)

    def test_review_manifest_is_bound_to_path_and_current_content(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\nreviewed-public-example\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "review.json"
            manifest.write_text(json.dumps({"schema_version": 1, "kind": "reviewed_public_pdf_derivatives", "files": [{"path": "paper.pdf", "sha256": digest}]}))
            approved = reviewed_artifacts(root, manifest, "reviewed_public_pdf_derivatives", ".pdf")
            self.assertEqual(audit_file(root, path, reviewed_pdfs=approved), [])
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                reviewed_artifacts(root, manifest, "reviewed_public_pdf_derivatives", ".pdf")

    def test_reviewed_archive_still_rejects_data_and_unsafe_members(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "reference.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("rows.csv", "entity_id,value\nexample,1\n")
                archive.writestr("../escape.txt", "example")
            reviewed = {"reference.zip": hashlib.sha256(path.read_bytes()).hexdigest()}
            rules = {finding.rule for finding in audit_file(root, path, reviewed_archives=reviewed)}
            self.assertEqual(rules, {"archive_data_or_binary_member", "unsafe_archive_member_path"})

    def test_reviewed_source_archive_is_screened_for_private_text(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "reference.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("notes.md", "fi" + "nn")
            reviewed = {"reference.zip": hashlib.sha256(path.read_bytes()).hexdigest()}
            findings = audit_file(root, path, reviewed_archives=reviewed)
            self.assertEqual(findings[0].rule, "provider_fingerprint")
            self.assertEqual(findings[0].path, "reference.zip!notes.md")

    def test_pdf_extraction_flags_source_term_in_a_real_content_stream(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "example.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 30 100 Td (" + ("fi" + "nn").encode() + b") Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
            writer.write(path)
            findings = audit_pdf_text(path, "example.pdf")
            self.assertEqual(findings[0].rule, "provider_fingerprint")
            self.assertEqual(findings[0].path, "example.pdf#page=1")

    def test_data_binary_and_oversized_files_are_blocked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for filename, content, rule in [("rows.parquet", b"x", "data_or_binary_artifact"), ("model.joblib", b"x", "data_or_binary_artifact"), ("model.safetensors", b"x", "data_or_binary_artifact"), ("mystery.bin", b"a\0b", "binary_content"), ("large.txt", b"x" * 101, "oversized_file")]:
                path = root / filename
                path.write_bytes(content)
                self.assertEqual(audit_file(root, path, max_bytes=100)[0].rule, rule)

    def test_strict_full_tree_blocks_model_suffix_even_when_content_is_utf8(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "model.safetensors"
            path.write_text("This text must not bypass the blocked model suffix.")
            output = io.StringIO()
            with patch("sys.argv", ["audit", "--root", str(root), "--all-public-text", "--include-reviewed-documents", str(path)]), redirect_stdout(output):
                self.assertEqual(audit_main(), 1)
            self.assertEqual(json.loads(output.getvalue())["findings"][0]["rule"], "data_or_binary_artifact")

    def test_only_explicitly_reviewed_pdf_hash_is_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\npublic-paper-test-fixture\n")
            self.assertEqual(audit_file(root, path)[0].rule, "unreviewed_pdf")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(audit_file(root, path, public_pdf_hashes=frozenset([digest])), [])
            path.write_bytes(path.read_bytes() + b"changed")
            self.assertEqual(audit_file(root, path, public_pdf_hashes=frozenset([digest]))[0].rule, "unreviewed_pdf")

    def test_pdf_added_to_head_does_not_become_reviewed_automatically(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "new-paper.pdf"
            path.write_bytes(b"%PDF-1.4\nnew-unreviewed-content\n")
            def history(_root, *arguments):
                if arguments[:1] == ("rev-parse",) and arguments[1].startswith(ORIGINAL_PUBLIC_BASE + ":"):
                    raise RuntimeError("Absent from original reviewed release")
                return b"same-blob-id\n"
            with patch("scripts.audit_public_release.git", side_effect=history):
                self.assertEqual(audit_file(root, path)[0].rule, "unreviewed_pdf")

    def test_placeholder_and_aggregate_metrics_are_allowed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "metrics.json"
            path.write_text(json.dumps({"api_key": "<PROVIDE_YOUR_OWN>", "rows": 80, "mean_loss": 1.25}))
            self.assertEqual(audit_file(root, path), [])

    def test_outside_root_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertEqual(audit_file(root, root.parent / "outside.txt")[0].rule, "path_outside_root")


if __name__ == "__main__":
    unittest.main()
