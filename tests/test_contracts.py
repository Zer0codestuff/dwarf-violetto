"""CPU checks for stopping, grading, checkpoint staging and resume safety."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwarf_violetto.checkpoint import checkpoint_files, staged_checkpoint
from dwarf_violetto.evaluation_state import save, validate_resume
from dwarf_violetto.generation import output_budget, finish_reason, validate_sampling
from dwarf_violetto.grading import boxed, normalize, is_correct
from dwarf_violetto.provenance import file_sha256, provenance, public_path


def fixture_checkpoint(folder):
    folder.mkdir(exist_ok=True)
    for name, value in (("config.json", {"model_type": "limite"}),
                        ("tokenizer.json", {}), ("tokenizer_config.json", {})):
        (folder / name).write_text(json.dumps(value))
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")


class GenerationTests(unittest.TestCase):
    def test_default_uses_remaining_context(self):
        self.assertEqual(output_budget(100, 131072), 130972)
        self.assertEqual(output_budget(100, 131072, 200000), 130972)
        self.assertEqual(output_budget(100, 131072, 12), 12)
        self.assertEqual(finish_reason("stop", 100, 131072), "stop")
        self.assertEqual(finish_reason("length", 100, 131072), "context_limit")
        self.assertEqual(finish_reason("length", 100, 131072, 12), "length")
        self.assertEqual(finish_reason("length", 100, 131072, 130972), "context_limit")

    def test_invalid_budgets_and_sampling(self):
        for arguments in ((10, 10), (11, 10), (1, 10, 0), (1, 10, -1)):
            with self.assertRaises(ValueError):
                output_budget(*arguments)
        for temperature, top_p in ((-1, .95), (float("nan"), .95), (float("inf"), .95),
                                   (.6, 0), (.6, 1.01), (.6, float("nan"))):
            with self.assertRaises(ValueError):
                validate_sampling(temperature, top_p)
        validate_sampling(0, 1)

    def test_last_box_and_incomplete_answers(self):
        text = r"First \boxed{2}, final \boxed{\frac{5}{18}}."
        self.assertEqual(boxed(text), r"\frac{5}{18}")
        self.assertEqual(normalize(boxed(text)), "5/18")
        self.assertTrue(is_correct(text, "5/18", "stop"))
        self.assertFalse(is_correct(text, "5/18", "length"))
        self.assertFalse(is_correct(text, "5/18", "context_limit"))
        self.assertFalse(is_correct(r"\boxed{5} followed by \boxed{", "5", "stop"))
        self.assertIsNone(boxed("No final answer"))


class CheckpointTests(unittest.TestCase):
    def test_conversion_failure_does_not_publish_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "q8"
            with self.assertRaises(RuntimeError):
                with staged_checkpoint(destination) as staging:
                    fixture_checkpoint(staging)
                    raise RuntimeError("Interrupted conversion")
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.iterdir()), [])
            with staged_checkpoint(destination) as staging:
                fixture_checkpoint(staging)
            self.assertEqual(len(checkpoint_files(destination)), 4)
            with self.assertRaises(FileExistsError):
                with staged_checkpoint(destination):
                    self.fail("Existing checkpoint must be preserved")

    def test_missing_shards_truncation_and_remote_code_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fixture_checkpoint(folder)
            weights = folder / "model.safetensors"
            weights.write_bytes(weights.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                checkpoint_files(folder)
            fixture_checkpoint(folder)
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "missing.safetensors"}}))
            with self.assertRaises(FileNotFoundError):
                checkpoint_files(folder)
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../model.safetensors"}}))
            with self.assertRaises(ValueError):
                checkpoint_files(folder)
            (folder / "config.json").write_text(json.dumps({"model_type": "limite", "model_file": "remote.py"}))
            with self.assertRaises(ValueError):
                checkpoint_files(folder)

    def test_resume_fingerprint_tracks_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fixture_checkpoint(folder)
            initial = provenance(folder)
            (folder / "tokenizer.json").write_text('{"changed": true}')
            tokenizer_changed = provenance(folder)
            self.assertNotEqual(initial, tokenizer_changed)
            weights = folder / "model.safetensors"
            weights.write_bytes(weights.read_bytes()[:-4] + b"5678")
            self.assertNotEqual(tokenizer_changed, provenance(folder))
            self.assertFalse(public_path(weights).startswith("/"))


class ResumeTests(unittest.TestCase):
    def test_identity_mismatch_is_rejected(self):
        current = {"settings": {"seed": 42}, "sha256": {"weights": "abc"},
                   "packages": {"mlx": "pinned"}, "cases": [{"id": "remainder"}]}
        validate_resume(current, current)
        for key in ("settings", "sha256", "packages"):
            changed = copy.deepcopy(current)
            changed[key] = {"changed": True}
            with self.assertRaises(ValueError):
                validate_resume(current, changed)
        duplicate = copy.deepcopy(current)
        duplicate["cases"] *= 2
        with self.assertRaises(ValueError):
            validate_resume(duplicate, current)

    def test_failed_replace_keeps_previous_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evaluation.json"
            save(destination, {"cases": ["completed"]})
            original = file_sha256(destination)
            with patch.object(Path, "replace", side_effect=OSError("Interrupted save")):
                with self.assertRaises(OSError):
                    save(destination, {"cases": []})
            self.assertEqual(file_sha256(destination), original)


if __name__ == "__main__":
    unittest.main()
