import json
import tempfile
import unittest
from pathlib import Path

from stage_capability_eval.collect_stage_packages import collect_packages


class StagePackageCollectionTests(unittest.TestCase):
    def test_collects_public_and_private_stage_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            oracle = root / "oracle.jsonl"
            source = root / "source.jsonl"
            row = {
                "instance_id": "repo__x-1", "repo": "repo/x", "base_commit": "abc",
                "problem_statement": "issue", "source_patch_files": ["a.py"],
                "patch": "diff --git a/a.py b/a.py", "test_patch": "diff --git a/test.py b/test.py",
                "FAIL_TO_PASS": ["f2p"], "PASS_TO_PASS": ["p2p"], "docker_image": "img",
            }
            manifest.write_text(json.dumps({k: v for k, v in row.items() if k not in {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}}) + "\n")
            source.write_text(json.dumps(row) + "\n")
            oracle.write_text(json.dumps({"instance_id": "repo__x-1", "files": [{"path": "a.py", "functions": ["f"]}]}) + "\n")
            result = collect_packages(manifest, oracle, source, root / "out")
            self.assertEqual(result["repro_count"], 1)
            self.assertEqual(result["patch_count"], 1)
            public = json.loads((root / "out/manifests/repro_given_loc.jsonl").read_text())
            self.assertNotIn("patch", public)
            private = json.loads((root / "out/evaluator_private/patch_verifier_metadata.jsonl").read_text())
            self.assertEqual(private["FAIL_TO_PASS"], ["f2p"])


if __name__ == "__main__":
    unittest.main()
