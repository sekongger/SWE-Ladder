import json
import tempfile
import unittest
from pathlib import Path

from stage_capability_eval.build_dev_sample import build_sample


class DevSampleTests(unittest.TestCase):
    def test_sample_is_deterministic_and_public(self):
        rows = [
            {
                "instance_id": f"r{i}__x-{i}",
                "repo": "r/a" if i % 2 else "r/b",
                "source_patch_files": [f"pkg/{i}.py"],
                "task_capabilities": {"loc_file": True},
                "problem_statement": "issue",
                "patch": "must not be copied",
            }
            for i in range(8)
        ]
        labels = {r["instance_id"]: {"mapping_status": "full" if i % 2 else "partial"} for i, r in enumerate(rows)}
        a = build_sample(rows, labels, size=6)
        b = build_sample(rows, labels, size=6)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertTrue(all("patch" not in row for row in a))
        self.assertEqual(len({row["instance_id"] for row in a}), len(a))


if __name__ == "__main__":
    unittest.main()
