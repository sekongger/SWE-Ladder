import json
import tempfile
import unittest
from pathlib import Path

from stage_capability_eval.loc_labels import label_row, parse_unified_diff, collect


SOURCE = "class C:\n    def f(self):\n        return 1\n    def g(self):\n        return 2\n"


def row(patch, source=SOURCE):
    return {"instance_id": "x", "repo": "o/r", "base_commit": "abc", "patch": patch,
            "parsed_commit": json.dumps({"file_diffs": [{"minus_file": {"path": "a/mod.py"}, "old_file_content": source}]})}


class LocLabelsTests(unittest.TestCase):
    def test_diff_parser_tracks_paths_and_hunks(self):
        got = parse_unified_diff("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -2,1 +2,1 @@\n-        return 1\n+        return 3\n")
        self.assertEqual(got[0]["path"], "mod.py")
        self.assertEqual(got[0]["hunks"][0]["old_changed_lines"], [2])

    def test_full_partial_and_no_hits(self):
        full = row("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -2,1 +2,1 @@\n-        return 1\n+        return 3\n")
        self.assertEqual(label_row(full)[0]["mapping_status"], "full")
        partial = row("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -2,1 +2,1 @@\n-    def f(self):\n+    def h(self):\n@@ -20,1 +20,1 @@\n-old\n+new\n")
        self.assertEqual(label_row(partial)[0]["mapping_status"], "partial")
        nohit = row("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -20,1 +20,1 @@\n-old\n+new\n")
        self.assertEqual(label_row(nohit)[0]["mapping_status"], "no_hit")

    def test_unavailable_without_checkout_or_snapshot(self):
        r = row("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -2,1 +2,1 @@\n-x\n+y\n")
        r.pop("parsed_commit")
        self.assertEqual(label_row(r)[0]["mapping_status"], "unavailable")

    def test_deterministic_output_and_cli_files(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "in.jsonl"; source.write_text(json.dumps(row("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -2,1 +2,1 @@\n-x\n+y\n")) + "\n")
            a = collect(source, Path(d) / "a"); b = collect(source, Path(d) / "b")
            self.assertEqual(Path(a["private"]).read_bytes(), Path(b["private"]).read_bytes())
            self.assertTrue(Path(a["oracle"]).exists())
            self.assertTrue((Path(d) / "a/manifests/loc_mapping_issues.jsonl").exists())


if __name__ == "__main__": unittest.main()
