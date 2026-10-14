import unittest
from stage_capability_eval.collect_manifest import build, canonical_json

def row(i='x'):
 return {k: ('[]' if k in ('FAIL_TO_PASS','PASS_TO_PASS') else ('diff --git a/a.py b/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@' if k in ('patch','test_patch') else 'x')) for k in ['instance_id','repo','base_commit','environment_setup_commit','docker_image','problem_statement','patch','test_patch','FAIL_TO_PASS','PASS_TO_PASS','run_tests']} | {'instance_id':i}

class ManifestTests(unittest.TestCase):
 def test_build_deterministic_and_sets(self):
  rows,sets,exc=build([row('a'),row('b')]); self.assertEqual(len(rows),2); self.assertEqual(sets['loc_file_set'],['a','b']); self.assertFalse(exc); self.assertEqual(canonical_json(rows),canonical_json(rows))
 def test_duplicate_and_missing_are_excluded(self):
  bad=row('a'); del bad['patch']; rows,_,exc=build([row('a'),row('a'),bad]); self.assertEqual(len(rows),1); self.assertEqual(len(exc),2)
 def test_test_only_patch_not_loc(self):
  r=row('t'); r['patch']='diff --git a/tests/test_a.py b/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1,1 +1,1 @@'; _,sets,_=build([r]); self.assertEqual(sets['loc_file_set'],[])
if __name__=='__main__': unittest.main()
