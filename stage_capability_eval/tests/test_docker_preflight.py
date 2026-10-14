import copy
import unittest

from stage_capability_eval.docker_preflight import summarize_reference, select_rows


class ReferenceTests(unittest.TestCase):
    def phases(self):
        return {'buggy': {'status': 'completed', 'FAIL_TO_PASS': {'t': 'FAILED'},
                          'PASS_TO_PASS': {'p': 'PASSED'}},
                'fixed': {'status': 'completed', 'FAIL_TO_PASS': {'t': 'PASSED'},
                          'PASS_TO_PASS': {'p': 'PASSED'}}}

    def test_reference_needs_all_declared_outcomes(self):
        phases = self.phases()
        self.assertTrue(summarize_reference(phases)['reference_valid'])
        for phase, group, outcome in [('buggy', 'FAIL_TO_PASS', 'ERROR'),
                                      ('fixed', 'FAIL_TO_PASS', 'NOT_COLLECTED'),
                                      ('buggy', 'PASS_TO_PASS', 'FAILED')]:
            broken = copy.deepcopy(phases)
            broken[phase][group]['t' if group == 'FAIL_TO_PASS' else 'p'] = outcome
            self.assertFalse(summarize_reference(broken)['reference_valid'])

    def test_infra_and_timeout_are_not_reference_failures(self):
        for status in ('infra_error', 'timeout'):
            phases = self.phases()
            phases['fixed']['status'] = status
            result = summarize_reference(phases)
            self.assertEqual(result['status'], status)
            self.assertIsNone(result['reference_valid'])
        phases = self.phases()
        phases['buggy']['FAIL_TO_PASS']['t'] = 'ERROR'
        self.assertEqual(summarize_reference(phases)['failure_category'], 'reference_test_error')

    def test_empty_f2p_is_invalid_but_no_p2p_can_be_valid(self):
        phases = self.phases()
        for p in phases.values(): p['PASS_TO_PASS'] = {}
        self.assertTrue(summarize_reference(phases)['reference_valid'])
        phases['fixed']['FAIL_TO_PASS'] = {}
        self.assertFalse(summarize_reference(phases)['reference_valid'])

    def test_manifest_order_and_missing_ids(self):
        source = [{'instance_id': 'a'}, {'instance_id': 'b'}]
        self.assertEqual(select_rows(source, [{'instance_id': 'b'}]), [source[1]])
        with self.assertRaises(ValueError): select_rows(source, [{'instance_id': 'missing'}])
        with self.assertRaises(ValueError): select_rows(source, [{'instance_id': 'a'}] * 2)


if __name__ == '__main__': unittest.main()
