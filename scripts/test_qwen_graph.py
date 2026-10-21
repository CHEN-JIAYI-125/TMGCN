"""Regression checks for immutable global-id alignment; no model download needed."""
import unittest
from offline.build_qwen_graph import sentence_rows, target_rows


class MappingTests(unittest.TestCase):
    def test_shared_ids_and_holes(self):
        records = [[dict(sentences=['a', 'b'], global_ids=[2, 7])],
                   [dict(sentences=['a'], global_ids=[2])]]
        self.assertEqual(sentence_rows(records), {2: 'a', 7: 'b'})

    def test_conflicting_id_rejected(self):
        with self.assertRaises(ValueError):
            sentence_rows([[dict(sentences=['a', 'b'], global_ids=[0, 0])]])

    def test_alignment_rejected(self):
        with self.assertRaises(ValueError):
            sentence_rows([[dict(sentences=['a', 'b'], global_ids=[0])]])

    def test_target_order(self):
        names, views = target_rows({'t': dict(description='d', favor_reason='f', against_reason='a')})
        self.assertEqual((names, views), (['t'], ['t', 'd', 'f', 'a']))


if __name__ == '__main__':
    unittest.main()
