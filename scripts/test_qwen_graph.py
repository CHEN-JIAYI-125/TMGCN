"""Regression checks for immutable global-id alignment; no model download needed."""
import unittest
from offline.build_qwen_graph import (
    build_utterance_prompt,
    collect_utterance_units,
    sentence_rows,
    target_rows,
    units_to_index,
)


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

    def test_same_conversation_opposite_targets_get_two_units(self):
        conv = ['这会毁掉行业。', '同意。']
        gids = [10, 11]
        records = [[
            dict(id=1, target='日本动漫文化', sentences=conv, global_ids=gids, speakers=[0, 1]),
            dict(id=2, target='反对日本动漫文化', sentences=conv, global_ids=gids, speakers=[0, 1]),
        ]]
        units = collect_utterance_units(records)
        self.assertEqual(len(units), 4)
        keys = {(u['target'], u['gid']) for u in units}
        self.assertEqual(keys, {
            ('日本动漫文化', 10), ('日本动漫文化', 11),
            ('反对日本动漫文化', 10), ('反对日本动漫文化', 11),
        })
        by_key = {(u['target'], u['gid']): u['prompt'] for u in units}
        self.assertNotEqual(by_key[('日本动漫文化', 10)], by_key[('反对日本动漫文化', 10)])
        self.assertIn('日本动漫文化', by_key[('日本动漫文化', 10)])
        self.assertIn('反对日本动漫文化', by_key[('反对日本动漫文化', 10)])
        self.assertNotIn('Favor', by_key[('日本动漫文化', 10)])
        self.assertNotIn('all_labels', by_key[('日本动漫文化', 10)])
        index = units_to_index(units)
        self.assertEqual(index['日本动漫文化']['10'], 0)
        self.assertEqual(index['反对日本动漫文化']['10'], 2)

    def test_legacy_encoder_forward_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this machine')
        from offline.qwen3_legacy import Qwen3LegacyEncoder
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False,
        )
        model = Qwen3LegacyEncoder(cfg).eval()
        ids = torch.tensor([[1, 2, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        out = model(ids, mask).last_hidden_state
        self.assertEqual(tuple(out.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_prompt_marks_current_utterance(self):
        text = build_utterance_prompt('Sora', ['先看演示。', '没什么用处。'], [0, 1], 1)
        self.assertIn('Instruct:', text)
        self.assertIn('当前会话中，当前话语对目标的立场是什么', text)
        self.assertIn('目标：Sora', text)
        self.assertIn('[当前话语]', text)
        self.assertIn('没什么用处。', text)
        self.assertNotIn('label', text.lower())


if __name__ == '__main__':
    unittest.main()
