import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from inkling2_pack import ATTENTION, EXPERTS, build, groups


class PackTest(unittest.TestCase):
    def test_roundtrip_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'text_config': {'num_hidden_layers': 24, 'n_routed_experts': 2, 'loop_config': {'core_start': 6, 'core_end': 18, 'num_passes': 3}}}
            (root / 'config.json').write_text(json.dumps(config))
            weights = {f'model.llm.{x}': 'base.safetensors' for x in ('embed.weight', 'norm.weight', 'unembed.weight')}
            for i in range(24):
                for x in ('attn_norm.weight', 'mlp_norm.weight', 'mlp.gate.weight', 'attn.attn_sink_proj.weight', 'attn.rel_logits_proj.proj', 'attn.k_sconv.weight', 'attn.v_sconv.weight', 'attn.wr_du.weight', 'mlp.experts.w13_weight'):
                    weights[f'model.llm.layers.{i}.{x}'] = 'base.safetensors'
            (root / 'index.json').write_text(json.dumps({'weight_map': weights}))
            packs = {}
            for i in range(24):
                header = {}
                names = list(ATTENTION) + [f'{e}.{j}' for e in EXPERTS for j in range(2)]
                for name in names:
                    down = 'down_proj' in name
                    dim_in, dim_out = (16, 32) if down else (32, 32)
                    rate = 56 if name.endswith('.1') else 48
                    for field, dtype, shape in [('trellis', 'I16', [dim_in // 16, dim_out // 16, rate]), ('suh', 'F16', [dim_in]), ('svh', 'F16', [dim_out]), ('mul1', 'I32', [])]:
                        header[f'{i}.{name}.{field}'] = {'dtype': dtype, 'shape': shape}
                raw = json.dumps(header).encode(); path = root / f'{i}.safe'
                path.write_bytes(struct.pack('<Q', len(raw)) + raw); packs[i] = path
            manifest = build(root / 'config.json', root / 'index.json', packs, [])
            self.assertEqual(manifest, json.loads(json.dumps(manifest)))
            self.assertEqual(len(manifest['visits']), 48)
            self.assertEqual(len({v['layer'] for v in manifest['visits']}), 24)
            linears = manifest['layers']['6']['linears']
            self.assertNotEqual(linears['6.experts.gate_up_proj.0']['K'], linears['6.experts.gate_up_proj.1']['K'])
            keys = [f['key'] for layer in manifest['layers'].values() for linear in layer['linears'].values() for f in linear['fields'].values()]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertIn('model.llm.layers.6.attn.wr_du.weight', manifest['source_tensors'])
            self.assertNotIn('model.llm.layers.6.mlp.experts.w13_weight', manifest['source_tensors'])
            with self.assertRaises(ValueError): build(root / 'config.json', root / 'index.json', {0: packs[0]}, [])
            path = packs[0]; path.write_bytes(struct.pack('<Q', 13) + b'{"x":1,"x":2}')
            with self.assertRaises(ValueError): groups(path, 0, 2)

if __name__ == '__main__': unittest.main()
