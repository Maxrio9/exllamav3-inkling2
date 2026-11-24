import copy
import importlib.util
import pathlib
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('spike', ROOT / 'tools/inkling2_pack_spike.py')
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)
PACK = ROOT.parent / 'packs/T2/layer-06.exl3.safetensors'


class Vector(list):
    def chunk(self, count, dim=-1):
        assert count == 2 and dim == -1
        n = len(self) // 2
        return Vector(self[:n]), Vector(self[n:])
    def contiguous(self):
        return self


class FakeLinear:
    def __init__(self, decoded_in_out):
        self.weight = decoded_in_out
        self.calls = 0
        self.config = types.SimpleNamespace(infer_params=types.SimpleNamespace(no_reconstruct=True))
    def forward(self, x, params):
        assert params == {}
        self.calls += 1
        return Vector(sum(x[i] * self.weight[i][j] for i in range(len(x)))
                      for j in range(len(self.weight[0])))


class PackSpikeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = spike.read_header(PACK)
        cls.groups = spike.parse_groups(cls.header, 6)

    def test_real_headers_and_mixed_rates(self):
        for comp in spike.EXPERTS:
            a, b = (self.groups[f'6.{comp}.{i}'] for i in (0, 1))
            self.assertEqual([spike.describe(g)['K'] for g in (a, b)], [3.0, 3.5])
        self.assertEqual(spike.describe(self.groups['6.self_attn.q_proj.linear'])['linear_in_out'], [2304, 3072])
        self.assertEqual(spike.describe(self.groups['6.experts.down_proj.0'])['linear_in_out'], [1152, 2304])

    def test_missing_field_and_group(self):
        for key in ('6.experts.down_proj.0.suh', '6.self_attn.q_proj.linear.trellis'):
            h = copy.deepcopy(self.header)
            del h[key]
            with self.assertRaises(ValueError): spike.parse_groups(h, 6)
        h = {k: v for k, v in self.header.items() if not k.startswith('6.experts.down_proj.0.')}
        with self.assertRaises(ValueError): spike.parse_groups(h, 6)

    def test_inconsistent_shape_dtype_and_rate(self):
        for field, attr, value in [('trellis', 'shape', [72, 144, 56]),
                                    ('suh', 'dtype', 'F32'), ('svh', 'shape', [2176])]:
            h = copy.deepcopy(self.header)
            h[f'6.experts.down_proj.0.{field}'][attr] = value
            with self.assertRaises(ValueError): spike.parse_groups(h, 6)

    def test_duplicate_json_key_rejected(self):
        with tempfile.NamedTemporaryFile() as f:
            body = b'{"6.x.trellis":{},"6.x.trellis":{}}'
            f.write(len(body).to_bytes(8, 'little') + body)
            f.flush()
            with self.assertRaises(ValueError): spike.read_header(f.name)

    def test_orientation_split_and_shared_handle(self):
        # Explicit HF [out,in]; fake compressed linear stores its transpose [in,out].
        hf = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        down_hf = [[1, 2], [3, 4], [5, 6]]
        gate = FakeLinear([list(row) for row in zip(*hf)])
        down = FakeLinear([list(row) for row in zip(*down_hf)])
        adapter = spike.FusedExpert(gate, down)
        x = Vector([2, -1, 3])
        a, b = adapter.gate_up_views(x)
        reference = [sum(w * v for w, v in zip(row, x)) for row in hf]
        self.assertEqual(a, reference[:2]); self.assertEqual(b, reference[2:])
        self.assertEqual(adapter.project_down(a), [sum(w * v for w, v in zip(row, a)) for row in down_hf])
        adapter.gate_up_views(x)
        self.assertIs(adapter.gate_up, gate)
        self.assertEqual(gate.calls, 2)
        gate.config.infer_params.no_reconstruct = False
        with self.assertRaises(ValueError): adapter.gate_up_views(x)


if __name__ == '__main__': unittest.main()
