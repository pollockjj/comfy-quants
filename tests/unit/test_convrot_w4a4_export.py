import unittest


class TestConvRotW4A4Export(unittest.TestCase):
    def test_quantizer_emits_packed_signed_w4_and_row_scales(self):
        import torch

        from comfy_quants.backends.convrot_w4a4_model_export import _quantize_convrot_w4a4_per_row
        from comfy_quants.formats.int4_common import unpack_signed_int4_pairs

        weight = torch.arange(512, dtype=torch.float32).reshape(2, 256) / 127.0 - 1.0
        packed, scale = _quantize_convrot_w4a4_per_row(weight, group_size=256)
        unpacked = unpack_signed_int4_pairs(packed)

        self.assertEqual(tuple(packed.shape), (2, 128))
        self.assertEqual(tuple(scale.shape), (2,))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertGreaterEqual(int(unpacked.min()), -7)
        self.assertLessEqual(int(unpacked.max()), 7)

    def test_marker_matches_stock_comfyui_contract(self):
        from comfy_quants.formats.convrot_w4a4 import convrot_w4a4_checkpoint_quant_config

        self.assertEqual(
            convrot_w4a4_checkpoint_quant_config(convrot_groupsize=64),
            {"format": "convrot_w4a4", "convrot_groupsize": 64},
        )


if __name__ == "__main__":
    unittest.main()
