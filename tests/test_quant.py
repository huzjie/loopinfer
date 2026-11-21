# -*- coding: utf-8 -*-
import unittest

from loopinfer.quant.config import QuantConfig
from loopinfer.quant.ops import quantize, dequantize, roundtrip_error, compression_ratio
from loopinfer.quant.codec import encode_float, decode_float


class TestQuant(unittest.TestCase):
    def test_roundtrip(self):
        vals = [0.01 * i - 0.15 for i in range(32)]
        for fmt in ["nvfp4", "mxfp8"]:
            qc = QuantConfig(format=fmt, block_size=16)
            codes, scales = quantize(vals, qc)
            restored = dequantize(codes, scales, qc)
            self.assertEqual(len(restored), len(vals))
            self.assertLess(roundtrip_error(vals, qc), 1.0)

    def test_compression(self):
        qc = QuantConfig(format="nvfp4", block_size=16)
        self.assertGreater(compression_ratio(64, qc), 1.0)

    def test_codec_roundtrip(self):
        # 1.0 exactly representable
        code = encode_float(1.0, 2, 1)
        self.assertAlmostEqual(decode_float(code, 2, 1), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
