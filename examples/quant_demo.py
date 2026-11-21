# -*- coding: utf-8 -*-
"""Quantization demo: NVFP4 vs MXFP8 roundtrip error and compression."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.quant import QuantConfig, quantize, dequantize, roundtrip_error, compression_ratio


def main():
    vals = [0.01 * i - 0.15 for i in range(64)]
    for fmt in ["nvfp4", "mxfp8"]:
        qc = QuantConfig(format=fmt, block_size=16)
        codes, scales = quantize(vals, qc)
        restored = dequantize(codes, scales, qc)
        err = roundtrip_error(vals, qc)
        ratio = compression_ratio(len(vals), qc)
        print(f"{fmt:6} bits={qc.total_bits} error={err:.6f} compression={ratio:.2f}x "
              f"codes={len(codes)} restored={len(restored)}")


if __name__ == "__main__":
    main()
