# -*- coding: utf-8 -*-
"""Quantize a flat list of weights and report error / compression."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.quant import QuantConfig, quantize, dequantize, roundtrip_error, compression_ratio


def main():
    vals = [0.01 * i - 0.2 for i in range(128)]
    for fmt in ["nvfp4", "mxfp8"]:
        qc = QuantConfig(format=fmt, block_size=16)
        codes, scales = quantize(vals, qc)
        err = roundtrip_error(vals, qc)
        ratio = compression_ratio(len(vals), qc)
        print(f"{fmt:6} bits={qc.total_bits} error={err:.6f} compression={ratio:.2f}x")


if __name__ == "__main__":
    main()
