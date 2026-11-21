# -*- coding: utf-8 -*-
"""Command-line interface: doctor / generate / benchmark / speculate / serve / config / version."""

from __future__ import annotations

import argparse
import json
import sys

from loopinfer import __version__
from loopinfer.engine.inference import InferenceEngine
from loopinfer.engine.config import EngineConfig
from loopinfer.config_loader import load_engine_config
from loopinfer.core.state import Random
from loopinfer.model.config import ModelConfig
from loopinfer.model.factory import build_model
from loopinfer.loop.config import LoopConfig
from loopinfer.moe.config import MoEConfig
from loopinfer.moe.layer import MoELayer
from loopinfer.attention.config import AttentionConfig
from loopinfer.attention.multihead import MultiHeadAttention
from loopinfer.drafter.speculator import SpeculativeDecoder
from loopinfer.quant.config import QuantConfig
from loopinfer.quant.ops import quantize, dequantize, roundtrip_error, compression_ratio
from loopinfer.core.sampler import list_samplers
from loopinfer.tokenizer import encode, decode


def _check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:  # noqa
        print(f"  [FAIL] {name}: {e}")
        return False


def cmd_doctor(args):
    print("loopinfer doctor: running self-checks")
    results = []

    def c1():
        cfg = EngineConfig.from_dict({})
        assert cfg.model.eff_depth == 48, f"eff_depth={cfg.model.eff_depth}"
    results.append(_check("config", c1))

    def c2():
        ids = encode("循环 Transformer 等效深度")
        assert ids, "empty encoding"
        assert decode(ids)
    results.append(_check("tokenizer", c2))

    def c3():
        lc = LoopConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=3)
        assert lc.eff_depth == 48
        assert lc.n_loop_layers == 12
    results.append(_check("loop", c3))

    def c4():
        mc = MoEConfig(dim=16, hidden_ffn=32, n_experts=32, top_k=8)
        layer = MoELayer(mc, Random(1))
        out, lb = layer.forward([0.1] * 16)
        assert len(out) == 16
        assert lb >= 0.0
    results.append(_check("moe", c4))

    def c5():
        ac = AttentionConfig(dim=16, n_heads=4, head_dim=4)
        attn = MultiHeadAttention(ac, Random(2))
        out = attn.forward([0.1] * 16)
        assert len(out) == 16
    results.append(_check("attention", c5))

    def c6():
        mc = ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, n_experts=32,
                         n_layers=12, loop_start=3, loop_end=9, loop_passes=2, vocab_size=64)
        model = build_model(mc)
        logits, x = model.forward_token(1)
        assert len(logits) == 64
        assert len(x) == 16
    results.append(_check("model", c6))

    def c7():
        mc = ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, n_experts=32,
                         n_layers=12, loop_start=3, loop_end=9, loop_passes=2, vocab_size=64)
        model = build_model(mc)
        dec = SpeculativeDecoder(model, rng=Random(3))
        accepted, draft = dec.decode_step(1, Random(4))
        assert 0 <= accepted <= len(draft)
    results.append(_check("drafter", c7))

    def c8():
        qc = QuantConfig(format="nvfp4", block_size=16)
        vals = [0.01 * i for i in range(32)]
        codes, scales = quantize(vals, qc)
        assert len(codes) == 32
        err = roundtrip_error(vals, qc)
        assert err < 1.0, f"quant error too large: {err}"
    results.append(_check("quant", c8))

    def c9():
        engine = InferenceEngine()
        r = engine.generate("循环层的等效深度", max_tokens=8)
        assert len(r.tokens) == 8
    results.append(_check("generate", c9))

    def c10():
        assert len(list_samplers()) >= 4
    results.append(_check("registry", c10))

    passed = sum(results)
    print(f"doctor: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def cmd_generate(args):
    engine = InferenceEngine(load_engine_config(args.config))
    r = engine.generate(args.prompt, max_tokens=args.max_tokens, sampler=args.sampler,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(f"prompt: {args.prompt}")
    print(f"output: {r.text!r}")
    print(f"tokens: {len(r.tokens)}  loop_passes: {r.loop_passes}  eff_depth: {r.eff_depth}")
    return 0


def cmd_benchmark(args):
    print("loopinfer benchmark: recurrent depth vs parameter count")
    print(f"{'passes':>7} {'eff_depth':>9} {'n_params':>9}")
    for p in args.loop_passes:
        cfg = ModelConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=p)
        model = build_model(cfg)
        print(f"{p:>7} {cfg.eff_depth:>9} {model.n_params():>9}")
    return 0


def cmd_speculate(args):
    engine = InferenceEngine(load_engine_config(args.config))
    rng = Random(args.seed)
    accepted, draft = engine.speculate(args.first_token, rng)
    print(f"drafted={len(draft)} accepted={accepted}")
    print(f"expected_speedup={engine.decoder.expected_speedup():.2f}x")
    print(f"actual_speedup={engine.spec_speedup()['actual']:.2f}x")
    print(f"acceptance_rate={engine.spec_speedup()['acceptance_rate']:.2%}")
    return 0


def cmd_config(args):
    engine = InferenceEngine(load_engine_config(args.config))
    print(json.dumps(engine.describe(), ensure_ascii=False, indent=2))
    return 0


def cmd_quant(args):
    qc = QuantConfig(format=args.format, block_size=args.block_size)
    vals = [0.01 * i - 0.15 for i in range(64)]
    err = roundtrip_error(vals, qc)
    ratio = compression_ratio(len(vals), qc)
    print(f"format={qc.format} bits={qc.total_bits} block={qc.block_size}")
    print(f"roundtrip_error={err:.6f}  compression_ratio={ratio:.2f}x")
    return 0


def cmd_serve(args):
    from loopinfer.serve import run_server
    run_server(args.port, args.config)
    return 0


def cmd_version(args):
    print(f"loopinfer {__version__}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="loopinfer", description="Looped Transformer inference runtime")
    p.add_argument("--config", default=None, help="path to JSON/YAML config")
    sub = p.add_subparsers(dest="cmd")

    pd = sub.add_parser("doctor", help="run self-checks")
    pd.set_defaults(func=cmd_doctor)

    pg = sub.add_parser("generate", help="generate text")
    pg.add_argument("--prompt", default="循环 Transformer 的等效深度是怎么算出来的？")
    pg.add_argument("--max-tokens", type=int, default=64)
    pg.add_argument("--sampler", default="greedy")
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument("--top-k", type=int, default=50)
    pg.add_argument("--top-p", type=float, default=0.9)
    pg.set_defaults(func=cmd_generate)

    pb = sub.add_parser("benchmark", help="compare loop passes")
    pb.add_argument("--loop-passes", type=int, nargs="+", default=[1, 2, 3, 4])
    pb.set_defaults(func=cmd_benchmark)

    ps = sub.add_parser("speculate", help="one speculative-decoding step")
    ps.add_argument("--first-token", type=int, default=1)
    ps.add_argument("--seed", type=int, default=42)
    ps.set_defaults(func=cmd_speculate)

    pc = sub.add_parser("config", help="dump resolved config")
    pc.set_defaults(func=cmd_config)

    pq = sub.add_parser("quant", help="quantization demo")
    pq.add_argument("--format", default="nvfp4", choices=["nvfp4", "mxfp8"])
    pq.add_argument("--block-size", type=int, default=16)
    pq.set_defaults(func=cmd_quant)

    pv = sub.add_parser("serve", help="start OpenAI-compatible HTTP server")
    pv.add_argument("--port", type=int, default=8080)
    pv.set_defaults(func=cmd_serve)

    pver = sub.add_parser("version", help="print version")
    pver.set_defaults(func=cmd_version)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
