from __future__ import annotations

from opentsp.benchmark import benchmark_tiny_gpt_adapter


def main() -> None:
    report = benchmark_tiny_gpt_adapter(warmup=3, repeats=20, clock_mhz_values=(50, 100, 200))

    print("Tiny GPT adapter benchmark")
    print("-" * 80)
    print(f"Model graph: {report.model_name}")
    print(f"Prompt token ids: {list(report.prompt_token_ids)}")
    print(f"Vocab size: {report.vocab_size}")
    print(f"Hidden size: {report.d_model}")
    print(f"FFN size: {report.d_ff}")
    print(f"KV-cache length: {report.cache_len}")

    print("\nCorrectness")
    print("-" * 80)
    print(f"FP32 next token: {report.fp32_next_token}")
    print(f"INT8/OpenTSP next token: {report.int8_next_token}")
    print(f"Token match: {'PASSED' if report.token_match else 'FAILED'}")
    print(f"Max abs logits error: {report.max_abs_logits_error:.6f}")

    print("\nCPU wall-clock timings")
    print("-" * 80)
    for timing in report.timings:
        print(
            f"{timing.name:<28} min={timing.min_ms:>8.4f} ms  "
            f"mean={timing.mean_ms:>8.4f} ms  std={timing.std_ms:>8.4f} ms  repeats={timing.repeats}"
        )

    print("\nDeterministic accelerator timeline estimate")
    print("-" * 80)
    print(f"Timeline events: {report.timeline_events}")
    print(f"Timeline cycles: {report.timeline_cycles}")
    print(f"Matmul micro-ops: {report.matmul_micro_ops}")
    print(f"Attention micro-ops: {report.attention_micro_ops}")
    print(f"Matmul cycles: {report.matmul_cycles}")
    print(f"Attention cycles: {report.attention_cycles}")

    print("\nClock-speed estimates")
    print("-" * 80)
    for estimate in report.clock_estimates:
        print(
            f"{estimate.clock_mhz:>4} MHz  "
            f"latency={estimate.latency_us:>10.3f} us/token  "
            f"throughput={estimate.tokens_per_second:>12.2f} tokens/sec"
        )

    if not report.token_match:
        raise SystemExit("Benchmark failed: token mismatch")
    print("\nBenchmark check: PASSED")


if __name__ == "__main__":
    main()
