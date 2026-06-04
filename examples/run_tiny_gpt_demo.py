from __future__ import annotations

from opentsp.tiny_gpt import TinyGPTConfig
from opentsp.tiny_gpt_adapter import run_tiny_gpt_opentsp


def main() -> None:
    config = TinyGPTConfig(vocab_size=64, d_model=16, d_ff=32, max_seq_len=16, cache_len=4)
    prompt = [4, 12, 9, 3, 21]
    result = run_tiny_gpt_opentsp(prompt, config=config, seed=123)

    print("Tiny GPT OpenTSP adapter demo")
    print("-" * 80)
    print(f"Prompt token ids: {list(result.step.prompt_token_ids)}")
    print(f"Graph: {result.program.graph_name}")
    print(f"Matmul ops accelerated: {result.matmul_op_count}")
    print(f"Attention ops scheduled: {result.attention_op_count}")
    print(f"FP32 next token: {result.fp32_next_token}")
    print(f"INT8/OpenTSP next token: {result.int8_next_token}")
    print(f"Token match: {'PASSED' if result.token_match else 'FAILED'}")
    print(f"Max abs logits error: {result.max_abs_logits_error:.6f}")
    print()

    print("Unified deterministic timeline")
    print("-" * 80)
    print(f"Timeline events: {result.timeline.event_count}")
    print(f"Timeline cycles: {result.timeline.total_cycles}")
    print(f"Estimated latency at 100 MHz: {result.timeline.total_cycles / 100_000_000 * 1e6:.3f} microseconds")
    print()

    print("Per-op timeline summary")
    print("-" * 80)
    for op_name, source, count, cycles in result.timeline.op_summary():
        print(f"{op_name:<14} {source:<14} events={count:<4} cycles={cycles:<5}")

    if not result.token_match:
        raise SystemExit("Tiny GPT adapter check failed: token mismatch")
    print("\nTiny GPT OpenTSP adapter check: PASSED")


if __name__ == "__main__":
    main()
