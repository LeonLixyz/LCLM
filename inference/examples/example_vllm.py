"""Minimal vLLM LCLM inference example (HF encoder + vLLM decoder).

Usage:
    python -m inference.examples.example_vllm \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --prompt "<|memory_start|>A long document...<|memory_end|> Summarize." \\
        --tp 2
"""

import argparse

from inference.vllm import LCLMVLLMDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size for decoder")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    runner = LCLMVLLMDecoder(
        checkpoint_path=args.checkpoint,
        tensor_parallel_size=args.tp,
    )
    outputs = runner.generate(
        prompts=[[{"role": "user", "content": args.prompt}]],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(outputs[0])


if __name__ == "__main__":
    main()
