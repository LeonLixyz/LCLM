"""Minimal HuggingFace-Transformers LCLM inference example.

Usage:
    python -m inference.examples.example_hf \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --prompt "<|memory_start|>A long document...<|memory_end|> Summarize."
"""

import argparse

from inference.hf import generate_text, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer, processor = load_model(args.checkpoint, device=args.device)
    generate_text(
        model=model,
        decoder_tokenizer=tokenizer,
        processor=processor,
        prompt=args.prompt,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
