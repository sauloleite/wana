"""Greedy held-out generation from one locally trained checkpoint."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from wana.adapters.io.jsonl import JsonlReader


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True).to(
        args.device
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("w", encoding="utf-8") as stream:
        for index, example in enumerate(JsonlReader().read(str(args.evaluation))):
            prompt = tokenizer.apply_chat_template(
                [{"role": m.role.value, "content": m.content} for m in example.messages[:-1]],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
            with torch.inference_mode():
                output = model.generate(
                    **ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                output[0, ids["input_ids"].shape[1] :], skip_special_tokens=True
            )
            stream.write(
                json.dumps(
                    {
                        "id": index,
                        "instruction": prompt,
                        "reference": example.messages[-1].content,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            if (index + 1) % 10 == 0:
                print(
                    f"{args.checkpoint.name}: {index + 1} responses, "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "dataset_sha256": hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
