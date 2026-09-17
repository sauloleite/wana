"""Optional CPU/Metal fine-tuning harness; never run by the library or normal CI.

Run separately on full, selected and random files with identical settings.
Requires wana[score]. Evaluate generated outputs with a separately chosen judge.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from wana.adapters.io.jsonl import JsonlReader


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", required=True, help="local model snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--legacy-no-eos", action="store_true", help="reproduce the original pilot")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    examples = list(JsonlReader().read(str(args.dataset)))
    losses = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        for index in torch.randperm(len(examples), generator=generator).tolist():
            e = examples[index]
            prompt = tokenizer.apply_chat_template(
                [{"role": m.role.value, "content": m.content} for m in e.messages[:-1]],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            answer = tokenizer.encode(e.messages[-1].content, add_special_tokens=False)
            if not answer:
                raise ValueError(f"example has no target: {e.id}")
            if not args.legacy_no_eos:
                if tokenizer.eos_token_id is None:
                    raise ValueError("training requires an EOS token")
                answer.append(tokenizer.eos_token_id)
            if not answer or len(prefix) + len(answer) > args.max_tokens:
                raise ValueError(f"example exceeds context or has no target: {e.id}")
            ids = torch.tensor([prefix + answer], device=args.device)
            labels = ids.clone()
            labels[:, : len(prefix)] = -100
            optimizer.zero_grad()
            loss = model(input_ids=ids, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 25 == 0:
                print(
                    f"{args.output.name}: {len(losses)}/{len(examples) * args.epochs} "
                    f"updates, mean loss={sum(losses) / len(losses):.4f}, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "training.json").write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "epochs": args.epochs,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "max_tokens": args.max_tokens,
                "target_eos": not args.legacy_no_eos,
                "model": args.model,
                "mean_training_loss": sum(losses) / len(losses),
                "updates": len(losses),
                "gradient_clip": 1.0,
                "device": args.device,
                "elapsed_seconds": time.monotonic() - started,
                "torch_version": torch.__version__,
                "transformers_version": importlib.metadata.version("transformers"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
