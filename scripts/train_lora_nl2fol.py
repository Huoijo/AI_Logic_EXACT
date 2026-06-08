#!/usr/bin/env python3
"""
train_lora_nl2fol.py

Replacement training script for AI_Logic_EXACT.

Main fixes included:
1. Avoids the fp16 GradScaler + bfloat16 crash:
   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'
2. Keeps fp16 and bf16 mutually exclusive.
3. Aligns tokenizer/model/generation special token ids safely.
4. Works with both newer and older TRL SFTTrainer/SFTConfig APIs by filtering args dynamically.
5. Supports the uploaded EXACT-style JSON schema:
   premises-NL, premises-FOL, questions, answers, explanation.

Typical Kaggle run:
    python scripts/train_lora_nl2fol.py \
      --data_path /kaggle/working/AI_Logic_EXACT/data/full_data.json \
      --model_name Qwen/Qwen2.5-0.5B-Instruct

Useful env overrides:
    MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
    DATA_PATH=/kaggle/working/AI_Logic_EXACT/data/full_data.json
    OUTPUT_DIR=/kaggle/working/AI_Logic_EXACT/outputs/lora_nl2fol
    TASK_MODE=nl2fol        # or qa
    USE_4BIT=1
    FORCE_FP16=0
    FORCE_BF16=0
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:  # older/partial TRL installs
    SFTConfig = None  # type: ignore[assignment]
    from transformers import TrainingArguments as SFTConfig  # type: ignore[no-redef]
    from trl import SFTTrainer  # type: ignore[no-redef]


# -----------------------------
# Small utilities
# -----------------------------

def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def existing_path(*candidates: Optional[str]) -> Optional[Path]:
    for candidate in candidates:
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
    return None


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def dump_jsonish(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


# -----------------------------
# Precision handling
# -----------------------------

def choose_precision(force_fp16: bool = False, force_bf16: bool = False) -> Tuple[bool, bool, torch.dtype]:
    """Return (use_bf16, use_fp16, compute_dtype).

    Important rule:
        bf16 and fp16 must never both be True.

    The crash in the user's log happens when an fp16 GradScaler path tries to
    unscale bfloat16 gradients. BF16 training must run with fp16=False.
    """
    if not torch.cuda.is_available():
        return False, False, torch.float32

    bf16_supported = bool(torch.cuda.is_bf16_supported())

    if force_bf16:
        if not bf16_supported:
            print("[precision] WARNING: FORCE_BF16=1 but this GPU does not report BF16 support. Falling back to FP16.")
            return False, True, torch.float16
        return True, False, torch.bfloat16

    if force_fp16:
        return False, True, torch.float16

    if bf16_supported:
        return True, False, torch.bfloat16

    return False, True, torch.float16


# -----------------------------
# Dataset formatting
# -----------------------------

def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("records", "data", "examples", "train"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a list of records, got {type(data).__name__}: {path}")

    records = [x for x in data if isinstance(x, dict)]
    if not records:
        raise ValueError(f"No JSON object records found in {path}")
    return records


def format_premises_nl(premises: Sequence[str]) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(premises, start=1))


def format_premises_fol(fols: Sequence[str]) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(fols, start=1))


def messages_to_text(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Use model chat template when available; fall back to a plain format."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        chunks: List[str] = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            chunks.append(f"### {role}:\n{content}")
        return "\n\n".join(chunks) + "\n"


def make_nl2fol_samples(records: Sequence[Dict[str, Any]], tokenizer: Any) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    system = (
        "You translate university regulation premises from natural language into First-Order Logic. "
        "Return only the FOL annotations, preserving premise order."
    )

    for record in records:
        premises_nl = as_list(record.get("premises-NL") or record.get("premises_nl") or record.get("nl_premises"))
        premises_fol = as_list(record.get("premises-FOL") or record.get("premises_fol") or record.get("fol_premises"))
        if not premises_nl or not premises_fol:
            continue

        user = (
            "Convert the following natural-language premises into First-Order Logic.\n\n"
            f"Natural-language premises:\n{format_premises_nl(premises_nl)}"
        )
        assistant = format_premises_fol(premises_fol)
        text = messages_to_text(
            tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        samples.append({"text": text})

    return samples


def make_qa_samples(records: Sequence[Dict[str, Any]], tokenizer: Any) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    system = (
        "You answer questions about university regulation premises. "
        "Use the given premises only. Give the answer and a concise logical explanation."
    )

    for record in records:
        premises_nl = as_list(record.get("premises-NL") or record.get("premises_nl") or record.get("nl_premises"))
        premises_fol = as_list(record.get("premises-FOL") or record.get("premises_fol") or record.get("fol_premises"))
        questions = as_list(record.get("questions"))
        answers = as_list(record.get("answers"))
        explanations = as_list(record.get("explanation") or record.get("explanations"))

        if not premises_nl or not questions:
            continue

        n = min(len(questions), len(answers) if answers else len(questions))
        for i in range(n):
            answer = answers[i] if i < len(answers) else ""
            explanation = explanations[i] if i < len(explanations) else ""
            user = (
                f"Natural-language premises:\n{format_premises_nl(premises_nl)}\n\n"
                f"FOL premises:\n{format_premises_fol(premises_fol)}\n\n"
                f"Question:\n{questions[i]}"
            )
            assistant = f"Answer: {answer}\nExplanation: {explanation}".strip()
            text = messages_to_text(
                tokenizer,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            )
            samples.append({"text": text})

    return samples


def train_eval_split(samples: List[Dict[str, str]], eval_ratio: float, seed: int) -> Tuple[Dataset, Optional[Dataset]]:
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)

    if len(samples) < 3 or eval_ratio <= 0:
        return Dataset.from_list(samples), None

    eval_size = max(1, int(round(len(samples) * eval_ratio)))
    eval_size = min(eval_size, len(samples) - 1)
    eval_samples = samples[:eval_size]
    train_samples = samples[eval_size:]
    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)


# -----------------------------
# Compatibility helpers
# -----------------------------

def make_config(config_cls: Any, raw_kwargs: Dict[str, Any]) -> Any:
    """Build SFTConfig/TrainingArguments while ignoring unsupported keys."""
    signature = inspect.signature(config_cls.__init__)
    accepted = set(signature.parameters.keys())
    accepted.discard("self")

    kwargs = dict(raw_kwargs)

    # Transformers renamed evaluation_strategy -> eval_strategy in newer releases.
    if "eval_strategy" in accepted and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in accepted and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[config] Dropped unsupported config keys for {config_cls.__name__}: {dropped}")

    return config_cls(**filtered)


def make_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    args: Any,
    peft_config: LoraConfig,
    max_seq_length: int,
    packing: bool,
) -> Any:
    """Instantiate SFTTrainer across TRL versions."""
    signature = inspect.signature(SFTTrainer.__init__)
    accepted = set(signature.parameters.keys())

    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset

    # Newer TRL prefers processing_class. Older versions used tokenizer.
    if "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer

    # Older TRL expected these in the trainer, newer TRL expects them in SFTConfig.
    if "dataset_text_field" in accepted:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in accepted:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in accepted:
        kwargs["packing"] = packing

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[trainer] Dropped unsupported trainer keys: {dropped}")

    return SFTTrainer(**filtered)


# -----------------------------
# Model/tokenizer
# -----------------------------

def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    # Qwen/Llama-like tokenizers sometimes have no pad token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"
    return tokenizer


def align_special_tokens(model: Any, tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "bos_token_id", None) is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        if getattr(tokenizer, "bos_token_id", None) is not None:
            gen_cfg.bos_token_id = tokenizer.bos_token_id


def load_model(
    model_name: str,
    tokenizer: Any,
    compute_dtype: torch.dtype,
    use_4bit: bool,
    gradient_checkpointing: bool,
) -> Any:
    quantization_config = None
    device_map = "auto" if torch.cuda.is_available() else None

    if use_4bit:
        if not torch.cuda.is_available():
            print("[model] WARNING: USE_4BIT=1 ignored because CUDA is not available.")
            use_4bit = False
        elif not is_bnb_available():
            print("[model] WARNING: USE_4BIT=1 ignored because bitsandbytes is not installed.")
            use_4bit = False

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"[model] Loading {model_name}")
    print(f"[model] dtype={compute_dtype}, use_4bit={use_4bit}, device_map={device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    # If tokenizer added a pad token, resize embeddings.
    try:
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
    except Exception as exc:
        print(f"[tokenizer] WARNING: Could not resize token embeddings: {exc}")

    align_special_tokens(model, tokenizer)

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    if gradient_checkpointing:
        try:
            model.config.use_cache = False
        except Exception:
            pass
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    return model


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()

    # IMPORTANT:
    # Prefer the real full dataset. The old script looked for fraction_dataset.json first,
    # so Kaggle silently trained on only 3 records even when DATASET=data/full_data.json
    # was passed through run_train_kaggle.sh.
    data_env = env("DATA_PATH") or env("DATASET")
    default_data = existing_path(
        data_env,
        str(repo_default / "data" / "full_data.json"),
        str(repo_default / "full_data.json"),
        "/kaggle/working/AI_Logic_EXACT/data/full_data.json",
        "/kaggle/working/AI_Logic_EXACT/full_data.json",
        "/mnt/data/full_data.json",
        # Fraction dataset is now only a final fallback for local smoke tests.
        str(repo_default / "data" / "fraction_dataset.json"),
        str(repo_default / "fraction_dataset.json"),
        "/kaggle/working/AI_Logic_EXACT/data/fraction_dataset.json",
        "/kaggle/working/AI_Logic_EXACT/fraction_dataset.json",
        "/mnt/data/fraction_dataset.json",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=env("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument(
        "--data_path",
        default=str(default_data) if default_data else (data_env or "data/full_data.json"),
    )
    parser.add_argument("--output_dir", default=env("OUTPUT_DIR", str(repo_default / "outputs" / "lora_nl2fol")))
    parser.add_argument("--task_mode", choices=["nl2fol", "qa"], default=env("TASK_MODE", "nl2fol"))

    parser.add_argument("--max_seq_length", type=int, default=int(env("MAX_SEQ_LENGTH", "2048") or 2048))
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=float(env("EPOCHS", env("NUM_TRAIN_EPOCHS", "3")) or 3),
    )
    parser.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", "2e-4") or 2e-4))
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=int(env("TRAIN_BS", env("BATCH_SIZE", "1")) or 1),
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=int(env("EVAL_BS", "1") or 1),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(env("GRAD_ACCUM", "8") or 8))
    parser.add_argument("--eval_ratio", type=float, default=float(env("EVAL_RATIO", "0.1") or 0.1))
    parser.add_argument("--seed", type=int, default=int(env("SEED", "42") or 42))

    # Useful for smoke/mini-train runs.
    parser.add_argument("--max_train_samples", type=int, default=int(env("MAX_TRAIN_SAMPLES", "0") or 0))
    parser.add_argument("--max_valid_samples", type=int, default=int(env("MAX_VALID_SAMPLES", "0") or 0))

    parser.add_argument("--use_4bit", type=str2bool, default=str2bool(env("USE_4BIT", "1")))
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=str2bool(env("GRADIENT_CHECKPOINTING", "1")))
    parser.add_argument("--force_fp16", type=str2bool, default=str2bool(env("FORCE_FP16", "0")))
    parser.add_argument("--force_bf16", type=str2bool, default=str2bool(env("FORCE_BF16", "0")))

    parser.add_argument("--lora_r", type=int, default=int(env("LORA_R", "16") or 16))
    parser.add_argument("--lora_alpha", type=int, default=int(env("LORA_ALPHA", "32") or 32))
    parser.add_argument("--lora_dropout", type=float, default=float(env("LORA_DROPOUT", "0.05") or 0.05))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Pass --data_path or set DATA_PATH to your EXACT JSON file."
        )

    use_bf16, use_fp16, compute_dtype = choose_precision(
        force_fp16=args.force_fp16,
        force_bf16=args.force_bf16,
    )
    print(f"[precision] bf16={use_bf16}, fp16={use_fp16}, compute_dtype={compute_dtype}")

    if use_bf16 and use_fp16:
        raise RuntimeError("Internal precision bug: bf16 and fp16 are both True. This must never happen.")

    tokenizer = load_tokenizer(args.model_name)
    records = load_json_records(data_path)

    if args.task_mode == "nl2fol":
        samples = make_nl2fol_samples(records, tokenizer)
    else:
        samples = make_qa_samples(records, tokenizer)

    if not samples:
        raise ValueError(
            "No training samples were produced. Check that your JSON has premises-NL and premises-FOL fields."
        )

    train_dataset, eval_dataset = train_eval_split(samples, args.eval_ratio, args.seed)

    # Apply smoke/mini-train limits after the train/eval split, so validation
    # remains separate from training and the metadata reflects the actual used samples.
    if args.max_train_samples > 0 and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if (
        eval_dataset is not None
        and args.max_valid_samples > 0
        and len(eval_dataset) > args.max_valid_samples
    ):
        eval_dataset = eval_dataset.select(range(args.max_valid_samples))

    print(f"[data] records={len(records)}, samples={len(samples)}, train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset is not None else 0}")
    print("[data] sample preview:")
    print(train_dataset[0]["text"][:1000])

    model = load_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        compute_dtype=compute_dtype,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    optim = "paged_adamw_8bit" if args.use_4bit and is_bnb_available() else "adamw_torch"
    eval_strategy = "steps" if eval_dataset is not None else "no"

    raw_config_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 5,
        "logging_steps": 1,
        "save_steps": 25,
        "save_total_limit": 2,
        "eval_steps": 25,
        "evaluation_strategy": eval_strategy,
        "eval_strategy": eval_strategy,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "max_grad_norm": 0.3,
        "optim": optim,
        "report_to": "none",
        "remove_unused_columns": True,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_seq_length,
        "packing": False,
    }

    sft_args = make_config(SFTConfig, raw_config_kwargs)

    trainer = make_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    print("[train] Starting training...")
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "model_name": args.model_name,
        "data_path": str(data_path),
        "task_mode": args.task_mode,
        "samples": len(samples),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_train_samples": args.max_train_samples,
        "max_valid_samples": args.max_valid_samples,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "compute_dtype": str(compute_dtype),
        "use_4bit": args.use_4bit,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    (output_dir / "training_meta.json").write_text(dump_jsonish(meta), encoding="utf-8")
    print(f"[done] Saved LoRA adapter + tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
