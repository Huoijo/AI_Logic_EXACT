from __future__ import annotations

import os
import torch

class QwenGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", load_4bit: bool = True):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        do_sample = temperature > 0
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=self.tokenizer.eos_token_id)
        if do_sample:
            kwargs["temperature"] = temperature
        out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

def maybe_load_qwen() -> QwenGenerator | None:
    if os.environ.get("USE_LLM", "1") == "0":
        return None
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    load_4bit = os.environ.get("LOAD_4BIT", "1") == "1"
    try:
        return QwenGenerator(model_name=model_name, load_4bit=load_4bit)
    except ModuleNotFoundError as e:
        print(f"[warn] LLM dependencies missing ({e}); falling back to rule-based mode.", flush=True)
        return None
    except Exception as e:
        if os.environ.get("STRICT_LLM", "0") == "1":
            raise
        print(f"[warn] Could not load LLM ({type(e).__name__}: {e}); falling back to rule-based mode.", flush=True)
        return None
