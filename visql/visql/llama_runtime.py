"""Llama runtime — text and vision wrappers.

Uses MANUAL chat templates (string concatenation with Llama-3 special tokens)
to avoid the `apply_chat_template -> KeyError: 'shape'` bug that hits certain
transformers/tokenizer version combos.

Also includes SQLCoder wrapper (used for the LoRA-tuned baseline in evals).
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import torch

from . import config as cfg

# Singleton model handles — lazy loaded, shared across instances to avoid
# reloading 6GB+ every time someone instantiates a wrapper.
_TEXT_MODEL = _TEXT_TOKEN = None
_VISION_MODEL = _VISION_PROC = None
_SQL_MODEL = _SQL_TOKEN = None

# ════════════════════════════════════════════════════════════════════
# LLAMA TEXT (router, etc.)
# ════════════════════════════════════════════════════════════════════
class LlamaText:
    """Llama-3.1-8B text generation. Used by the router (slide 5)."""

    def __init__(self, model_name: str = cfg.LLAMA_TEXT_MODEL,
                 load_in_4bit: bool = True, lora_path: Optional[str] = None):
        global _TEXT_MODEL, _TEXT_TOKEN
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        if _TEXT_MODEL is None or _TEXT_TOKEN is None:
            print(f"[LlamaText] loading {model_name}")
            _TEXT_TOKEN = AutoTokenizer.from_pretrained(model_name)
            if _TEXT_TOKEN.pad_token is None:
                _TEXT_TOKEN.pad_token = _TEXT_TOKEN.eos_token

            kwargs = {"torch_dtype": cfg.DTYPE, "device_map": "auto"}
            if load_in_4bit and cfg.DEVICE == "cuda":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            _TEXT_MODEL = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            _TEXT_MODEL.eval()

        self.model = _TEXT_MODEL
        self.tokenizer = _TEXT_TOKEN

        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()

    @torch.no_grad()
    def chat(self, system: str, user: str,
             max_new_tokens: int = cfg.TEXT_MAX_NEW_TOKENS,
             temperature: float = cfg.TEXT_TEMPERATURE) -> str:
        """Manual Llama-3 chat template — avoids tokenizer.apply_chat_template bug."""
        nl = chr(10)
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + nl + nl
            + system
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>" + nl + nl
            + user
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + nl + nl
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            eos_ids.append(eot_id)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
        response = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

# ════════════════════════════════════════════════════════════════════
# LLAMA VISION (style extraction, schema reading from screenshots)
# ════════════════════════════════════════════════════════════════════
class LlamaVision:
    """Llama-3.2-11B-Vision. Used by the multimodal subsystem (slide 6)."""

    def __init__(self, model_name: str = cfg.LLAMA_VISION_MODEL,
                 load_in_4bit: bool = True):
        global _VISION_MODEL, _VISION_PROC
        from transformers import (
            MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
        )

        if _VISION_MODEL is None or _VISION_PROC is None:
            print(f"[LlamaVision] loading {model_name}")
            _VISION_PROC = AutoProcessor.from_pretrained(model_name)
            kwargs = {"torch_dtype": cfg.DTYPE, "device_map": "auto"}
            if load_in_4bit and cfg.DEVICE == "cuda":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            _VISION_MODEL = MllamaForConditionalGeneration.from_pretrained(model_name, **kwargs)
            _VISION_MODEL.eval()

        self.model = _VISION_MODEL
        self.processor = _VISION_PROC

    @torch.no_grad()
    def chat_image(self, image, prompt: str,
                   max_new_tokens: int = cfg.VISION_MAX_NEW_TOKENS,
                   temperature: float = 0.0) -> str:
        from PIL import Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            top_p=0.9,
        )
        response = self.processor.decode(
            out[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

# ════════════════════════════════════════════════════════════════════
# SQLCODER (LoRA baseline — slide 7)
# ════════════════════════════════════════════════════════════════════
class SQLCoder:
    """SQLCoder-7B-2 with optional LoRA adapter.

    Used for the LoRA SQL eval baseline (slide 10: 0.74 exec-acc).
    Production pipeline uses Claude (see SQLAgent), but this is what we
    quantitatively compared against.
    """

    def __init__(self, model_name: str = cfg.SQL_BASE_MODEL,
                 lora_path: Optional[str] = None, load_in_4bit: bool = True):
        global _SQL_MODEL, _SQL_TOKEN
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        if _SQL_MODEL is None or _SQL_TOKEN is None:
            print(f"[SQLCoder] loading {model_name}")
            _SQL_TOKEN = AutoTokenizer.from_pretrained(model_name)
            if _SQL_TOKEN.pad_token is None:
                _SQL_TOKEN.pad_token = _SQL_TOKEN.eos_token

            kwargs = {"torch_dtype": cfg.DTYPE, "device_map": "auto"}
            if load_in_4bit and cfg.DEVICE == "cuda":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            _SQL_MODEL = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            _SQL_MODEL.eval()

        self.model = _SQL_MODEL
        self.tokenizer = _SQL_TOKEN
        self._lora_attached = False

        if lora_path:
            self.attach_lora(lora_path)

    def attach_lora(self, lora_path: str) -> None:
        from peft import PeftModel
        if self._lora_attached:
            return
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self._lora_attached = True
        print(f"[SQLCoder] attached LoRA adapter from {lora_path}")

    @torch.no_grad()
    def generate_sql(self, prompt: str,
                     max_new_tokens: int = cfg.SQL_MAX_NEW_TOKENS,
                     temperature: float = cfg.SQL_TEMPERATURE) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else 1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        return self._extract_sql(text)

    @staticmethod
    def _extract_sql(text: str) -> str:
        text = text.strip()
        if "```" in text:
            for p in text.split("```")[1:]:
                p = p.lstrip("sql").lstrip("SQL").strip()
                if any(p.upper().startswith(k) for k in ("SELECT", "WITH")):
                    return p.split(";")[0].strip() + ";"
        for kw in ("WITH", "SELECT"):
            i = text.upper().find(kw)
            if i >= 0:
                tail = text[i:]
                return (tail.split(";")[0].strip() + ";") if ";" in tail else tail.strip()
        return text
