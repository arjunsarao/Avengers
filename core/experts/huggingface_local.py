from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def _get_torch_dtype(dtype_name: str | None):
    if not dtype_name or dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def _clean_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


class LocalHuggingFaceChatClient:
    """Small OpenAI-compatible chat facade backed by a local HF causal LM."""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_name = model_config["name"]
        self.model_path = model_config.get("model_path", self.model_name)
        self.adapter_path = model_config.get("adapter_path")
        self.max_new_tokens = model_config.get("max_new_tokens", 2048)
        self.local_files_only = model_config.get("local_files_only", True)
        self.trust_remote_code = model_config.get("trust_remote_code", True)
        self.device_map = model_config.get("device_map", "auto")
        self.torch_dtype = _get_torch_dtype(model_config.get("torch_dtype", "auto"))
        self.generation_kwargs = model_config.get("generation_kwargs", {})
        self._lock = threading.Lock()

        tokenizer_path = model_config.get("tokenizer_path") or self.adapter_path or self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
        }
        if model_config.get("attn_implementation"):
            model_kwargs["attn_implementation"] = model_config["attn_implementation"]

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading Hugging Face adapters requires peft. Install it with `pip install peft`."
                ) from exc
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                local_files_only=self.local_files_only,
            )
        self.model.eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        logger.info(f"Loaded local Hugging Face model: {self.model_name} from {self.model_path}")

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 1.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        prompt_ids = self._encode_messages(messages)
        outputs = []
        completion_tokens = 0
        max_new_tokens = max_tokens or kwargs.get("max_new_tokens") or self.max_new_tokens
        do_sample = temperature is not None and temperature > 0
        generate_kwargs = _clean_generation_kwargs(
            {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **self.generation_kwargs,
            }
        )

        with self._lock, torch.inference_mode():
            for _ in range(n):
                generated = self.model.generate(prompt_ids, **generate_kwargs)
                new_tokens = generated[0, prompt_ids.shape[-1] :]
                completion_tokens += int(new_tokens.shape[-1])
                outputs.append(
                    self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                )

        choices = [
            SimpleNamespace(index=i, message=SimpleNamespace(content=text))
            for i, text in enumerate(outputs)
        ]
        usage = SimpleNamespace(
            prompt_tokens=int(prompt_ids.shape[-1]) * n,
            completion_tokens=completion_tokens,
        )
        return SimpleNamespace(choices=choices, usage=usage)

    def _encode_messages(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            encoded = self.tokenizer(prompt + "\nassistant:", return_tensors="pt").input_ids
        model_device = next(self.model.parameters()).device
        return encoded.to(model_device)


class LocalHuggingFaceEmbeddingClient:
    """Local embedding helper with the same get/batch methods used by routers."""

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_path = model_path
        self.local_files_only = self.config.get("local_files_only", True)
        self.trust_remote_code = self.config.get("trust_remote_code", True)
        self.device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = _get_torch_dtype(self.config.get("torch_dtype", "auto"))
        self.max_length = self.config.get("max_length", 8192)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()
        logger.info(f"Loaded local Hugging Face embedding model: {self.model_path}")

    def get(self, text: str) -> List[float]:
        return self.batch([text])[0]

    def batch(self, texts: List[str], max_batch_size: int = 32) -> List[List[float]]:
        embeddings: List[List[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), max_batch_size):
                batch = texts[start : start + max_batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.extend(pooled.cpu().tolist())
        return embeddings

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
