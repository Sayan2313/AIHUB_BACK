import os
from typing import Literal

import torch
from fastapi import UploadFile
from torch import nn

from aihub_back.models.LLM.core.config import Config, InferenceConfig
from aihub_back.models.LLM.core.llmbaseclass import LLMBaseclass
from aihub_back.models.LLM.core.peakmodel import PeakModel
from aihub_back.models.LLM.core.tokenizer import BPEModel
from aihub_back.models.LLM.peak.sampling import sample_next_token,apply_repetition_penalty


class PeakInference:
    def __init__(
            self,
            model: nn.Module,
            tokenizer,
            config: InferenceConfig,
            device: Literal["cpu", "cuda"] = "cpu",
    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        print(f"Using : {device}")

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            raise KeyError("'model' key needed")
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()

    @staticmethod
    def _format_prompt(
            user_prompt: str,
            context: str | None = None,
            system_prompt: str | None = None,
    ) -> str:
        formatted = "<|bos|>"

        if system_prompt:
            formatted += f"<|system|>{system_prompt.strip()}<|eos|>"
        elif context:
            # Default strict grounding rule matching your SQuAD RAG training rows
            formatted += (
                "<|system|>Answer using only the context. If the answer is not in"
                " the context, say No.<|eos|>"
            )

        if context:
            formatted += f"<|context|>{context.strip()}"

        formatted += f"<|user|>{user_prompt.strip()}<|eos|><|assistant|>"
        return formatted

    @torch.inference_mode()
    def generate(
            self,
            prompt: str,
            context: str | None = None,
            system_prompt: str | None = None,
            skip_special_tokens: bool = True,
    ) -> str:
        formatted_prompt = self._format_prompt(prompt, context, system_prompt)
        encoded = self.tokenizer.encode(formatted_prompt)
        prompt_ids = encoded.ids if hasattr(encoded, "ids") else encoded

        # Truncate from the left if prompt exceeds 256 context window, keeping <|assistant|> at the end
        if len(prompt_ids) >= 256:
            prompt_ids = prompt_ids[-240:]

        input_ids = torch.tensor(
            prompt_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        prompt_len = input_ids.size(1)

        remaining_ctx_length = min(
            self.config.max_tokens,
            256 - prompt_len,
        )

        kv_cache = None
        for _ in range(remaining_ctx_length):
            model_input = input_ids[:, -1:] if kv_cache is not None else input_ids

            logits, kv_cache = self.model(model_input, kv_cache=kv_cache)
            logits = logits[0, -1, :]

            # Apply repetition penalty only over generated tokens (or full sequence)
            logits = apply_repetition_penalty(self.config, logits, input_ids)
            next_token = sample_next_token(self.config, logits).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == self.eos_id:
                break

        # Decode ONLY the newly generated assistant tokens
        generated_ids = input_ids[0, prompt_len:].tolist()
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        ).strip()


class PeakBase(LLMBaseclass):
    def __init__(self)->None:
        super().__init__()
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = Config.from_json(file_dir + "/peak_config.json")
        self.model = PeakModel(self.config.model,dropout=0.0,is_inference_mode=True)
        self.tokenizer = BPEModel.load(file_dir + "/peak_vocab.json")
        self.inference = PeakInference(self.model,self.tokenizer,self.config.inference)
        self.inference.load(file_dir + "/peak_weight.pt")
    async def response(self,text:str,file: UploadFile | None = None)->str:
        return self.inference.generate(text)