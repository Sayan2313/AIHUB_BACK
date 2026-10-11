import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import UploadFile
from torch import nn


from aihub_back.models.LLM.core.config import Config, InferenceConfig
from aihub_back.models.LLM.core.llmbaseclass import LLMBaseclass
from aihub_back.models.LLM.core.peakmodel import PeakModel
from aihub_back.models.LLM.core.tokenizer import BPE


class PeakInference:
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 config: InferenceConfig,
    ):
        self.device = "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        print(f"Using : {self.device}")
    def load(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __sample_next_token(self, logits):
        if self.config.sampling_method == "argmax":
            return torch.argmax(logits,dim=-1).reshape(1)
        else:
            # Temperature
            logits = logits / self.config.temperature
            # Top-k sampling
            if self.config.top_k is not None:
                values, indices = torch.topk(
                    logits,
                    self.config.top_k,
                    dim=-1
                )
                # Mask For -inf
                mask = torch.ones_like(logits,dtype=torch.bool)
                mask.scatter_(dim=-1, index=indices, value=False)
                logits[mask] = -torch.inf
            # Convert to probability
            probs = torch.softmax(logits, dim=-1)

            if self.config.top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                # Sample from sorted probabilities
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                # Convert sorted position -> original token ID
                next_token = torch.gather(
                    sorted_indices,
                    -1,
                    sampled
                )
            else:
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

            return next_token
    def __apply_repetition_penalty(self,logits: torch.Tensor,generated_ids: torch.Tensor):
        penalty = self.config.repetition_penalty
        if penalty == 1.0:
            return logits
        token_ids = generated_ids[0].unique()
        token_logits = logits[token_ids]
        token_logits = torch.where(
            token_logits > 0,
            token_logits / penalty,
            token_logits * penalty
        )
        logits[token_ids] = token_logits
        return logits
    @torch.inference_mode
    def generate(self, prompt: str,keep_special_tokens=False) -> str:
        # Encode prompt
        input_ids = torch.tensor(self.tokenizer.encode(
            prompt,
        )).unsqueeze(0).to(self.device)
        generated_ids = input_ids.clone()

        # No cache initially
        kv_cache = None

        for _ in range(self.config.max_tokens-len(input_ids)):
            if kv_cache is not None:
                model_input = generated_ids[:, -1:]
            else:
                model_input = generated_ids

            # Forward pass
            logits,kv_cache = self.model(model_input,kv_cache=kv_cache)
            # Last token logits
            logits = logits[0, -1, :]
            # Repetition penalty
            logits = self.__apply_repetition_penalty(
                    logits,
                    generated_ids
            )
            next_token = self.__sample_next_token(logits)
            next_token = next_token.unsqueeze(0) # Match the Batch Dimension
            # Store generated token
            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=-1
            )
            # EOS
            if (
                self.tokenizer.special_tokens["<EOS>"] is not None
                and next_token.item() == self.tokenizer.special_tokens["<EOS>"]
            ):
                break
        return self.tokenizer.decode(
            generated_ids.squeeze().tolist(),
            keep_special_tokens=keep_special_tokens
        )


class PeakBase(LLMBaseclass):
    def __init__(self)->None:
        super().__init__()
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = Config.from_json(file_dir + "/peak_config.json")
        self.model = PeakModel(self.config.model,dropout=0.0,is_inference_mode=True)
        self.tokenizer = BPE.load(file_dir,"peak")
        self.inference = PeakInference(self.model,self.tokenizer,self.config.inference)
        self.inference.load(file_dir + "/peak_weight.pt")
    async def response(self,text:str,file: Optional[UploadFile] = None)->str:
        enhanced_prompt : str = "<BOS>" + text
        return self.inference.generate(enhanced_prompt)