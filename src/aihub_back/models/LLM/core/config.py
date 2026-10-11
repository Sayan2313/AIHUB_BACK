from typing import Literal
from dataclasses import dataclass
import json

@dataclass
class PeakConfig:
    model_type : str
    model_name : str
    vocab_size : int
    ctx_length : int
    d_model : int
    n_layers : int
    n_heads : int
    n_kv_heads : int
    d_ff : int

    @property
    def head_dim(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        elif  self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        return self.d_model // self.n_heads
@dataclass
class TrainerConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_learning_rate : float
    weight_decay: float
    max_grad_norm: float
    epochs: int
    warmup_ratio : float
    dropout: float
@dataclass
class InferenceConfig:
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty : float
    sampling_method : Literal["argmax","softmax"] = "softmax"
@dataclass
class DataConfig:
    dataset: str

@dataclass
class Config:
    model : PeakConfig
    training: TrainerConfig
    inference: InferenceConfig
    data : DataConfig

    @classmethod
    def from_json(cls,path : str):

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            model=PeakConfig(**data["model"]),
            training=TrainerConfig(**data["training"]),
            inference=InferenceConfig(**data["inference"]),
            data=DataConfig(**data["data"])
        )
