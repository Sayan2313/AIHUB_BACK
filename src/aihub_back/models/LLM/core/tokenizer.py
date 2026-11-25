from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPEModel:
    def __init__(
        self,
        vocab_size: int,
        min_frequency: int = 3,
        special_tokens: list[str] | None = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.special_tokens = special_tokens or [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|unk|>",
            "<|context|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|tool_call|>",
            "<|tool_result|>"
        ]

        self.tokenizer = Tokenizer(
            BPE(unk_token="<|unk|>")
        )

        # Byte-level tokenization
        self.tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )

        self.tokenizer.decoder = ByteLevelDecoder()

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def train(self, texts,length=None):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True
        )
        self.tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
            length=length,
        )

    def encode(self, text: str,tokens:bool = False) -> list[int] | list[str]:
        if tokens:
            return self.tokenizer.encode(text).tokens
        else:
            return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int],skip_special_tokens:bool = True) -> str:
        return self.tokenizer.decode(ids,skip_special_tokens)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        tokenizer = Tokenizer.from_file(path)
        instance = cls(vocab_size=tokenizer.get_vocab_size())
        instance.tokenizer = tokenizer
        return instance

    def __len__(self):
        return self.tokenizer.get_vocab_size()
