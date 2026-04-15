from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace,ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class CustomTokenizer:
    def __init__(self,vocab_path:str=None):
        if vocab_path:
            self.tokenizer = Tokenizer.from_file(vocab_path)
        else:
            self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
    def trainer(self, vocab_size=8000, min_frequency=2):
        """
        :returns: trainer object(including special_tokens)
        """
        return BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>","<unk>","<bos>","<eos>"]
        )

    def save(self, path):
        """
        Saves vocab_small.json + merges.txt
        """
        self.tokenizer.save(path)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)