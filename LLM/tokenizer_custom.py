from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder

class CustomTokenizer:
    def __init__(self,vocab_path:str=None):
        if vocab_path:
            self.tokenizer = Tokenizer.from_file(vocab_path)
        else:
            self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder()
    def train(self, files, vocab_size=10000, min_frequency=2):
        """
        files: list of text file paths
        """
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>","<unk>","<eos>"]
        )

        self.tokenizer.train(files, trainer)
    def save(self, path):
        """
        Saves vocab.json + merges.txt
        """
        self.tokenizer.model.save(path)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)