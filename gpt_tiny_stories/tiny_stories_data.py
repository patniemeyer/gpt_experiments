import tiktoken
from torch.utils.data import Dataset
from shared.util import *
from datasets import load_dataset  # huggingface datasets


class TinyStoriesDataset(Dataset):
    @staticmethod
    def default_config():
        return Config(
            context_len=128,
        )

    def __init__(self, config: Config = None, split: str = 'train'):
        config = config or self.default_config()
        self.context_len = config.context_len
        self.dataset = load_dataset("roneneldan/TinyStories")[split]

    # Define the vocabulary
    encoder = tiktoken.get_encoding("gpt2")
    eot_token_id = encoder.eot_token
    pad_token_id = encoder.eot_token

    @classmethod
    def encode(cls, s: str) -> list[int]:
        return cls.encoder.encode(s)

    @classmethod
    def decode_ints(cls, tokens: list[int]) -> str:
        return cls.encoder.decode(tokens)

    # decode from a Tensor
    @classmethod
    def decode(cls, tokens: torch.Tensor) -> str:
        return cls.encoder.decode(tokens.tolist())

    @classmethod
    def get_vocab_size(cls):
        return cls.encoder.n_vocab

    def get_context_length(self):
        return self.context_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i]['text']
        x = self.encode(x)
        x = x + [self.encoder.eot_token] * (self.context_len - len(x))  # pad if needed
        x = x[:self.context_len]
        y = x[1:] + [self.encoder.eot_token]
        return torch.tensor(x), torch.tensor(y)


# Test
if __name__ == '__main__':
    dataset = TinyStoriesDataset(config=TinyStoriesDataset.default_config(), split='train')
    print("len(dataset):", len(dataset))
    for i in range(10):
        x, y = dataset[i]
        # print(len(x), len(y))
        print("\nsample:", x)
        # print("target:", y)

