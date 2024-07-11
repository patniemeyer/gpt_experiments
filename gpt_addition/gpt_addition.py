from copy import copy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from shared.gpt_model import GPT
from shared.util import *
import wandb


class GPTAdditionDataset(Dataset):
    @staticmethod
    def default_config():
        return Config(
            digits=2,
            digits_min=None,
            num_samples=None,
            context_len=None,
            permute=None,
        )

    def __init__(self, config: Config = None):
        config = config or self.default_config()
        self.digits = config.digits_max or config.digits
        self.digits_min = config.digits_min
        self.num_samples = config.num_samples
        self.context_len = config.context_len
        print(f"Sampling {self.num_samples} {self.digits} digit addition problems, covering "
              f"{self.num_samples / (10 ** (self.digits * 2))} of the space...")
        samples = self.generate_samples_log_freq()
        print("Generating dataset...")
        self.data = self.generate_dataset(samples)
        print("Dataset generated.")

    # Divide randomized dataset into train and test sets
    def split(self, test_frac: int = 0.2, test_max: int = 1000) -> (Dataset, Dataset):
        num_test = min(int(len(self.data) * test_frac), test_max)
        test = copy(self)  # shallow copy
        test.data = self.data[:num_test]
        train = copy(self)
        train.data = self.data[num_test:]
        return train, test

    # Define the vocabulary
    ignore_token = 'X'
    pad_token = '.'
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', pad_token, ignore_token]
    char_to_int = {char: i for i, char in enumerate(vocab)}
    pad_token_id = char_to_int[pad_token]
    ignore_token_id = char_to_int[ignore_token]
    int_to_char = {i: char for i, char in enumerate(vocab)}

    @classmethod
    def encode(cls, symbols: list[str]) -> list[int]:
        return [cls.char_to_int[s] for s in symbols]

    @classmethod
    def decode(cls, integers: torch.Tensor):
        return [cls.int_to_char[i.item()] for i in integers]

    @classmethod
    def get_vocab_size(self):
        return len(self.vocab)

    def get_context_length(self):
        if self.context_len is not None:
            return self.context_len

        # Calculate the max sequence length for up to n digit addition problems, e.g.
        # 2 * n + 2 'symbols' + (n + 1) 'answers digits' + 1 'end of sequence'
        return 3 * self.digits + 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = torch.tensor(self.data[i][0]), torch.tensor(self.data[i][1])
        return x, y

    # ['5', '+', '9', '7', '=', '1', '0', '2', '.']
    # ['7', '4', '+', '1', '=', '7', '5', '.']
    def generate_samples_log_freq(self):
        assert self.num_samples is not None, "requires sampling"

        results = []
        for count in tqdm(range(self.num_samples), "Generating samples"):
            # pick a random integer between digits_min (or 1) and digits
            digits_min = self.digits_min or 1
            problem = self.generate_sample(digits_min, self.digits) + [self.pad_token]
            results.append(problem)

        return results

    @staticmethod
    def generate_sample(digits_min: int, digits_max: int):
        i1_len = random.randint(digits_min, digits_max)
        i2_len = random.randint(digits_min, digits_max)
        i = random.randint(10 ** (digits_min - 1) - 1, 10 ** i1_len - 1)
        j = random.randint(10 ** (digits_min - 1) - 1, 10 ** i2_len - 1)
        sum_ = i + j
        return [str(x) for x in str(i) + "+" + str(j) + "=" + str(sum_)]

    def shift_and_pad(self, sample: list, width: int, offset: int) -> list:
        list = (['.'] * offset) + sample
        list = list + ['.'] * (width - len(list))
        return list[:width]

    def generate_dataset(self, samples: list) -> list:
        fixed_width = self.get_context_length()
        dataset = []
        for sample in tqdm(samples, "Generating dataset"):
            assert len(sample) <= fixed_width, "sample too long"
            end_offset = fixed_width - sample.index('=') - 1
            for i in range(end_offset)[::-1]:
                dataset.append(
                    [self.shift_and_pad(sample, fixed_width, i + 1), self.shift_and_pad(sample, fixed_width, i)])

        # Encode the symbols and return the dataset as a tensor
        return [(self.encode(pair[0]), self.encode(pair[1])) for pair in dataset]

    @classmethod
    def evaluate_equation(cls, equation: np.array):
        equation = [e for e in equation if e != cls.pad_token and e != cls.ignore_token]  # Remove ignored
        try:
            a = int("".join(equation[0:equation.index('+')]))
            b = int("".join(equation[equation.index('+') + 1:equation.index('=')]))
            c = int("".join(equation[equation.index('=') + 1:]))
        except ValueError:
            return False
        return a + b == c

    @classmethod
    def evaluate_model(cls, model: GPT, dataset: Dataset, max_batches: int = None, quiet: bool = False):
        loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
        total_correct, total_count = 0, 0
        examples_correct, examples_incorrect = [], []
        for batch_num, (input_batch, target_batch) in enumerate(loader):
            # remove samples where target batch ends in '.'
            input_batch = input_batch[target_batch[:, -1] != cls.pad_token_id]
            if input_batch.size(0) == 0:
                continue
            assert input_batch.size(0) != 0, "Empty batch"
            correct, count = cls.evaluate_batch(model, input_batch, examples_correct, examples_incorrect)
            total_correct += correct
            total_count += count
            if max_batches is not None and batch_num + 1 >= max_batches:
                break
        perc = total_correct / total_count
        if not quiet:
            print("Score: %d/%d = %.2f%% correct" % (total_correct, total_count, 100 * perc))
            print("Sample correct:\n", '\n'.join(map(str, examples_correct)), sep='')
            print("Sample wrong:\n", '\n'.join(map(str, examples_incorrect)), sep='')
            wandb.log({"score": perc})
        return total_correct, total_count

    @classmethod
    def evaluate_batch(cls, model: GPT, input_batch: torch.Tensor,
                       examples_correct: list = None, examples_incorrect: list = None,
                       examples_max: int = 5) -> (int, int):  # correct, total
        device = next(model.parameters()).device  # move the input batch to the target device

        input_batch = input_batch.to(device)

        assert input_batch.size(0) != 0, "Empty batch"

        # Generate output tokens shifted into the sequences (b, t)
        output_batch, indices = model.generate(input_batch, end_of_sequence_token_id=cls.pad_token_id,
                                               do_sampling=False)

        # validate the results
        correct = 0
        for i in range(len(output_batch)):
            input_decoded = GPTAdditionDataset.decode(input_batch[indices[i]])
            output_decoded = GPTAdditionDataset.decode(output_batch[i])
            passed = cls.evaluate_equation(output_decoded)
            correct += int(passed)
            if passed and len(examples_correct) < examples_max:
                examples_correct.append([input_decoded, output_decoded])
            elif not passed and len(examples_incorrect) < examples_max:
                examples_incorrect.append([input_decoded, output_decoded])

        return correct, input_batch.size(0)


if __name__ == '__main__':
    config = GPTAdditionDataset.default_config().add(
        digits_min=4,
        digits_max=4,
        num_samples=100,
        # context_len=10,  # long enough for 2 digits
        # context_len=13,  # long enough for 3 digits
        # context_len=16,  # long enough for 4 digits
    )
    dataset = GPTAdditionDataset(config)

    print("dataset size:", len(dataset), "for samples:", config.num_samples)
    for i in range(len(dataset)):
        pair = dataset[i]
        print(i, GPTAdditionDataset.decode(pair[0]), GPTAdditionDataset.decode(pair[1]))
