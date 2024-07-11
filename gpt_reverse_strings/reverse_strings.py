from copy import copy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from shared.gpt_model import GPT
from shared.util import *
import wandb


class ReverseStringsDataset(Dataset):
    @staticmethod
    def default_config():
        return Config(
            digits_min=None,
            digits_max=6,
            num_samples=10000,
            context_len=None,
        )

    def __init__(self, config: Config = None):
        config = config or self.default_config()
        self.digits_min = config.digits_min
        self.digits_max = config.digits_max
        self.num_samples = config.num_samples
        self.context_len = config.context_len
        samples = self.generate_samples_log_freq()
        self.data = self.generate_dataset(samples)

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
    vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', pad_token, ignore_token]
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

        # string, reverse string, colon separator and end of sequence
        return self.digits_max * 2 + 1 + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = torch.tensor(self.data[i][0]), torch.tensor(self.data[i][1])
        return x, y

    def generate_samples_log_freq(self):
        results = []
        for _ in tqdm(range(self.num_samples), desc="Generating samples"):
            # pick a random integer between digits_min (or 1) and digits
            digits_min = self.digits_min or 1
            problem = self.generate_sample(digits_min, self.digits_max) + [self.pad_token]
            # padding = [self.pad_token] * (self.get_context_length() - len(problem))
            # padded_problem = padding + problem
            results.append(problem)

        return results

    # pick a random integer with between digits_min and digits_max digits
    @staticmethod
    def generate_sample(digits_min: int, digits_max: int):
        len = random.randint(digits_min, digits_max)
        i = random.randint(10 ** (digits_min - 1) - 1, 10 ** len - 1)
        return [str(x) for x in (str(i) + ':' + str(i)[::-1])]

    def shift_and_pad(self, sample: list, width: int, offset: int) -> list:
        list = (['.'] * offset) + sample
        list = list + ['.'] * (width - len(list))
        return list[:width]

    def generate_dataset(self, samples: list) -> list:
        fixed_width = self.get_context_length()
        dataset = []
        for sample in tqdm(samples, desc="Generating dataset from samples"):
            # find the index of the ':'
            end_offset = fixed_width - sample.index(':') - 1
            for i in range(end_offset)[::-1]:
                dataset.append(
                    [self.shift_and_pad(sample, fixed_width, i + 1), self.shift_and_pad(sample, fixed_width, i)])

        # Encode the symbols and return the dataset as a tensor
        return [(self.encode(pair[0]), self.encode(pair[1])) for pair in dataset]

    @classmethod
    def evaluate_sample(cls, equation: np.array):
        equation = [e for e in equation if e != cls.pad_token and e != cls.ignore_token]  # Remove ignored
        try:
            a = ("".join(equation[:equation.index(':')]))
            b = ("".join(equation[equation.index(':') + 1:]))
        except ValueError:
            return False
        return a == b[::-1]

    @classmethod
    def evaluate_model(cls, model: GPT, dataset: Dataset, max_batches: int = None, quiet: bool = False):
        loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
        total_correct, total_count = 0, 0
        examples_correct, examples_incorrect = [], []
        for batch_num, (input_batch, target_batch) in enumerate(loader):

            # remove samples where target batch ends in '.'
            input_batch = input_batch[target_batch[:, -1] != cls.pad_token_id]

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
            input_decoded = cls.decode(input_batch[indices[i]])
            output_decoded = cls.decode(output_batch[i])
            passed = cls.evaluate_sample(output_decoded)
            correct += int(passed)
            if passed and len(examples_correct) < examples_max:
                examples_correct.append([input_decoded, output_decoded])
            elif not passed and len(examples_incorrect) < examples_max:
                examples_incorrect.append([input_decoded, output_decoded])

        return correct, input_batch.size(0)


if __name__ == '__main__':
    Data = ReverseStringsDataset
    config = Data.default_config().add(
        digits=4,
        num_samples=100,
        context_len=16,
    )
    train_dataset, test_dataset = Data(config).split()

    # print("Train dataset size:", len(train_dataset))
    # for i in range(config.num_samples):
    #     pair = train_dataset[i]
    #     print(Data.decode(pair[0]), Data.decode(pair[1]))
