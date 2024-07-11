from copy import copy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from shared.gpt_model import GPT
from shared.util import *
from tqdm import tqdm
import wandb

black = '⬛'  # U+2B1B
color_list = [
    '🟦',  # U+1F7E6
    '🟥',  # U+1F7E5
    '🟩',  # U+1F7E9
    '🟨',  # U+1F7E8
    '⬜',  # U+2B1C
    '🟪',  # U+1F7EA
    '🟧',  # U+1F7E7
    '🩵',  # U+1FA75
    '🟫',  # U+1F7EB
]
# black = 'A'  # U+2B1B
# color_list = [
#     'B',  # U+1F7E6
#     'C',  # U+1F7E5
#     'D',  # U+1F7E9
#     'E',  # U+1F7E8
#     'F',  # U+2B1C
#     'G',  # U+1F7EA
#     'H',  # U+1F7E7
#     'I',  # U+1FA75
#     'J',  # U+1F7EB
# ]

# Data samples are a newline delimited grid followed by a target description in the form of:
# (rows, cols){color:count, color:count, ...}
# The color count map may optionally be of fixed length (show zero counts for missing colors)
"""
⬛⬛🟥⬛
⬛🟥🟦🟥
⬛⬛🟥⬛
⬛⬛⬛⬛
(4,4){🟦:1,🟥:4}
"""


class GridDataset(Dataset):
    @staticmethod
    def default_config():
        return Config(
            num_samples=1000,
            colors_min=1,
            colors_max=1,
            filled_min=1,
            filled_max=4,
            grid_rows_min=2,
            grid_rows_max=4,
            grid_cols_min=2,
            grid_cols_max=4,
            fixed_length_output=False,
        )

    def __init__(self, config: Config = None):
        config = config or self.default_config()

        self.num_samples = config.num_samples
        self.colors_min = config.colors_min
        self.colors_max = config.colors_max
        self.filled_min = config.filled_min
        self.filled_max = config.filled_max
        self.grid_rows_min = config.grid_rows_min
        self.grid_rows_max = config.grid_rows_max
        self.grid_cols_min = config.grid_cols_min
        self.grid_cols_max = config.grid_cols_max
        self.fixed_length_output = config.fixed_length_output

        print(f"Sampling {self.num_samples} grids with {self.colors_min}-{self.colors_max} colors")

        samples = self.generate_samples()
        print("Generating dataset...")
        self.data = self.generate_dataset(samples)
        print("Dataset generated.")

    # Divide randomized dataset into train and test sets
    # If remove_partial is true only complete test samples with no answer material will be retained.
    def split(self, test_frac: int = 0.2, test_max: int = 1000, remove_partial: bool = True) -> (Dataset, Dataset):
        num_test = min(int(len(self.data) * test_frac), test_max)
        test = copy(self)  # shallow copy
        test.data = self.data[:num_test]

        # Retain only the full test samples in the test data. i.e. remove shifted versions that
        # reveal part of the answer. Note that this may drastically reduce the number of test samples.
        if remove_partial:
            start_answer = self.char_to_int['(']
            test.data = [sample for sample in test.data if sample[0][-1] == start_answer]

        train = copy(self)
        train.data = self.data[num_test:]
        return train, test

    # Define the token vocabulary
    ignore_token = 'X'
    pad_token = '.'
    newline = '\n'
    vocab = ([digit for digit in '0123456789']
             + [black] + color_list
             + ['{', '}', '(', ')', ',', ':', pad_token, newline, ignore_token])
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

    # Calculate the max context length:
    # grid\n(rows, cols){color:count,color:count, ...}
    def get_context_length(self):
        max_rows_digits = len(str(self.grid_rows_max))
        max_cols_digits = len(str(self.grid_cols_max))
        max_count_digits = len(str(self.grid_rows_max * self.grid_cols_max))
        colors_max = len(color_list) if self.fixed_length_output else self.colors_max
        return (
            # grid puls newline on each row
                self.grid_rows_max * (self.grid_cols_max + 1)
                # (rows,cols)
                + 3 + max_rows_digits + max_cols_digits
                # {color:count,color:count, ...}
                + 2 + colors_max * (3 + max_count_digits)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = torch.tensor(self.data[i][0]), torch.tensor(self.data[i][1])
        return x, y

    def generate_samples(self):
        assert self.num_samples is not None, "requires samples"
        results = []
        for _ in tqdm(range(self.num_samples), desc="Generating samples"):
            # random grid size
            row_count = random.randint(self.grid_rows_min, self.grid_rows_max)
            cols_count = random.randint(self.grid_cols_min, self.grid_cols_max)
            colors_count = random.randint(self.colors_min, self.colors_max)
            colors: list[int] = random.sample(color_list, colors_count)
            filled_count = random.randint(self.filled_min, self.filled_max)
            problem = self.generate_sample(row_count, cols_count, colors, filled_count, self.fixed_length_output)
            results.append(problem)

        return results

    @staticmethod
    def generate_sample(
            rows: int, cols: int, colors: list[int], fill_count: int, fixed_length_output: bool
    ) -> list[str]:
        # create an array representing the grid, filled with zeros
        grid = [[black] * cols for _ in range(rows)]
        # fill_count random cells, cycling through the colors
        for i in range(fill_count):
            color = colors[i % len(colors)]
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            grid[row][col] = color

        # flatten the grid
        flat = [item for row in grid for item in (row + ['\n'])]

        # create the target description
        colorspace = color_list if fixed_length_output else colors
        target = f"({rows},{cols})" + "{" + ",".join(
            [f"{color}:{flat.count(color)}" for color in sorted(colorspace)]) + "}"
        # break target into an array of characters
        target = [char for char in target]

        return flat + target

    # Position the `sample` at starting `offset` in a context window of `width`
    def shift_and_pad(self, sample: list, width: int, offset: int) -> list:
        list = ([self.pad_token] * offset) + sample
        list = list + [self.pad_token] * (width - len(list))
        return list[:width]

    # Slide the sample across the context window, starting with the "answer" portion of the sample
    # truncated at the right edge (using padding on the left as needed) and incrementally revealing
    # tokens of the answer.
    # If full_window is true, the sample will slide across the entire context window which may
    # entail adding padding on the right.  Else it will stop at the right edge of the sample
    # plus one to allow for an end of sequence pad character.
    def generate_dataset(self, samples: list, full_window: bool = False) -> list:
        context_length = self.get_context_length()
        dataset = []
        for sample in tqdm(samples, desc="Generating dataset"):
            assert len(sample) <= context_length, "sample too long"
            description_start_char = '('
            # offset to position the start character at right of the context window
            right_offset = context_length - sample.index(description_start_char) - 1
            # The left offset at which to stop sliding
            left_offset = 0 if full_window else (context_length - len(sample) - 1)
            # index ranges from left_offset (sample at left of context) to offset (desired position at right)
            for i in range(left_offset, right_offset)[::-1]:
                dataset.append(
                    [self.shift_and_pad(sample, context_length, i + 1), self.shift_and_pad(sample, context_length, i)])

        # Encode the symbols and return the dataset as a tensor
        return [(self.encode(pair[0]), self.encode(pair[1])) for pair in dataset]

    """
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', 
    '⬛', '⬛', '⬛', '\n', '⬛', '🟦', '⬛', '\n', '⬛', '🟦', '⬛', '\n', '(', '3', ',', '3', ')', '{', '🟦', ':', '2', 
    '}']
    """

    @classmethod
    def evaluate_grid_description(cls, input):
        # Remove padding and ignored tokens
        input_str = ''.join([e for e in input if e != cls.pad_token and e != cls.ignore_token])
        # print("input str:", input_str)

        # Extract dimensions and color counts from the string
        dimensions_part = input_str[input_str.find('(') + 1:input_str.find(')')]
        color_count_part = input_str[input_str.find('{') + 1:input_str.find('}')]

        # Parse dimensions
        rows, cols = map(int, dimensions_part.split(','))
        # print(f"Rows: {rows}, Cols: {cols}")

        # Parse color counts
        color_counts = {}
        color_items = color_count_part.split(',')
        for item in color_items:
            key, value = item.split(':')
            color_counts[key.strip()] = int(value.strip())
        # print("Color counts:", color_counts)

        # Validate grid size
        actual_rows = input_str.count('\n')  # Rows are separated by newlines
        actual_cols = len(input_str[:input_str.find('\n')])
        # print(f"Actual Rows: {actual_rows}, Actual Cols: {actual_cols}")

        if (rows, cols) != (actual_rows, actual_cols):
            return False

        # Validate color counts
        actual_color_counts = {}
        for char in input:
            if char in color_list:
                actual_color_counts[char] = input.count(char) - 1  # Subtract 1 for the color count part
        # print("Actual color counts:", actual_color_counts)

        return color_counts == actual_color_counts

    @classmethod
    def evaluate_model(cls, model: GPT, dataset: Dataset, max_batches: int = None, quiet: bool = False):
        loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
        total_correct, total_count = 0, 0
        examples_correct, examples_incorrect = [], []
        for batch_num, (input_batch, target_batch) in enumerate(loader):
            # Remove samples where target batch ends in '.'
            # input_batch = input_batch[target_batch[:, -1] != cls.pad_token_id]
            if input_batch.size(0) == 0:
                continue
            assert input_batch.size(0) != 0, "Empty batch"
            correct, count = cls.evaluate_batch(
                model, input_batch, examples_correct, examples_incorrect,
                max_gen_token_count=dataset.get_context_length()
            )
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
                       examples_max: int = 3,
                       max_gen_token_count=32,
                       ) -> (int, int):  # correct, total
        device = next(model.parameters()).device  # move the input batch to the target device
        input_batch = input_batch.to(device)
        assert input_batch.size(0) != 0, "Empty batch"

        # Generate output tokens shifted into the sequences (b, t)
        output_batch, indices = model.generate(
            input_batch,
            end_of_sequence_token_id=cls.pad_token_id,
            max_gen_token_count=max_gen_token_count,
            do_sampling=False)

        # validate the results
        correct = 0
        for i in range(len(output_batch)):
            input_decoded = GridDataset.decode(input_batch[indices[i]])
            output_decoded = GridDataset.decode(output_batch[i])

            passed = False
            try:
                passed = cls.evaluate_grid_description(output_decoded)
            except Exception as e:
                # print(e)
                ...

            correct += int(passed)
            if passed and len(examples_correct) < examples_max:
                examples_correct.append([input_decoded, output_decoded])
            elif not passed and len(examples_incorrect) < examples_max:
                examples_incorrect.append([input_decoded, output_decoded])

        return correct, input_batch.size(0)


# Main
if __name__ == '__main__':

    config = GridDataset.default_config().add(
        num_samples=100,
        colors_min=1,
        colors_max=4,
        filled_min=1,
        filled_max=8,
        grid_rows_min=2,
        grid_rows_max=8,
        grid_cols_min=2,
        grid_cols_max=8,
        fixed_length_output=False,
    )
    dataset = GridDataset(config)


    # Show some training and target data pairs
    def show_dataset(dataset):
        print("context length:", dataset.get_context_length())
        print("dataset size:", len(dataset), "for samples:", config.num_samples)

        # find the longest sample minus padding tokens
        max_len = max([len(sample[0]) - sample[0].count(dataset.pad_token_id) for sample in dataset.data])
        print("max sample length:", max_len)

        for i in range(len(dataset)):
            pair = dataset[i]
            # print(i, GridDataset.decode(pair[0]), GridDataset.decode(pair[1]))
            print(i, '\n',
                  ''.join(GridDataset.decode(pair[0])).replace('\n', '-'), '\n',
                  ''.join(GridDataset.decode(pair[1])).replace('\n', '-'))


    def show_samples():
        samples = dataset.generate_samples()
        # for i in range(min(10, len(samples))):
        for i in range(len(samples)):
            print(''.join(samples[i]))


    # test data
    # dataset = dataset.split()[1]

    # show_dataset(dataset)
    show_samples()
