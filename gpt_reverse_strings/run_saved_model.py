import wandb
from reverse_strings import ReverseStringsDataset as Data
from shared.gpt_model import GPT
from shared.util import *
from icecream import ic


def toBatch(sample: list):
    return torch.tensor([Data.encode(sample)])


def showresults1():
    # batch1 = toBatch(['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', '1', '2', '+', '3', '4', '='])
    # batch1 = toBatch(['X', 'X', 'X', 'X', '1', '1', '+', '1', '1', '='])
    # batch1 = toBatch(['1', '+', '2', '=', '3', '.', '1', '+', '2', '='])
    batch1 = toBatch(['X', 'X', '1', '+', '2', '='])
    output_batch, indices = model.generate(
        batch1, end_of_sequence_token_id=Data.pad_token_id, do_sampling=False, maintain_len=False)
    result = Data.decode(output_batch[0])
    print(result)
    passed = Data.evaluate_equation(result)
    print("Passed" if passed else "Failed")


def runtest(model: GPT, config:Config, context_len: int, start: int, upto: int):
    # Sample the trained model on different context length
    print("Model train on digits: ", config.data.digits_max)
    model.eval()
    for i in range(start, upto):
        with torch.no_grad():
            test_dataset = Data(Config(context_len=context_len, digits_max=i, digits_min=i, num_samples=250))
            # test only full equations
            test_dataset = [(sample, target) for sample, target in test_dataset if
                            sample[-1] == Data.char_to_int[':']]
            # print("dataset length: ", len(test_dataset))
            # for i in range(20):
            #     pair = testmore_dataset[i]
            #     print(Data.decode(pair[0]), Data.decode(pair[1]))
            # exit()
            correct, total = Data.evaluate_model(
                model, test_dataset, max_batches=500, quiet=True)
            print(f"Digits: {i}, Correct: {correct}, Accuracy: {correct / total}")
            wandb.log({f"digits_{i}": correct / total})


if __name__ == '__main__':
    wandb.init(mode="disabled")
    seed_everything(1337)

    from learn_reverse_strings import config

    # ic(config)
    train_dataset, test_dataset = Data(config.data).split()
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()

    model = GPT(config.model)
    ic(model.load_state_dict(torch.load("reverse_strings.pth")))
    model.eval()

    # showresults1()
    runtest(model, config, context_len=16, start=4, upto=8)
