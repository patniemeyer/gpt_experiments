import wandb
from gpt_addition import GPTAdditionDataset as Data
from shared.gpt_model import GPT
from shared.util import *
from icecream import ic


def toBatch(sample: list):
    return torch.tensor([Data.encode(sample)])


def showlogits1():
    #              0    1    2    3    4    5    6    7    8    9
    b0 = toBatch(['X', 'X', 'X', 'X', '4', '3', '+', '1', '1', '='])
    b1 = toBatch(['X', 'X', 'X', 'X', '4', '3', '+', '1', '1', '='])
    b2 = toBatch(['X', '4', '3', '+', '1', '1', '='])
    b3 = toBatch(['1', '2', '+', '3', '4', '='])
    b4 = toBatch(['1', '+', '1', '='])
    b5 = toBatch(['X', 'X', 'X', 'X', '1', '2', '+', '3', '4', '='])  # default 2 digit len
    for b in [b0, b1, b2, b3, b4, b5]:
        print()
        print(Data.decode(b[0]))
        logits, _ = model(b)
        logits = torch.argmax(logits, dim=-1)
        for i in range(len(logits)):
            print(Data.decode(logits[i]))


def showlogits2():
    # b = toBatch(['X', '4', '3', '+', '1', '1', '='])
    b = toBatch(['X', 'X', 'X', 'X', '1', '2', '+', '3', '4', '='])
    print("input: ", Data.decode(b[0]))
    logits, _ = model(b)
    logits = torch.argmax(logits, dim=-1)
    print("logits:", Data.decode(logits[0]))
    print()

    # b2 = torch.cat((b[0], logits[0, -1].unsqueeze(0))).unsqueeze(0)
    b2 = torch.cat((b[0][1:], logits[0, -1].unsqueeze(0))).unsqueeze(0)  # maintain length
    print("input: ", Data.decode(b2[0]))
    logits, _ = model(b2)
    logits = torch.argmax(logits, dim=-1)
    print("logits:", Data.decode(logits[0]))


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


def runtest_old():
    # Sample the trained model on different context length
    model.eval()
    with torch.no_grad():
        testmore_dataset = Data(Config(digits=4, digits_min=4, num_samples=10000))
        # test only full equations
        testmore_dataset = [(sample, target) for sample, target in testmore_dataset if
                            sample[-1] == Data.char_to_int['=']]
        print("dataset length: ", len(testmore_dataset))
        # for i in range(20):
        #     pair = testmore_dataset[i]
        #     print(Data.decode(pair[0]), Data.decode(pair[1]))
        # exit()

        Data.evaluate_model(model, testmore_dataset, max_batches=500)
    model.train()  # revert model to training mode


def runtest(model: GPT, config: Config, context_len: int, start: int, upto: int):
    # Sample the trained model on different context length
    print("\nModel train on digits: ", config.data.digits_max)
    model.eval()
    for i in range(start, upto + 1):
        with torch.no_grad():
            config = Config(context_len=context_len, digits_max=i, digits_min=i, num_samples=2000)
            test_dataset = Data(config)
            # test only base equations (note that we'll still have num_samples)
            test_dataset = [(sample, target) for sample, target in test_dataset if
                            sample[-1] == Data.char_to_int['=']]
            # for i in range(20):
            #     pair = test_dataset[i]
            #     print(Data.decode(pair[0]), Data.decode(pair[1]))
            correct, total = Data.evaluate_model(
                model, test_dataset, max_batches=500, quiet=True)
            print(f"Digits: {i}, Correct: {correct}, Accuracy: {correct / total}")
            wandb.log({f"digits_{i}": correct / total})


if __name__ == '__main__':
    wandb.init(mode="disabled")
    seed_everything(1337)
    from learn_addition import config

    # ic(config)
    train_dataset, test_dataset = Data(config.data.add(num_samples=0)).split()
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()

    model = GPT(config.model)
    ic(model.load_state_dict(torch.load("gpt_addition.pth")))
    model.eval()

    # showlogits1()
    # showlogits2()
    # showresults1()
    runtest(model, config, context_len=None, start=1, upto=3)
