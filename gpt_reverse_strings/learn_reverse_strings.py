from reverse_strings import ReverseStringsDataset as Data
from shared.gpt_model import GPT
from shared.train import Trainer
from shared.util import *
import wandb


def run(config: Config, name: str = None):
    seed_everything(1337 + 42)

    # Construct train and test datasets
    train_dataset, test_dataset = Data(config.data).split()

    # Show some data
    print("Train dataset size:", len(train_dataset))
    # for i in range(50):
    #     pair = train_dataset[i]
    #     print(Data.decode(pair[0]), Data.decode(pair[1]))
    # exit()

    # Construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()
    print(config, '\n')

    model = GPT(config.model)

    # WandB setup
    wandb.init(
        entity="patniemeyer",
        project="transformer-stringsrev-project",
        # group="strings-reverse",
        config=config,
        name=name,
        # mode="online",
        mode="disabled",
    )
    wandb.watch(model, log="all")

    # Construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # Training callback
    def batch_end_callback(trainer: Trainer):
        if trainer.iter_num % 500 == 0 and trainer.iter_num > 0:
            # Evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                print("\nEvaluate on training data:")
                Data.evaluate_model(model, train_dataset, max_batches=10)
                print("\nEvaluate on test data:")
                correct, count = Data.evaluate_model(model, test_dataset)
                # if correct >= count * 0.99:
                # if correct == count:
                if trainer.epoch_num >= 4:
                    print(f"Stopping at {trainer.iter_num} iterations, {trainer.epoch_num} epochs.")
                    trainer.exit = True
                from run_saved_model import runtest
                runtest(model, config, context_len=config.model.context_len, start=4, upto=8)

            model.train()  # revert model to training mode

            print("Saving model...")
            torch.save(model.state_dict(), "reverse_strings.pth")

    trainer.set_callback('on_batch_end', batch_end_callback)

    # Begin training!
    trainer.run()


if __name__ == '__main__':
    config = Config(
        model=GPT.default_config().add(
            n_layer=2,
            # ignore_index=Data.ignore_token_id,
            ignore_index=None,
            # TODO:
            embd_pdrop=0.15,
            block_pdrop=0.15,
        ),
        data=Data.default_config().add(
            # digits_min=3,
            digits_max=4,
            num_samples=int(3e5),
            context_len=16,
        ),
        trainer=Trainer.default_config().add(
            batch_size=32,
            learning_rate=3e-3,  # faster but bounces around
            # max_iters=int(1e6),
            max_iters=50000,
            # TODO:
            weight_decay=0.3,
            device='mps',
        ),
    )

    # name = None
    name = "4 layers and more dropout"
    run(config, name=name)
