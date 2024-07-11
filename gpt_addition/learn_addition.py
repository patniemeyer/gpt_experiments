from gpt_addition import GPTAdditionDataset
from shared.gpt_model import GPT
from shared.train import Trainer
from shared.util import *
import wandb


# -----------------------------------------------------------------------------
# Learn Addition
# -----------------------------------------------------------------------------

def run(config: Config, online: bool = False, name: str = None):
    seed_everything(1337 + 42)

    # Construct train and test datasets
    train_dataset, test_dataset = GPTAdditionDataset(config.data).split()

    # Show some data
    print("Train dataset size:", len(train_dataset))
    # for i in range(50):
    #     pair = train_dataset[i]
    #     print(GPTAdditionDataset.decode(pair[0]), GPTAdditionDataset.decode(pair[1]))
    # exit()

    # Construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()
    print(config, '\n')

    model = GPT(config.model)

    # WandB setup
    wandb.init(
        entity="patniemeyer",
        project="transformer-addition-project",
        group="gpt-addition",
        config=config,
        mode="online" if online else "disabled",
        name=name,
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
                GPTAdditionDataset.evaluate_model(model, train_dataset, max_batches=10)
                correct, count = GPTAdditionDataset.evaluate_model(model, test_dataset)
                # if correct >= count * 0.99:
                # if correct == count:
                if trainer.epoch_num >= 4:
                    print(f"Stopping at {trainer.iter_num} iterations, {trainer.epoch_num} epochs.")
                    trainer.exit = True
                # measure perf per digit
                # from run_saved_model import runtest
                # runtest(model, config, context_len=config.model.context_len, start=1, upto=5)
            model.train()  # revert model to training mode

            print("Saving model...")
            torch.save(model.state_dict(), "gpt_addition.pth")

    trainer.set_callback('on_batch_end', batch_end_callback)

    # Begin training!
    trainer.run()


config = Config(
    model=GPT.default_config().add(
        n_layer=2,
        # ignore_index=GPTAdditionDataset.ignore_token_id,
        ignore_index=None,
        # TODO
        # embd_pdrop=0.15,
        # block_pdrop=0.15,
    ),
    data=GPTAdditionDataset.default_config().add(
        digits=2,
        num_samples=int(1e5),
        # context_len=10,  # long enough for 2 digits
        # context_len=13,  # long enough for 3 digits
        # context_len=32,  # long enough for 4 digits
    ),
    trainer=Trainer.default_config().add(
        batch_size=32,
        learning_rate=3e-3,  # faster but bounces around
        # TODO
        # weight_decay=0.3,
    ),
)

if __name__ == '__main__':
    # name = None
    name = "4 layers and smaller context"
    run(config, name=name, online=False)
