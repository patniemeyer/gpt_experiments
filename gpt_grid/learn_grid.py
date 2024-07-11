from grid_dataset import GridDataset
from shared.gpt_model import GPT
from shared.train import Trainer
from shared.util import *
import wandb


# -----------------------------------------------------------------------------
# Learn Grid Descriptions
# -----------------------------------------------------------------------------

def run(config: Config, use_wandb: bool = False, name: str = None):
    seed_everything(1337 + 42)

    # Construct train and test datasets
    train_dataset, test_dataset = GridDataset(config.data).split(test_max=2000)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # Construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()
    print(config, '\n')
    model = GPT(config.model)

    # WandB setup
    wandb.init(
        entity="patniemeyer",
        project="transformer-grid-project",
        group="grid-descrdiption",
        config=config,
        mode="online" if use_wandb else "disabled",
        name=name,
    )
    wandb.watch(model, log="all")

    # Construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # Training callback
    def batch_end_callback(trainer: Trainer):
        if trainer.iter_num % 1000 == 0 and trainer.iter_num > 0:
            # Evaluate both the train and test score
            model.eval() # set model to no gradients
            with torch.no_grad():
                # print("\nEvaluate on training set:")
                # GridDataset.evaluate_model(model, train_dataset, max_batches=10)
                print("\nEvaluate on test set:")
                correct, count = GridDataset.evaluate_model(model, test_dataset)
                # Stopping conditions
                # if correct == count:
                # if trainer.epoch_num >= 4:
                if correct >= count * 0.99:
                    print(f"Stopping at {trainer.iter_num} iterations, {trainer.epoch_num} epochs.")
                    trainer.exit = True
            model.train()  # revert model to training mode

            print("Saving model...")
            torch.save(model.state_dict(), "grid.pth")

    trainer.set_callback('on_batch_end', batch_end_callback)

    # Begin training!
    trainer.run()


config = Config(
    model=GPT.default_config().add(
        # n_layer=1,
        n_layer=3,
        # embd_pdrop=0.15,
        # block_pdrop=0.15,
    ),
    data=GridDataset.default_config().add(
        num_samples=int(1e5),
        colors_min=1,
        colors_max=4,
        filled_min=1,
        filled_max=8,
        grid_rows_min=2,
        grid_rows_max=8,
        grid_cols_min=2,
        grid_cols_max=8,
        fixed_length_output=False,
    ),
    trainer=Trainer.default_config().add(
        batch_size=32,
        learning_rate=1e-3,  # faster but bounces around
        device="mps"  # "cuda" or "cpu"
        # weight_decay=0.3,
    ),
)

if __name__ == '__main__':
    # name = None
    name = f"{config.model.n_layer} layers"
    run(config, name=name, use_wandb=True)
