import re

from shared.gpt_model import GPT
from shared.train import Trainer
from shared.util import *
from tiny_stories_data import TinyStoriesDataset
import wandb
from icecream import ic

config = Config(
    model=GPT.default_config().add(
        n_layer=3,
        n_head=16,
        n_embd=512,
        ignore_index=None
    ),
    data=TinyStoriesDataset.default_config().add(
        context_len=128,
    ),
    trainer=Trainer.default_config().add(
        batch_size=16,
        learning_rate=3e-3,  # faster but bounces around
        max_iters=int(1e6),
        device='mps',
    ),
)

if __name__ == '__main__':

    seed_everything(1337 + 42)

    # Construct train and test datasets
    train_dataset = TinyStoriesDataset(config.data, split='train')

    # Construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.context_len = train_dataset.get_context_length()
    ic(config, '\n')

    model = GPT(config.model)

    if re.search(r"SinPositional.*reverse=True", str(model.transformer.pos_embedding)):
        print("WARNING: SinPosition reversed!")

    # WandB setup
    wandb.init(
        entity="patniemeyer",
        project="transformer-tinystories-project",
        # group="gpt-addition",
        config=config,
        # mode="online",
        mode="disabled",
    )
    wandb.watch(model, log="all")

    # Construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)


    # Training callback
    def batch_end_callback(trainer: Trainer):
        if trainer.iter_num % 500 == 0 and trainer.iter_num > 0:
            print("Saving model...")
            torch.save(model.state_dict(), "tiny_stories.pth")


    trainer.set_callback('on_batch_end', batch_end_callback)

    # Begin training!
    trainer.run()
