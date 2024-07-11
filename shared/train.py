import time
from collections import defaultdict
import tqdm.auto as tqdm
import wandb
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from .util import *


class Trainer:

    @staticmethod
    def default_config():
        return Config(
            device='auto',
            num_workers=1,  # Dataloader
            # optimizer parameters
            max_iters=None,
            batch_size=100,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            # betas=(0.9, 0.999),
            weight_decay=0.1,  # Only applied to matrix weights
            # TODO: describe
            grad_norm_clip=1.0,
        )

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging
        # and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = 0
        self.exit = False

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # Set up Adam with weight decay groups determined by the model
        self.optimizer = torch.optim.AdamW(
            model.get_weight_decay_groups(config), lr=config.learning_rate, betas=config.betas)
        # Plain Adam with no weight decay
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
        # SGD, needs a much higher learning rate
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        # Experiment: weight rarer samples
        # weights = [sequence.eq(12).sum().float() for sequence, _ in self.train_dataset]

        # set up the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True),
            # sampler=WeightedRandomSampler(weights, replacement=True, num_samples=len(self.train_dataset)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.epoch_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        batches = tqdm.tqdm(range(config.max_iters or int(1e6)), desc="Training")
        for _ in batches:
            if self.exit:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                self.epoch_num += 1
                # print(f"\nepoch {self.epoch_num}")
                batch = next(data_iter)

            # Set up the batch
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # Experiment: Ignore all but the last token
            # y[:, :-1] = -1

            # Forward the model
            logits, self.loss = model(x, y)
            batches.set_postfix_str(f"Loss: {self.loss:.4f}, Epoch: {self.epoch_num}", refresh=True)
            wandb.log({"loss": self.loss, })

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            # TODO: describe
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if (config.max_iters is not None and self.iter_num >=
                    config.max_iters):
                break
