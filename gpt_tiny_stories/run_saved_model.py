import wandb
from tiny_stories_data import TinyStoriesDataset as Stories
from shared.gpt_model import GPT
from shared.util import *

wandb.init(mode="disabled")
seed_everything(1337+3)

from learn_tiny_stories import config

device = "cpu"
# TODO: We are just doing this to init the config
dataset = Stories(config.data, split='train')
config.model.vocab_size = dataset.get_vocab_size()
config.model.context_len = dataset.get_context_length()

model = GPT(config.model).to(device)
model.load_state_dict(torch.load("tiny_stories.pth"))
# model.load_state_dict(torch.load("tiny_stories-3-16-512-57500.pth"))
model.eval()

# sample = "Lily wanted to get a cat or a dog.  Her mom said that she could not have a cat, so she"
# sample = "Lily wanted to get a cat or a dog. "
sample = "Kate"

def runtest():
    input = torch.tensor([Stories.encode(sample)]).to(device)
    model.eval()
    with torch.no_grad():
        output_batch, _ = model.generate(
            input=input,
            end_of_sequence_token_id=Stories.eot_token_id,
            maintain_len=False,
            max_gen_token_count=128,
            do_sampling=False)
        output = Stories.decode(output_batch[0])
        print(output)


runtest()
