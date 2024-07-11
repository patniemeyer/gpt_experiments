import math
import torch.nn
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F
from .util import *


class GPT(nn.Module):
    @staticmethod
    def default_config():
        return Config(
            n_layer=2,
            n_head=3,
            n_embd=48,

            # Determined by the dataset
            vocab_size=None,

            # Dropout:
            # The harder the problem the more these help with generalization.
            # 0.1 works well, .05 ramps slower but is overall better
            #
            # The token embedding dropout, applied once after positional embeddings are added.
            embd_pdrop=0.05,
            # The transformer block dropout, applied in each block for:
            #   self attention, feed forward, and post-residual.
            block_pdrop=0.05,

            ignore_index=None,

            pos_embedding="learned",  # "learned" or "sin"
            reverse_pos_enc=False,  # for sin
        )

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.ignore_index = config.ignore_index

        assert all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])

        # Positional embedding
        if config.pos_embedding == "sin":
            pos_embedding = SinPositionalEncoding(config.n_embd, dropout=config.embd_pdrop,
                                                  reverse=config.reverse_pos_enc)
        else:
            pos_embedding = LearnedPositionalEmbedding(config.n_embd, dropout=config.embd_pdrop,
                                                       max_len=config.context_len)

        self.transformer = nn.ModuleDict(dict(
            tok_embedding=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.ignore_index),
            pos_embedding=pos_embedding,
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self.init_weights)

        for pn, p in self.named_parameters():
            # print("pn:", pn)
            if pn.endswith('c_proj.weight') or pn.endswith('linear2.weight'):
                # print("special init for ", pn)
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder
        # parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        total = 0
        for name, param in self.transformer.named_parameters(recurse=True):
            # if not name.startswith('h.0'): continue
            # print(name, param.shape)
            total += param.numel()
            # print("add: ", int(param.numel())/1e6, "total: ", int(total/1e6))
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    # Init weights as Karpathy does, including scaled init to the residual projections, per GPT-2 paper
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.5)
            # bias is initialized to zero by default
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.5)

    # Setup weight decay as in Karpathy's minGPT
    def get_weight_decay_groups(self, config):
        decay = set()
        no_decay = set()

        decay_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        nodecay_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_name = f'{module_name}.{param_name}' if module_name else param_name
                # Don't decay any bias terms
                if param_name.endswith('bias'):
                    no_decay.add(full_name)
                elif param_name.endswith('weight') and isinstance(module, decay_weight_modules):
                    decay.add(full_name)
                elif param_name.endswith('weight') and isinstance(module, nodecay_weight_modules):
                    no_decay.add(full_name)

        # Check that we have considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        conflict_set = decay & no_decay
        all_set = decay | no_decay
        assert not conflict_set, f"parameters {str(conflict_set)} are in both decay/no_decay sets!"
        assert param_dict.keys() == all_set, \
            f"parameters {str(param_dict.keys() - all_set)} were not assigned to decay/no_decay sets!"

        return [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

    # The main GPT forward pass
    def forward(self, input: torch.Tensor, target: torch.Tensor = None):
        b, t = input.size()

        assert b != 0, "Empty batch"

        if self.transformer.pos_embedding is not SinPositionalEncoding:
            assert t <= self.transformer.pos_embedding.max_len, (
                f"Cannot forward sequence of length {t}, max len is {self.transformer.pos_embedding.max_len}")

        input_mask = None
        if self.ignore_index is not None:
            input_mask = (input != self.ignore_index)  # False (ignore) for padding
        tok_emb = self.transformer.tok_embedding(input)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.pos_embedding(tok_emb)
        for block in self.transformer.h:
            x = block(x, input_mask=input_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # TODO: Experiment - show logits
        # showlogits = torch.argmax(logits, dim=-1)
        # from gpt_addition import GPTAdditionDataset
        # print("\ninput: ", GPTAdditionDataset.decode(input[0]))
        # print("logits:     ", GPTAdditionDataset.decode(showlogits[0]))

        # If we are training calculate the loss
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),
                                   ignore_index=self.ignore_index or -100)

        return logits, loss

    @torch.no_grad()
    def generate(self, input: torch.Tensor, end_of_sequence_token_id: int,
                 max_gen_token_count: int = 32,
                 maintain_len: bool = True,
                 temperature: float = 1.0,
                 do_sampling: bool = False, top_k: bool = None) -> (list, list):

        input = input.clone()  # TODO
        assert input.size(0) != 0, "Empty batch"

        org_len = input.size(1)
        completed_sequences = []
        completed_indices = []
        orig_indices = torch.arange(input.size(0)).to(input.device)

        for _ in range(max_gen_token_count):
            # Forward the model. (logits, loss)
            logits, _ = self.forward(input)

            # Our prediction of the next token is based solely on the logits of the final hidden state token.
            logits = logits[:, -1, :] / temperature  # (b, t, logits) -> (b, 1, logits)

            # Optionally trim the logits to top k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax converts the logits to normalized probabilities.
            probs = F.softmax(logits, dim=-1)

            # Either sample from the distribution or take the most likely element
            if do_sampling:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            # Append the new token to the sequence (copying the original)
            input = torch.cat((input, next_token), dim=1)

            # Shift and maintain the block size as we append the new tokens
            if maintain_len:
                input = input if input.size(1) <= org_len else input[:, -org_len:]

            # Determined which sequences are finished and move them to the completed list
            completed = input[:, -1] == end_of_sequence_token_id
            completed_sequences.extend(input[completed])
            completed_indices.extend(orig_indices[completed])
            input, orig_indices = input[~completed], orig_indices[~completed]

            if len(input) == 0:
                break

        completed_sequences.extend(input)  # add any remaining sequences
        completed_indices.extend(orig_indices)

        return completed_sequences, completed_indices


# Sinusoidal positional encodings
# TODO: Reverse is prob not what we want here... the ends have different frequency ranges.
class SinPositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 256, reverse: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.reverse = reverse

        pe = torch.zeros(1, max_len, n_embd)
        position = torch.arange(max_len).unsqueeze(1)  # [[0], [1], [2], ... [max_len-1]
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))  # n_embd // 2 div terms
        pe[0, :, 0::2] = torch.sin(position * div_term)  # even pos sin
        pe[0, :, 1::2] = torch.cos(position * div_term)  # odd pos cos
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            # Since we ignore padding on the left we reverse the positional encoding (slicing from
            # the end) so that the meaningful tokens have consistent positional encoding.
            # x = x + self.pe[:, -x.size(1):, :]  # [batch_size, seq_len, embedding_dim]
            # Alternately we could slice from the beginning as usual but flip the encoding:
            x = x + torch.flip(self.pe[:, :x.size(1), :], [1])  # [batch_size, seq_len, embedding_dim]
        else:
            # Standard:
            x = x + self.pe[:, :x.size(1), :]  # [batch_size, seq_len, embedding_dim]
        return self.dropout(x)

    # tostring
    def __str__(self):
        return f"SinPositionalEncoding(max_len={self.max_len}, reverse={self.reverse})"


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 256):
        super(LearnedPositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(self.max_len, n_embd)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        b, t, _ = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.embedding(pos)
        x = self.dropout(x + pos_emb)
        return x


# TransformerEncoderLayer that matches the implementation in min-mingpt-adder.py
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.block_pdrop,
            activation="gelu",  # see newGelu, slightly better
            norm_first=True,  # critical
            batch_first=True,
        )
        self.nhead = config.n_head

    def forward(self, x, input_mask=None):
        b, t, _ = x.shape
        assert b != 0, "Empty batch"

        # Create the causal mask (b * num_heads, t, t)
        mininf = -1e9
        causal_mask = torch.zeros((t, t))
        causal_mask += torch.triu(torch.ones(t, t) * mininf, diagonal=1)  # (t,t)
        causal_mask = causal_mask.unsqueeze(0)  # (1, t, t)
        causal_mask = causal_mask.expand(b, -1, -1)  # (b, t, t)
        causal_mask = causal_mask.repeat_interleave(self.nhead, dim=0)  # (b * num_heads, t, t)
        mask = causal_mask

        # Same as:
        # mask = torch.nn.Transformer.generate_square_subsequent_mask(t) # (t,t)
        # mask = mask.unsqueeze(0)  # (1, t, t)
        # mask = mask.expand(b, -1, -1)  # (b, t, t)
        # mask = mask.repeat_interleave(self.nhead, dim=0)  # (b * num_heads, t, t)

        # Create the padding mask
        if input_mask is not None:
            # I am not confident that padding masking is working. Please see test_mask.py
            pad_mask = input_mask.unsqueeze(1)  # (b, 1, t)
            pad_mask = pad_mask.expand(-1, t, -1)  # (b, t, t) replicating the t seq
            pad_mask = pad_mask.repeat_interleave(self.nhead, dim=0)
            # Combine the masks
            mask = causal_mask + pad_mask

        mask = mask.to(x.device)
        # Note: is_causal must be False for our padding mask to work (else replaced with its own mask)
        x = self.transformer_layer(x, src_mask=mask, is_causal=False)
        return x
